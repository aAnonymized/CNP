from __future__ import annotations

"""
Gradient-Trail Spurious Concept Discovery on ResNet layer4 spatial tokens
+ Top-rel_num protection
+ Wrong-Class Positive Suppression (WPS)

Key design for this version
---------------------------
1) NO input-image patch cropping.
2) The spatial units are ALWAYS the final layer4 feature-map tokens:
       [B, D, H, W] -> [B, H*W, D]
   For ResNet50/224x224 this is normally [B, 49, 2048].
3) Stage 1 learns class-specific non-negative concept banks from layer4 tokens.
4) Gradient probes are applied directly to the detached layer4 feature map.
5) FN/FP gradient trails rank candidate spurious concepts.
6) Stage 2 uses a frozen Stage-1 model to localize which layer4 tokens are
   dominated by a discovered spurious concept.
7) Top-rel_num GT-related current-student tokens have absolute priority and are
   removed from the nuisance mask.
8) Only positive WRONG-class evidence of the remaining nuisance tokens is
   suppressed. GT evidence is never directly penalized by WPS.

Dependencies:
    torch, numpy, scipy, scikit-learn, matplotlib (only for visualization)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import math
import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Basic model helpers
# =============================================================================

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _unwrap_tensor(x, name: str) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0 and torch.is_tensor(x[0]):
        return x[0]
    raise TypeError(
        f"{name} must be a Tensor or tuple/list whose first item is a Tensor."
    )


def _get_classifier(model: nn.Module) -> nn.Linear:
    """Locate the final nn.Linear classifier of a ResNet-like model."""
    base = _unwrap_model(model)
    candidates = []

    if hasattr(base, "get_classifier"):
        try:
            candidates.append(base.get_classifier())
        except Exception:
            pass

    for name in ("fc", "head", "classifier"):
        if hasattr(base, name):
            candidates.append(getattr(base, name))

    for candidate in candidates:
        if isinstance(candidate, nn.Linear):
            return candidate
        if isinstance(candidate, nn.Sequential):
            for sub in reversed(candidate):
                if isinstance(sub, nn.Linear):
                    return sub

    raise AttributeError("Unable to locate final nn.Linear classifier.")


def _capture_layer4_and_forward(
    model: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a normal forward pass and capture final layer4 output.

    Returns:
        logits:   [B,C]
        features: [B,D,H,W]
    """
    base = _unwrap_model(model)
    if not hasattr(base, "layer4"):
        raise AttributeError(
            "The model must be ResNet-like and contain model.layer4."
        )

    holder: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        holder["features"] = _unwrap_tensor(output, "layer4 output")

    handle = base.layer4.register_forward_hook(hook)
    try:
        logits = _unwrap_tensor(model(inputs), "model(inputs)")
    finally:
        handle.remove()

    if "features" not in holder:
        raise RuntimeError("Failed to capture layer4 features.")

    features = holder["features"]
    if features.ndim != 4:
        raise ValueError(
            f"layer4 must be [B,D,H,W], got {tuple(features.shape)}"
        )
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")

    return logits, features


def _regions_from_features(features: torch.Tensor) -> torch.Tensor:
    """
    Directly use final layer4 spatial positions as patches/regions.

        [B,D,H,W] -> [B,H*W,D]
    """
    if features.ndim != 4:
        raise ValueError("features must be [B,D,H,W].")
    return features.flatten(2).transpose(1, 2)


def _gap(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError("features must be [B,D,H,W].")
    return features.mean(dim=(2, 3))


# =============================================================================
# NNLS helpers
# =============================================================================

def _nnls_single(a: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Exact CPU NNLS used for Stage-1 gradient-trail scoring.

        min_{u >= 0} ||a - u @ components||_2

    Args:
        a:          [D]
        components: [K,D]
    Returns:
        u:          [K]
    """
    u, _ = nnls(
        components.T.astype(np.float64),
        a.astype(np.float64),
    )
    return u.astype(np.float32)


def _batch_projected_nnls_torch(
    x: torch.Tensor,
    components: torch.Tensor,
    steps: int = 40,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GPU-friendly approximate NNLS for dense layer4 token localization.

        min_{U >= 0} ||X - U H||_F^2

    Args:
        x:          [N,D]
        components: [K,D]
    Returns:
        u:          [N,K]

    Stage-1 scoring uses exact scipy NNLS. This approximation is only used for
    dense [B*R] spatial token localization during Stage 2 / evaluation.
    """
    if x.ndim != 2 or components.ndim != 2:
        raise ValueError("x and components must both be rank-2 tensors.")
    if x.shape[1] != components.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: x={x.shape[1]}, components={components.shape[1]}."
        )

    x = x.clamp_min(0)
    h = components.clamp_min(0)

    # Positive-correlation initialization.
    denom = h.pow(2).sum(dim=1).clamp_min(eps)  # [K]
    u = ((x @ h.t()) / denom.unsqueeze(0)).clamp_min(0)

    # Fixed projected-gradient step using ||H H^T||_2.
    gram = h @ h.t()
    try:
        lipschitz = torch.linalg.eigvalsh(gram.float()).max().to(x.dtype)
    except Exception:
        lipschitz = gram.abs().sum(dim=1).max()
    lr = 1.0 / lipschitz.clamp_min(eps)

    for _ in range(int(steps)):
        grad = (u @ h - x) @ h.t()
        u = (u - lr * grad).clamp_min(0)

    return u


# =============================================================================
# Outputs
# =============================================================================

@dataclass
class GradientTrailDiscoveryResult:
    scores: Dict[int, np.ndarray]
    e_fn: Dict[int, np.ndarray]
    e_fp: Dict[int, np.ndarray]
    fn_counts: Dict[int, int]
    fp_counts: Dict[int, int]
    ranked_concepts: Dict[int, List[int]]
    bias_concepts: Dict[int, List[int]]
    probe_relative_change_mean: float

    def summary(self) -> Dict[str, object]:
        return {
            "fn_counts": self.fn_counts,
            "fp_counts": self.fp_counts,
            "ranked_concepts": self.ranked_concepts,
            "bias_concepts": self.bias_concepts,
            "probe_relative_change_mean": self.probe_relative_change_mean,
        }


@dataclass
class GradientTrailOutput:
    loss_region: torch.Tensor
    loss_wps: torch.Tensor
    logits: torch.Tensor

    # Masks are [B,R]
    raw_spurious_mask: torch.Tensor
    spurious_mask: torch.Tensor
    priority_relevant_mask: torch.Tensor
    relevant_mask: torch.Tensor

    # Concept localization diagnostics, [B,R]
    dominant_concept_ids: torch.Tensor
    bias_score: torch.Tensor

    num_raw_spurious_regions: int
    num_protected_regions: int
    num_spurious_regions: int
    num_relevant_regions: int
    num_valid_images: int

    # Compatibility aliases with the user's existing region-loss logging.
    @property
    def loss_R(self) -> torch.Tensor:
        return self.loss_wps.new_zeros(())

    @property
    def loss_A(self) -> torch.Tensor:
        return self.loss_wps.new_zeros(())

    def statistics(self) -> Dict[str, float]:
        return {
            "loss_region": float(self.loss_region.detach().item()),
            "loss_wps": float(self.loss_wps.detach().item()),
            "loss_R": 0.0,
            "loss_A": 0.0,
            "num_raw_spurious_regions": float(self.num_raw_spurious_regions),
            "num_protected_regions": float(self.num_protected_regions),
            "num_spurious_regions": float(self.num_spurious_regions),
            "num_relevant_regions": float(self.num_relevant_regions),
            "num_valid_images": float(self.num_valid_images),
        }


# =============================================================================
# Main module
# =============================================================================

class GradientTrailResNet(object):
    """
    Layer4-only Gradient-Trail discovery + patch localization + WPS.

    IMPORTANT:
    ----------
    This implementation deliberately DOES NOT crop/rescale input images.
    Every patch is a spatial token from the final layer4 feature map.

    Stage 1 (frozen discovery model)
    --------------------------------
      1) Extract final layer4 tokens [B,R,D].
      2) For each predicted class y, collect those nonnegative tokens.
      3) Fit an NMF concept bank C_y in R^D.
      4) On classification errors, directly probe the layer4 feature map:

            F' = ReLU(F - probe_step * d CE / dF)

      5) GAP(F) and GAP(F') are decomposed with exact NNLS in the true-class
         bank (FN trail) and predicted-class bank (FP trail).
      6) Score each concept with

            S_yk = 0.5 * (E_FN_yk + E_FP_yk).

      7) Keep concepts with S_yk > bias_threshold, optionally capped by
         max_bias_concepts_per_class.

    Stage 2 (student training)
    --------------------------
      1) Current student layer4 tokens are used for WPS gradients.
      2) A frozen Stage-1 snapshot produces stable layer4 tokens for concept
         localization.
      3) Each frozen token is decomposed by the GT-class concept bank.
      4) A token is raw-spurious iff its DOMINANT concept is one of that class's
         discovered bias concepts. No extra patch threshold is introduced.
      5) Current-student Top-rel_num tokens most aligned with w_y are protected.
      6) On remaining nuisance tokens, suppress only strongest POSITIVE
         wrong-class cosine evidence:

            max_{c != y} ReLU(cos(z, w_c))^2.

         GT-class evidence is not directly penalized.

    No learnable parameter is introduced by this object.
    """

    def __init__(
        self,
        num_classes: int,
        num_concepts: int = 10,
        probe_step: float = 2.0e4,
        bias_threshold: float = 0.55,
        max_bias_concepts_per_class: Optional[int] = 1,
        rel_num: int = 4,
        max_tokens_per_class: Optional[int] = 3000,
        nmf_max_iter: int = 500,
        localization_nnls_steps: int = 40,
        lambda_region: float = 0.1,
        random_seed: int = 0,
        eps: float = 1e-8,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if num_concepts < 2:
            raise ValueError("num_concepts must be >= 2.")
        if probe_step <= 0:
            raise ValueError("probe_step must be > 0.")
        if not (-1.0 <= bias_threshold <= 1.0):
            raise ValueError("bias_threshold must lie in [-1,1].")
        if max_bias_concepts_per_class is not None and max_bias_concepts_per_class < 1:
            raise ValueError("max_bias_concepts_per_class must be None or >=1.")
        if rel_num < 0:
            raise ValueError("rel_num must be >=0.")
        if max_tokens_per_class is not None and max_tokens_per_class < 2:
            raise ValueError("max_tokens_per_class must be None or >=2.")
        if nmf_max_iter < 1:
            raise ValueError("nmf_max_iter must be >=1.")
        if localization_nnls_steps < 1:
            raise ValueError("localization_nnls_steps must be >=1.")
        if lambda_region < 0:
            raise ValueError("lambda_region must be >=0.")

        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.probe_step = float(probe_step)
        self.bias_threshold = float(bias_threshold)
        self.max_bias_concepts_per_class = max_bias_concepts_per_class
        self.rel_num = int(rel_num)
        self.max_tokens_per_class = max_tokens_per_class
        self.nmf_max_iter = int(nmf_max_iter)
        self.localization_nnls_steps = int(localization_nnls_steps)
        self.lambda_region = float(lambda_region)
        self.random_seed = int(random_seed)
        self.eps = float(eps)

        # class y -> [K_y,D], CPU float32, nonnegative and L2-normalized rows.
        self.concept_banks: Dict[int, torch.Tensor] = {}

        self.scores: Dict[int, np.ndarray] = {}
        self.e_fn: Dict[int, np.ndarray] = {}
        self.e_fp: Dict[int, np.ndarray] = {}
        self.ranked_concepts: Dict[int, List[int]] = {}
        self.bias_concepts: Dict[int, List[int]] = {}
        self.discovery_result: Optional[GradientTrailDiscoveryResult] = None

        # Frozen Stage-1 model used ONLY to create stable Stage-2 masks.
        self._assignment_model: Optional[nn.Module] = None
        self._assignment_device = None

    # -------------------------------------------------------------------------
    # Layer4 regions
    # -------------------------------------------------------------------------
    def _regions_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return _regions_from_features(features)

    # -------------------------------------------------------------------------
    # Stage 1A: build class-specific NMF banks from layer4 TOKENS
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def fit_concept_banks(
        self,
        model: nn.Module,
        audit_loader,
        device=None,
        verbose: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Collect ONLY final-layer4 spatial tokens; no input cropping is used.

        Tokens are grouped by the image's PREDICTED class, then an NMF concept
        bank is fitted independently for every predicted class.
        """
        base = _unwrap_model(model)
        if device is None:
            device = next(base.parameters()).device

        was_training = base.training
        base.eval()

        # Streaming random-key sampling avoids first-batch bias while obeying cap.
        token_pool: Dict[int, Optional[torch.Tensor]] = {
            y: None for y in range(self.num_classes)
        }
        key_pool: Dict[int, Optional[torch.Tensor]] = {
            y: None for y in range(self.num_classes)
        }
        seen_tokens = {y: 0 for y in range(self.num_classes)}

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.random_seed)

        try:
            for batch in audit_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError(
                        "audit_loader must return (inputs, labels, ...) even though "
                        "labels are not used while fitting the concept bank."
                    )

                inputs = batch[0]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]
                inputs = inputs.to(device, non_blocking=True)

                logits, features = _capture_layer4_and_forward(model, inputs)
                preds = logits.argmax(dim=1)
                regions = self._regions_from_features(features).clamp_min(0)
                # [B,R,D]

                for y in range(self.num_classes):
                    image_mask = preds.eq(y)
                    if not bool(image_mask.any().item()):
                        continue

                    z = (
                        regions[image_mask]
                        .reshape(-1, regions.shape[-1])
                        .detach()
                        .cpu()
                        .float()
                    )
                    if z.numel() == 0:
                        continue

                    seen_tokens[y] += int(z.shape[0])
                    keys = torch.rand(z.shape[0], generator=gen)

                    if token_pool[y] is None:
                        merged_z = z
                        merged_keys = keys
                    else:
                        merged_z = torch.cat([token_pool[y], z], dim=0)
                        merged_keys = torch.cat([key_pool[y], keys], dim=0)

                    if (
                        self.max_tokens_per_class is not None
                        and merged_z.shape[0] > int(self.max_tokens_per_class)
                    ):
                        keep = torch.topk(
                            merged_keys,
                            k=int(self.max_tokens_per_class),
                            largest=True,
                            sorted=False,
                        ).indices
                        merged_z = merged_z.index_select(0, keep)
                        merged_keys = merged_keys.index_select(0, keep)

                    token_pool[y] = merged_z
                    key_pool[y] = merged_keys

            self.concept_banks = {}

            if verbose:
                print("========== Gradient Trail: layer4 token concept banks ==========")
                print("NO input patch cropping; regions are final layer4 spatial tokens.")

            for y in range(self.num_classes):
                z = token_pool[y]
                if z is None or z.shape[0] < 2:
                    warnings.warn(
                        f"Class {y}: insufficient predicted-class layer4 tokens; bank skipped."
                    )
                    continue

                z = z.clamp_min(0)
                n, d = z.shape
                r_y = min(self.num_concepts, int(n), int(d))
                if r_y < 2:
                    warnings.warn(f"Class {y}: NMF rank <2; bank skipped.")
                    continue

                nmf = NMF(
                    n_components=r_y,
                    init="nndsvda",
                    solver="cd",
                    beta_loss="frobenius",
                    max_iter=self.nmf_max_iter,
                    random_state=self.random_seed + y,
                )
                nmf.fit(z.numpy())

                components = torch.from_numpy(
                    nmf.components_.astype(np.float32)
                ).clamp_min(0)

                # NMF scale is ambiguous. Unit-normalize concept directions so
                # dense coefficients are more comparable across concepts.
                components = F.normalize(
                    components,
                    p=2,
                    dim=1,
                    eps=self.eps,
                )

                self.concept_banks[y] = components.cpu()

                if verbose:
                    print(
                        f"class {y}: seen_tokens={seen_tokens[y]} | "
                        f"used_tokens={n} | concepts={r_y} | "
                        f"recon_err={float(nmf.reconstruction_err_):.6f}"
                    )

            if len(self.concept_banks) == 0:
                raise RuntimeError("No concept bank could be constructed.")

            return self.concept_banks

        finally:
            if was_training:
                base.train()
            else:
                base.eval()

    # -------------------------------------------------------------------------
    # Stage 1B: direct layer4 gradient probe
    # -------------------------------------------------------------------------
    def _probe_layer4_batch(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Directly probe the final layer4 feature map.

        No model parameter is updated and no gradient is backpropagated through
        the encoder. We detach F, make only F require gradients, and use the
        standard ResNet GAP + linear classifier head.

        Returns:
            preds:      [B]
            a:          [B,D] = GAP(F)
            a_prime:    [B,D] = GAP(F')
            rel_change: [B]   = ||F'-F|| / ||F||
        """
        classifier = _get_classifier(model)

        with torch.no_grad():
            logits_full, features_full = _capture_layer4_and_forward(model, inputs)
            preds = logits_full.argmax(dim=1)
            features = features_full.detach().clamp_min(0)

        if classifier.weight.shape[1] != features.shape[1]:
            raise ValueError(
                "Classifier input dim {} != layer4 channel dim {}.".format(
                    classifier.weight.shape[1], features.shape[1]
                )
            )

        f_probe = features.clone().requires_grad_(True)
        a_probe = _gap(f_probe)
        probe_logits = classifier(a_probe)
        loss = F.cross_entropy(probe_logits, labels, reduction="sum")

        grad_f = torch.autograd.grad(
            loss,
            f_probe,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]

        f_prime = (
            f_probe - self.probe_step * grad_f
        ).detach().clamp_min(0)

        a = _gap(features).detach().clamp_min(0)
        a_prime = _gap(f_prime).detach().clamp_min(0)

        delta = (f_prime - features).flatten(1).norm(p=2, dim=1)
        base_norm = features.flatten(1).norm(p=2, dim=1).clamp_min(self.eps)
        rel_change = delta / base_norm

        return preds.detach(), a, a_prime, rel_change.detach()

    def score_bias_concepts(
        self,
        model: nn.Module,
        audit_loader,
        device=None,
        verbose: bool = True,
    ) -> GradientTrailDiscoveryResult:
        """
        Gradient-trail score from FN and FP errors.

        FN_y = {GT=y, pred!=y}
        FP_y = {pred=y, GT!=y}

        Exact NNLS is used on GAP(F) and GAP(F') with the class-specific concept
        bank learned from layer4 tokens.
        """
        if len(self.concept_banks) == 0:
            raise RuntimeError(
                "Run fit_concept_banks(...) before score_bias_concepts(...)."
            )

        base = _unwrap_model(model)
        if device is None:
            device = next(base.parameters()).device

        was_training = base.training
        base.eval()

        fn_sum: Dict[int, np.ndarray] = {}
        fp_sum: Dict[int, np.ndarray] = {}
        fn_count = {y: 0 for y in range(self.num_classes)}
        fp_count = {y: 0 for y in range(self.num_classes)}

        for y, bank in self.concept_banks.items():
            k_y = int(bank.shape[0])
            fn_sum[y] = np.zeros(k_y, dtype=np.float64)
            fp_sum[y] = np.zeros(k_y, dtype=np.float64)

        rel_change_sum = 0.0
        rel_change_count = 0

        try:
            for batch in audit_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError("audit_loader must return (inputs, labels, ...).")

                inputs = batch[0]
                labels = batch[1]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                preds, a, a_prime, rel_change = self._probe_layer4_batch(
                    model,
                    inputs,
                    labels,
                )

                rel_change_sum += float(rel_change.sum().item())
                rel_change_count += int(rel_change.numel())

                a_np = a.detach().cpu().float().numpy()
                ap_np = a_prime.detach().cpu().float().numpy()
                labels_np = labels.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()

                for i in range(inputs.shape[0]):
                    yi = int(labels_np[i])
                    pi = int(preds_np[i])
                    if yi == pi:
                        continue

                    # FN trail for the true class.
                    if yi in self.concept_banks:
                        bank_y = self.concept_banks[yi].numpy()
                        u0 = _nnls_single(a_np[i], bank_y)
                        u1 = _nnls_single(ap_np[i], bank_y)
                        active0 = (u0 > 0.0).astype(np.float64)
                        active1 = (u1 > 0.0).astype(np.float64)
                        fn_sum[yi] += active1 - active0
                        fn_count[yi] += 1

                    # FP trail for the predicted class.
                    if pi in self.concept_banks:
                        bank_p = self.concept_banks[pi].numpy()
                        u0 = _nnls_single(a_np[i], bank_p)
                        u1 = _nnls_single(ap_np[i], bank_p)
                        active0 = (u0 > 0.0).astype(np.float64)
                        active1 = (u1 > 0.0).astype(np.float64)
                        fp_sum[pi] += active0 - active1
                        fp_count[pi] += 1

            scores: Dict[int, np.ndarray] = {}
            e_fn: Dict[int, np.ndarray] = {}
            e_fp: Dict[int, np.ndarray] = {}
            ranked: Dict[int, List[int]] = {}
            bias: Dict[int, List[int]] = {}

            probe_relative_change_mean = (
                rel_change_sum / float(max(rel_change_count, 1))
            )

            if verbose:
                print("========== Gradient Trail: spurious concept scores ==========")
                print(
                    "mean ||F'-F||/||F|| = {:.6f} | probe_step={:.6g}".format(
                        probe_relative_change_mean,
                        self.probe_step,
                    )
                )

            for y, bank in self.concept_banks.items():
                k_y = int(bank.shape[0])

                if fn_count[y] > 0:
                    efn = fn_sum[y] / float(fn_count[y])
                else:
                    efn = np.zeros(k_y, dtype=np.float64)
                    warnings.warn(
                        f"Class {y}: no FN in audit set; E_FN set to zero."
                    )

                if fp_count[y] > 0:
                    efp = fp_sum[y] / float(fp_count[y])
                else:
                    efp = np.zeros(k_y, dtype=np.float64)
                    warnings.warn(
                        f"Class {y}: no FP in audit set; E_FP set to zero."
                    )

                s = 0.5 * (efn + efp)
                order = np.argsort(-s).tolist()

                selected = [
                    int(k)
                    for k in order
                    if float(s[k]) > self.bias_threshold
                ]
                if self.max_bias_concepts_per_class is not None:
                    selected = selected[: int(self.max_bias_concepts_per_class)]

                e_fn[y] = efn.astype(np.float32)
                e_fp[y] = efp.astype(np.float32)
                scores[y] = s.astype(np.float32)
                ranked[y] = [int(k) for k in order]
                bias[y] = selected

                if verbose:
                    top_text = ", ".join(
                        [
                            "k={}:S={:.3f},FN={:.3f},FP={:.3f}".format(
                                k,
                                float(s[k]),
                                float(efn[k]),
                                float(efp[k]),
                            )
                            for k in order[: min(5, len(order))]
                        ]
                    )
                    print(
                        "class {}: FN={} FP={} | selected={} | {}".format(
                            y,
                            fn_count[y],
                            fp_count[y],
                            selected,
                            top_text,
                        )
                    )

            self.scores = scores
            self.e_fn = e_fn
            self.e_fp = e_fp
            self.ranked_concepts = ranked
            self.bias_concepts = bias

            result = GradientTrailDiscoveryResult(
                scores=scores,
                e_fn=e_fn,
                e_fp=e_fp,
                fn_counts=fn_count,
                fp_counts=fp_count,
                ranked_concepts=ranked,
                bias_concepts=bias,
                probe_relative_change_mean=float(probe_relative_change_mean),
            )
            self.discovery_result = result
            return result

        finally:
            if was_training:
                base.train()
            else:
                base.eval()

    def discover(
        self,
        model: nn.Module,
        audit_loader,
        device=None,
        verbose: bool = True,
        make_assignment_snapshot: bool = True,
    ) -> GradientTrailDiscoveryResult:
        """
        One-call Stage-1 discovery.
        """
        self.fit_concept_banks(
            model=model,
            audit_loader=audit_loader,
            device=device,
            verbose=verbose,
        )

        result = self.score_bias_concepts(
            model=model,
            audit_loader=audit_loader,
            device=device,
            verbose=verbose,
        )

        if make_assignment_snapshot:
            self.prepare_stage2_assignment_model(
                model=model,
                device=device,
            )

        if verbose:
            print("========== Gradient Trail: final selected concepts ==========")
            print(self.bias_concepts)
            print("=============================================================")

        return result

    # -------------------------------------------------------------------------
    # Frozen Stage-1 snapshot for stable Stage-2 spatial assignment
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def prepare_stage2_assignment_model(
        self,
        model: nn.Module,
        device=None,
    ) -> None:
        base = _unwrap_model(model)
        if device is None:
            device = next(base.parameters()).device

        frozen = copy.deepcopy(base).to(device)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)

        self._assignment_model = frozen
        self._assignment_device = device

    # -------------------------------------------------------------------------
    # Stage-2 raw spurious token localization
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _localize_from_regions(
        self,
        regions: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dense layer4-token concept decomposition.

        A token is raw-spurious iff its dominant NNLS concept belongs to the
        discovered bias-concept set of the chosen class.

        Args:
            regions:   [B,R,D], nonnegative layer4 tokens from frozen Stage-1 model
            class_ids: [B], normally GT labels during training

        Returns:
            raw_spurious_mask:   [B,R] bool
            dominant_concept_id: [B,R] long, -1 when class bank unavailable
            bias_score:          [B,R] in [0,1], fraction of concept mass assigned
                                 to selected bias concepts (diagnostic only)
        """
        if self.discovery_result is None:
            raise RuntimeError("Run discover(...) before Stage 2 localization.")

        if regions.ndim != 3:
            raise ValueError("regions must be [B,R,D].")

        b, r, d = regions.shape
        device = regions.device
        class_ids = class_ids.to(device=device, dtype=torch.long)

        raw_mask = torch.zeros((b, r), device=device, dtype=torch.bool)
        dominant_ids = torch.full(
            (b, r),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        bias_score = torch.zeros(
            (b, r),
            device=device,
            dtype=regions.dtype,
        )

        z_all = regions.clamp_min(0)

        for y in class_ids.unique().tolist():
            y = int(y)
            image_mask = class_ids.eq(y)
            n_img = int(image_mask.sum().item())
            if n_img == 0:
                continue
            if y not in self.concept_banks:
                continue

            bank = self.concept_banks[y].to(
                device=device,
                dtype=regions.dtype,
            )
            if bank.shape[1] != d:
                raise ValueError(
                    f"Class {y} concept dim {bank.shape[1]} != region dim {d}."
                )

            z = z_all[image_mask].reshape(-1, d)
            coeff = _batch_projected_nnls_torch(
                z,
                bank,
                steps=self.localization_nnls_steps,
                eps=self.eps,
            )  # [n_img*R,K_y]

            dominant = coeff.argmax(dim=1)  # [N]
            coeff_sum = coeff.sum(dim=1).clamp_min(self.eps)

            bias_ids = self.bias_concepts.get(y, [])
            if len(bias_ids) > 0:
                bias_id_t = torch.tensor(
                    bias_ids,
                    device=device,
                    dtype=torch.long,
                )
                selected_coeff = coeff.index_select(1, bias_id_t)
                selected_mass = selected_coeff.sum(dim=1)
                score = selected_mass / coeff_sum

                # No extra patch threshold: dominant concept decides the hard mask.
                is_bias_dominant = (
                    dominant.unsqueeze(1).eq(bias_id_t.unsqueeze(0)).any(dim=1)
                )
            else:
                score = torch.zeros_like(coeff_sum)
                is_bias_dominant = torch.zeros_like(
                    dominant,
                    dtype=torch.bool,
                )

            raw_mask[image_mask] = is_bias_dominant.view(n_img, r)
            dominant_ids[image_mask] = dominant.view(n_img, r)
            bias_score[image_mask] = score.view(n_img, r)

        return raw_mask, dominant_ids, bias_score

    @torch.no_grad()
    def _stage2_spurious_partition(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Use the FROZEN Stage-1 snapshot for stable concept localization.

        Returns:
            raw_spurious_mask:   [B,R]
            raw_relevant_mask:   [B,R]
            dominant_concept_ids:[B,R]
            bias_score:          [B,R]
        """
        if self._assignment_model is None:
            raise RuntimeError(
                "No frozen Stage-1 assignment model. Run discover(..., "
                "make_assignment_snapshot=True) or call "
                "prepare_stage2_assignment_model(model)."
            )

        assignment_inputs = inputs.to(
            self._assignment_device,
            non_blocking=True,
        )
        assignment_labels = labels.to(
            self._assignment_device,
            non_blocking=True,
        ).long()

        _, features = _capture_layer4_and_forward(
            self._assignment_model,
            assignment_inputs,
        )
        regions = self._regions_from_features(features).clamp_min(0)

        raw_mask, dominant_ids, bias_score = self._localize_from_regions(
            regions=regions,
            class_ids=assignment_labels,
        )

        raw_relevant = ~raw_mask
        return raw_mask, raw_relevant, dominant_ids, bias_score

    # -------------------------------------------------------------------------
    # Top-rel_num priority protection
    # -------------------------------------------------------------------------
    def _priority_relevant_mask(
        self,
        student_regions: torch.Tensor,
        labels: torch.Tensor,
        classifier: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select Top-rel_num current-student tokens most aligned with the GT class
        direction. These tokens have absolute priority and can never be nuisance.
        """
        b, r, d = student_regions.shape
        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with regions [*,{},{}].".format(
                    tuple(classifier.weight.shape),
                    r,
                    d,
                )
            )
        if labels.shape[0] != b:
            raise ValueError("labels batch size must match student_regions.")

        if self.rel_num == 0:
            return (
                torch.zeros((b, r), dtype=torch.bool, device=student_regions.device),
                torch.zeros((b, r), dtype=student_regions.dtype, device=student_regions.device),
            )

        k = min(int(self.rel_num), int(r))

        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=student_regions.device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        gt_w = class_n.index_select(0, labels.long())
        gt_similarity = torch.einsum("brd,bd->br", region_n, gt_w)
        top_idx = gt_similarity.topk(
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        priority_mask = torch.zeros(
            (b, r),
            dtype=torch.bool,
            device=student_regions.device,
        )
        priority_mask.scatter_(1, top_idx, True)
        return priority_mask, gt_similarity

    @staticmethod
    def _apply_priority_protection(
        raw_spurious_mask: torch.Tensor,
        priority_relevant_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_spurious_mask = raw_spurious_mask.to(
            device=priority_relevant_mask.device,
            dtype=torch.bool,
        )
        priority_relevant_mask = priority_relevant_mask.bool()

        protected = raw_spurious_mask & priority_relevant_mask
        spurious_mask = raw_spurious_mask & (~priority_relevant_mask)
        relevant_mask = ~spurious_mask
        return spurious_mask, relevant_mask, protected

    # -------------------------------------------------------------------------
    # WPS: suppress only positive wrong-class evidence
    # -------------------------------------------------------------------------
    def _nuisance_wrong_class_suppression_loss(
        self,
        student_regions: torch.Tensor,
        labels: torch.Tensor,
        spurious_mask: torch.Tensor,
        classifier: nn.Linear,
    ) -> torch.Tensor:
        """
        Wrong-Class Positive Suppression (WPS).

        For each FINAL nuisance token z_i^s with GT y_i:

            l(z_i^s) = [ max_{c != y_i} ReLU(cos(z_i^s, w_c)) ]^2

        The GT direction w_{y_i} is excluded from the auxiliary penalty.
        Negative wrong-class evidence is also left untouched.
        Classifier weights are detached; WPS changes the representation rather
        than moving class anchors.
        """
        if student_regions.ndim != 3:
            raise ValueError("student_regions must be [B,R,D].")

        b, r, d = student_regions.shape
        if labels.shape[0] != b:
            raise ValueError("labels batch size must match student_regions.")
        if classifier.weight.shape != (self.num_classes, d):
            raise ValueError(
                "Classifier weight shape {} incompatible with region dim {}.".format(
                    tuple(classifier.weight.shape),
                    d,
                )
            )

        device = student_regions.device
        labels = labels.to(device=device, dtype=torch.long)
        spurious_mask = spurious_mask.to(device=device, dtype=torch.bool)

        if spurious_mask.shape != (b, r):
            raise ValueError(
                f"spurious_mask must be {(b, r)}, got {tuple(spurious_mask.shape)}."
            )

        if not bool(spurious_mask.any().item()):
            return student_regions.sum() * 0.0

        region_n = F.normalize(
            student_regions,
            p=2,
            dim=2,
            eps=self.eps,
        )
        class_n = F.normalize(
            classifier.weight.detach().to(
                device=device,
                dtype=student_regions.dtype,
            ),
            p=2,
            dim=1,
            eps=self.eps,
        )

        similarity = torch.einsum(
            "brd,cd->brc",
            region_n,
            class_n,
        )  # [B,R,C]

        # Exclude GT class from the max over wrong classes.
        gt_mask = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).bool()[:, None, :]  # [B,1,C]

        wrong_similarity = similarity.masked_fill(
            gt_mask,
            -torch.inf,
        )
        strongest_wrong = wrong_similarity.max(dim=2).values  # [B,R]

        # Suppress ONLY positive wrong-class evidence.
        patch_loss = F.relu(strongest_wrong).pow(2)
        nuisance_patch_loss = patch_loss[spurious_mask]

        if nuisance_patch_loss.numel() == 0:
            return student_regions.sum() * 0.0

        return nuisance_patch_loss.mean()

    # -------------------------------------------------------------------------
    # Public Stage-2 forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> GradientTrailOutput:
        """
        Direct replacement pattern for the user's existing RaVL forward:

            out = gradient_trail(
                student_model=model,
                inputs=inputs,
                labels=labels,
            )

            loss = loss_cls + lambda_region * out.loss_region
        """
        if self.discovery_result is None:
            raise RuntimeError("Run discover(model, audit_loader) before Stage 2.")
        if self._assignment_model is None:
            raise RuntimeError(
                "Frozen Stage-1 assignment model is missing. Run discover(..., "
                "make_assignment_snapshot=True)."
            )

        logits, student_features = _capture_layer4_and_forward(
            student_model,
            inputs,
        )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {logits.shape[1]}."
            )

        student_regions = self._regions_from_features(student_features)
        classifier = _get_classifier(student_model)

        # 1) Highest-priority task-relevant regions from CURRENT student.
        priority_mask, _ = self._priority_relevant_mask(
            student_regions=student_regions,
            labels=labels,
            classifier=classifier,
        )

        # 2) Raw spurious mask from FROZEN Stage-1 concept space.
        (
            raw_spurious_mask,
            _,
            dominant_ids,
            bias_score,
        ) = self._stage2_spurious_partition(
            inputs=inputs,
            labels=labels,
        )

        raw_spurious_mask = raw_spurious_mask.to(priority_mask.device)
        dominant_ids = dominant_ids.to(priority_mask.device)
        bias_score = bias_score.to(priority_mask.device)

        if raw_spurious_mask.shape != priority_mask.shape:
            raise ValueError(
                "Frozen localization shape {} != current student region shape {}.".format(
                    tuple(raw_spurious_mask.shape),
                    tuple(priority_mask.shape),
                )
            )

        # 3) Top-rel_num can never be nuisance.
        spurious_mask, relevant_mask, protected = self._apply_priority_protection(
            raw_spurious_mask=raw_spurious_mask,
            priority_relevant_mask=priority_mask,
        )

        # 4) WPS only on FINAL nuisance regions.
        loss_wps = self._nuisance_wrong_class_suppression_loss(
            student_regions=student_regions,
            labels=labels,
            spurious_mask=spurious_mask,
            classifier=classifier,
        )

        valid_images = spurious_mask.any(dim=1)

        return GradientTrailOutput(
            loss_region=loss_wps,
            loss_wps=loss_wps,
            logits=logits,
            raw_spurious_mask=raw_spurious_mask.detach(),
            spurious_mask=spurious_mask.detach(),
            priority_relevant_mask=priority_mask.detach(),
            relevant_mask=relevant_mask.detach(),
            dominant_concept_ids=dominant_ids.detach(),
            bias_score=bias_score.detach(),
            num_raw_spurious_regions=int(raw_spurious_mask.sum().item()),
            num_protected_regions=int(protected.sum().item()),
            num_spurious_regions=int(spurious_mask.sum().item()),
            num_relevant_regions=int(relevant_mask.sum().item()),
            num_valid_images=int(valid_images.sum().item()),
        )

    __call__ = forward

    # -------------------------------------------------------------------------
    # Loss helper: CE stays the main task loss
    # -------------------------------------------------------------------------
    def combine_with_classification_loss(
        self,
        classification_loss: torch.Tensor,
        output: GradientTrailOutput,
        lambda_region: Optional[float] = None,
    ) -> torch.Tensor:
        if lambda_region is None:
            lambda_region = self.lambda_region
        return classification_loss + float(lambda_region) * output.loss_region

    # -------------------------------------------------------------------------
    # Diagnostics: classwise patch ratio
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_classwise_spurious_ratio(
        self,
        student_model: nn.Module,
        data_loader,
        device=None,
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Report raw/final Gradient-Trail spurious-token coverage per GT class.
        """
        base = _unwrap_model(student_model)
        if device is None:
            device = next(base.parameters()).device

        was_training = base.training
        base.eval()

        n_img = torch.zeros(self.num_classes, dtype=torch.long)
        n_patch = torch.zeros(self.num_classes, dtype=torch.long)
        n_raw = torch.zeros(self.num_classes, dtype=torch.long)
        n_protected = torch.zeros(self.num_classes, dtype=torch.long)
        n_final = torch.zeros(self.num_classes, dtype=torch.long)
        n_hit = torch.zeros(self.num_classes, dtype=torch.long)
        bias_score_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        bias_score_count = torch.zeros(self.num_classes, dtype=torch.long)

        try:
            for batch_idx, batch in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TypeError("data_loader must return (inputs, labels, ...).")

                inputs = batch[0]
                labels = batch[1]
                if isinstance(inputs, (tuple, list)):
                    inputs = inputs[0]

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                _, current_features = _capture_layer4_and_forward(
                    student_model,
                    inputs,
                )
                current_regions = self._regions_from_features(current_features)
                classifier = _get_classifier(student_model)

                priority_mask, _ = self._priority_relevant_mask(
                    student_regions=current_regions,
                    labels=labels,
                    classifier=classifier,
                )

                raw_mask, _, _, bias_score = self._stage2_spurious_partition(
                    inputs=inputs,
                    labels=labels,
                )
                raw_mask = raw_mask.to(device)
                bias_score = bias_score.to(device)

                final_mask, _, protected = self._apply_priority_protection(
                    raw_spurious_mask=raw_mask,
                    priority_relevant_mask=priority_mask,
                )

                b, r = final_mask.shape

                for c in range(self.num_classes):
                    cm = labels.eq(c)
                    nc = int(cm.sum().item())
                    if nc == 0:
                        continue

                    raw_c = raw_mask[cm]
                    pro_c = protected[cm]
                    final_c = final_mask[cm]
                    score_c = bias_score[cm]

                    n_img[c] += nc
                    n_patch[c] += nc * r
                    n_raw[c] += int(raw_c.sum().item())
                    n_protected[c] += int(pro_c.sum().item())
                    n_final[c] += int(final_c.sum().item())
                    n_hit[c] += int(final_c.any(dim=1).sum().item())

                    if bool(raw_c.any().item()):
                        bias_score_sum[c] += float(score_c[raw_c].sum().item())
                        bias_score_count[c] += int(raw_c.sum().item())

        finally:
            if was_training:
                base.train()
            else:
                base.eval()

        def safe_ratio(num: torch.Tensor, den: torch.Tensor) -> np.ndarray:
            return (
                num.double() / den.double().clamp_min(1)
            ).cpu().numpy()

        raw_ratio = safe_ratio(n_raw, n_patch)
        final_ratio = safe_ratio(n_final, n_patch)
        protected_ratio = safe_ratio(n_protected, n_patch)
        hit_ratio = safe_ratio(n_hit, n_img)
        mean_bias_score = (
            bias_score_sum / bias_score_count.double().clamp_min(1)
        ).cpu().numpy()

        if verbose:
            print("=" * 105)
            print("Class-wise Gradient-Trail spurious-token statistics")
            print("Selected class-specific spurious concepts:")
            print(self.bias_concepts)
            print("=" * 105)
            print(
                "{:>6s} {:>8s} {:>10s} {:>11s} {:>10s} {:>10s} {:>10s}".format(
                    "Class",
                    "Images",
                    "Raw(%)",
                    "Protect(%)",
                    "Final(%)",
                    "ImgHit(%)",
                    "BiasScore",
                )
            )
            print("-" * 105)
            for c in range(self.num_classes):
                print(
                    "{:6d} {:8d} {:10.2f} {:11.2f} {:10.2f} {:10.2f} {:10.4f}".format(
                        c,
                        int(n_img[c].item()),
                        100.0 * raw_ratio[c],
                        100.0 * protected_ratio[c],
                        100.0 * final_ratio[c],
                        100.0 * hit_ratio[c],
                        float(mean_bias_score[c]),
                    )
                )
            print("=" * 105)
            total_final = int(n_final.sum().item())
            total_patch = int(n_patch.sum().item())
            print(
                "Overall final spurious ratio: {:.2f}% ({}/{})".format(
                    100.0 * total_final / float(max(total_patch, 1)),
                    total_final,
                    total_patch,
                )
            )
            print("=" * 105)

        return {
            "num_images": n_img.numpy(),
            "num_total_patches": n_patch.numpy(),
            "num_raw_spurious_patches": n_raw.numpy(),
            "num_protected_patches": n_protected.numpy(),
            "num_final_spurious_patches": n_final.numpy(),
            "raw_spurious_ratio": raw_ratio,
            "protected_ratio": protected_ratio,
            "final_spurious_ratio": final_ratio,
            "image_with_spurious_ratio": hit_ratio,
            "mean_raw_bias_score": mean_bias_score,
        }

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    @staticmethod
    def _vis_to_numpy_image(
        image: torch.Tensor,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> np.ndarray:
        x = image.detach().cpu().float().clone()
        if x.ndim != 3:
            raise ValueError("image must be [C,H,W].")

        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
            if mean_t.shape[0] == 1 and x.shape[0] == 3:
                mean_t = mean_t.repeat(3, 1, 1)
                std_t = std_t.repeat(3, 1, 1)
            x = x * std_t + mean_t
        else:
            xmin = float(x.min().item())
            xmax = float(x.max().item())
            if xmin < 0.0 or xmax > 1.0:
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] >= 3:
            x = x[:3]
        else:
            raise ValueError("Only 1-channel or >=3-channel images are supported.")

        return x.clamp(0, 1).permute(1, 2, 0).numpy()

    @staticmethod
    def _draw_boxes(
        ax,
        mask: np.ndarray,
        image_h: int,
        image_w: int,
        linewidth: float = 1.5,
    ) -> None:
        from matplotlib.patches import Rectangle

        hf, wf = mask.shape
        for rr in range(hf):
            for cc in range(wf):
                if not bool(mask[rr, cc]):
                    continue
                x0 = cc * image_w / wf
                x1 = (cc + 1) * image_w / wf
                y0 = rr * image_h / hf
                y1 = (rr + 1) * image_h / hf
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        linewidth=linewidth,
                    )
                )

    @torch.no_grad()
    def visualize_spurious_regions(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        save_dir: str = "./gradient_trail_visualization",
        file_prefix: str = "sample",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        max_images: int = 8,
    ) -> List[Dict[str, object]]:
        """
        Save diagnostic panels:
            Original | Dominant Concept | Raw Spurious | Priority Relevant |
            Final Spurious | Bias Score
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        base = _unwrap_model(student_model)
        device = next(base.parameters()).device
        was_training = base.training
        base.eval()

        try:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            _, student_features = _capture_layer4_and_forward(
                student_model,
                inputs,
            )
            student_regions = self._regions_from_features(student_features)
            classifier = _get_classifier(student_model)
            priority_mask, _ = self._priority_relevant_mask(
                student_regions=student_regions,
                labels=labels,
                classifier=classifier,
            )

            raw_mask, _, dominant_ids, bias_score = self._stage2_spurious_partition(
                inputs=inputs,
                labels=labels,
            )
            raw_mask = raw_mask.to(device)
            dominant_ids = dominant_ids.to(device)
            bias_score = bias_score.to(device)

            final_mask, _, protected = self._apply_priority_protection(
                raw_spurious_mask=raw_mask,
                priority_relevant_mask=priority_mask,
            )

            b, _, image_h, image_w = inputs.shape
            _, _, hf, wf = student_features.shape
            if hf * wf != final_mask.shape[1]:
                raise ValueError("Spatial-grid mismatch in visualization.")

            n_show = min(int(max_images), b)
            summaries: List[Dict[str, object]] = []

            for i in range(n_show):
                img_np = self._vis_to_numpy_image(inputs[i], mean=mean, std=std)
                dom = dominant_ids[i].view(hf, wf).detach().cpu().numpy()
                raw = raw_mask[i].view(hf, wf).detach().cpu().numpy()
                pri = priority_mask[i].view(hf, wf).detach().cpu().numpy()
                final = final_mask[i].view(hf, wf).detach().cpu().numpy()
                score = bias_score[i].view(hf, wf).detach().cpu().numpy()

                fig, axes = plt.subplots(1, 6, figsize=(22, 4))

                axes[0].imshow(img_np)
                axes[0].set_title(f"Original | y={int(labels[i].item())}")

                im1 = axes[1].imshow(dom, interpolation="nearest")
                axes[1].set_title("Dominant Concept")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(img_np)
                self._draw_boxes(axes[2], raw, image_h, image_w)
                axes[2].set_title(f"Raw Spurious ({int(raw.sum())})")

                axes[3].imshow(img_np)
                self._draw_boxes(axes[3], pri, image_h, image_w)
                axes[3].set_title(f"Priority Top-{self.rel_num}")

                axes[4].imshow(img_np)
                self._draw_boxes(axes[4], final, image_h, image_w)
                axes[4].set_title(f"Final Spurious ({int(final.sum())})")

                im5 = axes[5].imshow(score, vmin=0.0, vmax=1.0, interpolation="nearest")
                axes[5].set_title("Bias Concept Mass")
                fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

                for ax in axes:
                    ax.axis("off")

                fig.tight_layout()
                path = Path(save_dir) / f"{file_prefix}_{i:03d}.png"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                plt.close(fig)

                summaries.append(
                    {
                        "index": i,
                        "label": int(labels[i].item()),
                        "num_raw_spurious": int(raw.sum()),
                        "num_priority": int(pri.sum()),
                        "num_protected": int(protected[i].sum().item()),
                        "num_final_spurious": int(final.sum()),
                        "mean_bias_score": float(score[raw].mean()) if raw.any() else 0.0,
                        "path": str(path),
                    }
                )

            return summaries

        finally:
            if was_training:
                base.train()
            else:
                base.eval()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save_discovery(self, path: str) -> None:
        if self.discovery_result is None or len(self.concept_banks) == 0:
            raise RuntimeError("Nothing to save; run discover(...) first.")

        torch.save(
            {
                "num_classes": self.num_classes,
                "num_concepts": self.num_concepts,
                "probe_step": self.probe_step,
                "bias_threshold": self.bias_threshold,
                "max_bias_concepts_per_class": self.max_bias_concepts_per_class,
                "rel_num": self.rel_num,
                "concept_banks": self.concept_banks,
                "scores": self.scores,
                "e_fn": self.e_fn,
                "e_fp": self.e_fp,
                "ranked_concepts": self.ranked_concepts,
                "bias_concepts": self.bias_concepts,
                "discovery_result": self.discovery_result,
            },
            path,
        )

    def load_discovery(
        self,
        path: str,
        model_for_assignment: Optional[nn.Module] = None,
        device=None,
    ) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)

        if int(state["num_classes"]) != self.num_classes:
            raise ValueError(
                "Saved num_classes={} but object num_classes={}.".format(
                    state["num_classes"],
                    self.num_classes,
                )
            )

        self.concept_banks = {
            int(k): v.float().cpu()
            for k, v in state["concept_banks"].items()
        }
        self.scores = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in state.get("scores", {}).items()
        }
        self.e_fn = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in state.get("e_fn", {}).items()
        }
        self.e_fp = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in state.get("e_fp", {}).items()
        }
        self.ranked_concepts = {
            int(k): [int(x) for x in v]
            for k, v in state.get("ranked_concepts", {}).items()
        }
        self.bias_concepts = {
            int(k): [int(x) for x in v]
            for k, v in state.get("bias_concepts", {}).items()
        }
        self.discovery_result = state.get("discovery_result", None)

        if model_for_assignment is not None:
            self.prepare_stage2_assignment_model(
                model=model_for_assignment,
                device=device,
            )


# =============================================================================
# Minimal executable smoke test
# =============================================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    class TinyResNet(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=False),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer4(x)
            z = x.mean(dim=(2, 3))
            return self.fc(z)

    torch.manual_seed(7)
    np.random.seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyResNet(num_classes=3).to(device)

    # Synthetic balanced audit/training data.
    x = torch.rand(48, 3, 32, 32)
    y = torch.arange(48) % 3
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=8,
        shuffle=False,
    )

    gt = GradientTrailResNet(
        num_classes=3,
        num_concepts=3,
        probe_step=5.0,
        # Low smoke-test threshold only so synthetic random model can select concepts.
        bias_threshold=-1.0,
        max_bias_concepts_per_class=1,
        rel_num=2,
        max_tokens_per_class=64,
        nmf_max_iter=100,
        localization_nnls_steps=10,
        lambda_region=0.1,
        random_seed=0,
    )

    result = gt.discover(
        model=model,
        audit_loader=loader,
        device=device,
        verbose=True,
        make_assignment_snapshot=True,
    )

    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    model.train()
    out = gt(
        student_model=model,
        inputs=batch_x,
        labels=batch_y,
    )

    ce = F.cross_entropy(out.logits, batch_y)
    total = gt.combine_with_classification_loss(
        classification_loss=ce,
        output=out,
    )

    model.zero_grad(set_to_none=True)
    total.backward()

    # Basic assertions.
    assert out.spurious_mask.ndim == 2
    assert out.spurious_mask.shape[0] == batch_x.shape[0]
    assert out.raw_spurious_mask.shape == out.spurious_mask.shape
    assert out.priority_relevant_mask.shape == out.spurious_mask.shape
    assert torch.isfinite(total).all()

    stats = gt.evaluate_classwise_spurious_ratio(
        student_model=model,
        data_loader=loader,
        device=device,
        max_batches=2,
        verbose=True,
    )

    print("bias concepts:", result.bias_concepts)
    print("output statistics:", out.statistics())
    print("final_spurious_ratio:", stats["final_spurious_ratio"])
    print("GradientTrailResNet layer4-only + WPS smoke test passed.")
