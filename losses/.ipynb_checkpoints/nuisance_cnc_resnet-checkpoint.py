
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Output dataclasses
# ================================================================

@dataclass
class NuisanceDiscoveryResult:
    nuisance_threshold: float
    reference_threshold: float

    num_images: int
    num_regions: int

    num_nuisance_candidates: int
    num_reference_candidates: int

    num_nuisance_prototypes: int
    num_reference_prototypes: int

    nuisance_silhouette: float
    reference_silhouette: float

    mean_reliance: float
    mean_harmful_ratio: float
    mean_instability: float
    mean_nuisance_score: float

    def summary(self) -> Dict[str, float]:
        return {
            "nuisance_threshold": self.nuisance_threshold,
            "reference_threshold": self.reference_threshold,

            "num_images": float(self.num_images),
            "num_regions": float(self.num_regions),

            "num_nuisance_candidates": float(
                self.num_nuisance_candidates
            ),
            "num_reference_candidates": float(
                self.num_reference_candidates
            ),

            "num_nuisance_prototypes": float(
                self.num_nuisance_prototypes
            ),
            "num_reference_prototypes": float(
                self.num_reference_prototypes
            ),

            "nuisance_silhouette": float(
                self.nuisance_silhouette
            ),
            "reference_silhouette": float(
                self.reference_silhouette
            ),

            "mean_reliance": self.mean_reliance,
            "mean_harmful_ratio": self.mean_harmful_ratio,
            "mean_instability": self.mean_instability,
            "mean_nuisance_score": self.mean_nuisance_score,
        }


@dataclass
class NuisanceCnCOutput:
    loss_cnc: torch.Tensor
    logits: torch.Tensor

    nuisance_profile: torch.Tensor
    nuisance_strength: torch.Tensor

    patch_nuisance_weight: torch.Tensor
    patch_nuisance_id: torch.Tensor
    nuisance_mask: torch.Tensor

    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    profile_distance: torch.Tensor

    num_positive_pairs: int
    num_negative_pairs: int
    num_valid_anchors: int

    num_nuisance_patches: int
    num_total_patches: int

    def statistics(self) -> Dict[str, float]:
        return {
            "loss_cnc": float(
                self.loss_cnc.detach().item()
            ),

            "mean_nuisance_strength": float(
                self.nuisance_strength
                .detach()
                .mean()
                .item()
            ),

            "num_positive_pairs": float(
                self.num_positive_pairs
            ),
            "num_negative_pairs": float(
                self.num_negative_pairs
            ),
            "num_valid_anchors": float(
                self.num_valid_anchors
            ),

            "nuisance_patch_ratio": (
                self.num_nuisance_patches
                / max(self.num_total_patches, 1)
            ),
        }


# ================================================================
# Model helpers
# ================================================================

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _unwrap_tensor(x, name: str) -> torch.Tensor:
    if torch.is_tensor(x):
        return x

    if (
        isinstance(x, (tuple, list))
        and len(x) > 0
        and torch.is_tensor(x[0])
    ):
        return x[0]

    if isinstance(x, dict):
        for key in ("logits", "pred", "output"):
            if key in x and torch.is_tensor(x[key]):
                return x[key]

    raise TypeError(
        "{} is not a supported model output.".format(name)
    )


def _get_classifier(model: nn.Module) -> nn.Linear:
    base = _unwrap_model(model)

    candidates = []

    if hasattr(base, "get_classifier"):
        try:
            candidates.append(
                base.get_classifier()
            )
        except Exception:
            pass

    for name in ("fc", "head", "classifier"):
        if hasattr(base, name):
            candidates.append(
                getattr(base, name)
            )

    for candidate in candidates:
        if isinstance(candidate, nn.Linear):
            return candidate

        if isinstance(candidate, nn.Sequential):
            for sub in reversed(candidate):
                if isinstance(sub, nn.Linear):
                    return sub

    raise AttributeError(
        "Cannot find final nn.Linear classifier."
    )


def _capture_layer4_and_forward(
    model: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ResNet-like model.

    Returns
    -------
    logits:
        [B,C]

    features:
        [B,D,Hf,Wf]
    """
    base = _unwrap_model(model)

    if not hasattr(base, "layer4"):
        raise AttributeError(
            "Model must be ResNet-like and contain model.layer4."
        )

    holder = {}

    def hook(_module, _inputs, output):
        holder["features"] = _unwrap_tensor(
            output,
            "layer4 output",
        )

    handle = base.layer4.register_forward_hook(
        hook
    )

    try:
        logits = _unwrap_tensor(
            model(inputs),
            "model(inputs)",
        )
    finally:
        handle.remove()

    if "features" not in holder:
        raise RuntimeError(
            "Failed to capture layer4 features."
        )

    features = holder["features"]

    if features.ndim != 4:
        raise ValueError(
            "layer4 must be [B,D,H,W], got {}".format(
                tuple(features.shape)
            )
        )

    if logits.ndim != 2:
        raise ValueError(
            "logits must be [B,C], got {}".format(
                tuple(logits.shape)
            )
        )

    return logits, features


def _regions_from_features(
    features: torch.Tensor,
) -> torch.Tensor:
    """
    [B,D,H,W] -> [B,H*W,D]
    """
    return (
        features
        .flatten(2)
        .transpose(1, 2)
    )


def _true_class_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    m(x,y) = logit_y - max_{c != y} logit_c
    """
    labels = labels.view(-1).long()

    true_logit = (
        logits
        .gather(
            dim=1,
            index=labels[:, None],
        )
        .squeeze(1)
    )

    other = logits.clone()

    other.scatter_(
        dim=1,
        index=labels[:, None],
        value=float("-inf"),
    )

    max_other = (
        other
        .max(dim=1)
        .values
    )

    return true_logit - max_other


# ================================================================
# Main
# ================================================================

class NuisanceCnCResNet(object):
    """
    ============================================================
    Stage 1: offline nuisance discovery
    ============================================================

    1) Run a normal pretrained classifier.
    2) Capture layer4 patch features.
    3) For every patch, estimate:
           R_i : model reliance
           H_i : harmful ratio
           U_i : intervention instability
    4) Build:
           rho_i = R_i * HarmGate(H_i) * InstabilityFactor(U_i)
    5) Select:
           high-rho patches -> nuisance pool
           low-rho patches  -> reference pool
    6) Search prototype count automatically with cosine Silhouette:
           K = num_classes * factor
       Example:
           num_classes = 8
           prototype_k_factors = [1,2,3,4]
           candidate K = [8,16,24,32]
    7) Nuisance/reference pools choose their BEST K independently.

    ============================================================
    Stage 2: nuisance profile + CnC
    ============================================================

    Frozen Stage-1 encoder:
        patch -> P_N / P_C matching -> nuisance profile q_x

    Positive:
        same Y + different nuisance profile

    Hard negative:
        different Y + similar nuisance profile

    Final loss:
        L = L_CE + lambda_cnc * L_CnC

    ============================================================
    Notes
    ============================================================

    - No teacher
    - No GRL
    - No projection head
    - This class has no trainable parameters
    """

    def __init__(
        self,
        num_classes: int,

        # --------------------------------------------------------
        # CnC
        # --------------------------------------------------------
        temperature: float = 0.07,
        lambda_cnc: float = 0.50,
        profile_distance: str = "cosine",

        delta_pos: float = 0.20,
        delta_neg: float = 0.05,

        max_positives: Optional[int] = 8,
        max_negatives: Optional[int] = 32,

        fallback_to_nearest: bool = True,

        # --------------------------------------------------------
        # Prototype matching
        # --------------------------------------------------------
        profile_temperature: float = 0.10,
        prototype_temperature: float = 0.10,

        nuisance_margin: float = 0.0,

        min_nuisance_similarity: Optional[
            float
        ] = None,

        absolute_gate_temperature: float = 0.10,

        nuisance_patch_threshold: float = 0.50,

        include_clean_state: bool = True,
        hard_patch_assignment: bool = False,

        # --------------------------------------------------------
        # Discovery
        # --------------------------------------------------------
        intervention_modes: Sequence[str] = (
            "zero",
            "mean",
            "loo",
        ),

        reliance_quantile: float = 0.90,

        harmful_ratio_center: float = 0.50,

        harmful_gate_temperature: float = 0.10,

        instability_floor: float = 0.50,

        nuisance_top_fraction: float = 0.20,
        reference_bottom_fraction: float = 0.30,

        class_balanced_discovery: bool = True,

        # --------------------------------------------------------
        # Prototype K search
        # --------------------------------------------------------
        prototype_k_factors: Optional[
            Sequence[int]
        ] = (1, 2, 3, 4),

        silhouette_sample_size: int = 3000,

        # --------------------------------------------------------
        # Fixed-K fallback
        # Only used when prototype_k_factors=None
        # --------------------------------------------------------
        num_nuisance_prototypes: int = 8,
        num_reference_prototypes: int = 8,

        # --------------------------------------------------------
        # K-Medoids
        # --------------------------------------------------------
        kmedoids_iterations: int = 30,

        max_cluster_regions: Optional[
            int
        ] = 20000,

        assignment_chunk_size: int = 8192,

        random_seed: int = 0,
    ) -> None:

        if num_classes is None:
            raise ValueError(
                "num_classes is None. Please pass the real "
                "number of classes, e.g. model.fc.out_features."
            )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        if profile_temperature <= 0:
            raise ValueError(
                "profile_temperature must be > 0"
            )

        if prototype_temperature <= 0:
            raise ValueError(
                "prototype_temperature must be > 0"
            )

        if lambda_cnc < 0:
            raise ValueError(
                "lambda_cnc must be >= 0"
            )

        if profile_distance not in (
            "cosine",
            "js",
            "l1",
        ):
            raise ValueError(
                "profile_distance must be "
                "'cosine', 'js' or 'l1'."
            )

        if not (
            0.0
            <= nuisance_patch_threshold
            <= 1.0
        ):
            raise ValueError(
                "nuisance_patch_threshold "
                "must be in [0,1]."
            )

        if not (
            0.0
            < reliance_quantile
            < 1.0
        ):
            raise ValueError(
                "reliance_quantile "
                "must be in (0,1)."
            )

        if not (
            0.0
            <= harmful_ratio_center
            <= 1.0
        ):
            raise ValueError(
                "harmful_ratio_center "
                "must be in [0,1]."
            )

        if not (
            0.0
            <= instability_floor
            <= 1.0
        ):
            raise ValueError(
                "instability_floor "
                "must be in [0,1]."
            )

        if not (
            0.0
            < nuisance_top_fraction
            < 1.0
        ):
            raise ValueError(
                "nuisance_top_fraction "
                "must be in (0,1)."
            )

        if not (
            0.0
            < reference_bottom_fraction
            < 1.0
        ):
            raise ValueError(
                "reference_bottom_fraction "
                "must be in (0,1)."
            )

        if (
            nuisance_top_fraction
            + reference_bottom_fraction
            > 1.0
        ):
            raise ValueError(
                "nuisance_top_fraction + "
                "reference_bottom_fraction "
                "must <= 1."
            )

        if (
            num_nuisance_prototypes < 1
            or num_reference_prototypes < 1
        ):
            raise ValueError(
                "prototype counts must be >= 1."
            )

        if silhouette_sample_size < 10:
            raise ValueError(
                "silhouette_sample_size must be >= 10."
            )

        modes = tuple(
            str(x).lower()
            for x in intervention_modes
        )

        valid_modes = {
            "zero",
            "mean",
            "loo",
        }

        if (
            len(modes) == 0
            or len(
                set(modes)
                - valid_modes
            ) > 0
        ):
            raise ValueError(
                "Invalid intervention_modes: "
                "{}".format(modes)
            )

        if prototype_k_factors is not None:
            prototype_k_factors = tuple(
                sorted(
                    set(
                        int(x)
                        for x in prototype_k_factors
                    )
                )
            )

            if len(prototype_k_factors) == 0:
                raise ValueError(
                    "prototype_k_factors cannot be empty."
                )

            if any(
                x < 1
                for x in prototype_k_factors
            ):
                raise ValueError(
                    "all prototype_k_factors must be >= 1."
                )

        self.num_classes = int(
            num_classes
        )

        # CnC
        self.temperature = float(
            temperature
        )

        self.lambda_cnc = float(
            lambda_cnc
        )

        self.profile_distance = (
            profile_distance
        )

        self.delta_pos = float(
            delta_pos
        )

        self.delta_neg = float(
            delta_neg
        )

        self.max_positives = (
            max_positives
        )

        self.max_negatives = (
            max_negatives
        )

        self.fallback_to_nearest = bool(
            fallback_to_nearest
        )

        # Matching
        self.profile_temperature = float(
            profile_temperature
        )

        self.prototype_temperature = float(
            prototype_temperature
        )

        self.nuisance_margin = float(
            nuisance_margin
        )

        self.min_nuisance_similarity = (
            min_nuisance_similarity
        )

        self.absolute_gate_temperature = float(
            absolute_gate_temperature
        )

        self.nuisance_patch_threshold = float(
            nuisance_patch_threshold
        )

        self.include_clean_state = bool(
            include_clean_state
        )

        self.hard_patch_assignment = bool(
            hard_patch_assignment
        )

        # Discovery
        self.intervention_modes = (
            modes
        )

        self.reliance_quantile = float(
            reliance_quantile
        )

        self.harmful_ratio_center = float(
            harmful_ratio_center
        )

        self.harmful_gate_temperature = float(
            harmful_gate_temperature
        )

        self.instability_floor = float(
            instability_floor
        )

        self.nuisance_top_fraction = float(
            nuisance_top_fraction
        )

        self.reference_bottom_fraction = float(
            reference_bottom_fraction
        )

        self.class_balanced_discovery = bool(
            class_balanced_discovery
        )

        # Auto K
        self.prototype_k_factors = (
            prototype_k_factors
        )

        self.silhouette_sample_size = int(
            silhouette_sample_size
        )

        # Fixed-K fallback
        self.num_nuisance_prototypes = int(
            num_nuisance_prototypes
        )

        self.num_reference_prototypes = int(
            num_reference_prototypes
        )

        # K-Medoids
        self.kmedoids_iterations = int(
            kmedoids_iterations
        )

        self.max_cluster_regions = (
            max_cluster_regions
        )

        self.assignment_chunk_size = int(
            assignment_chunk_size
        )

        self.random_seed = int(
            random_seed
        )

        self.eps = 1e-8

        # Results
        self.nuisance_prototypes_raw = None
        self.nuisance_prototypes_norm = None

        self.reference_prototypes_raw = None
        self.reference_prototypes_norm = None

        self.discovery_result = None

        # Frozen Stage-1 coordinate system
        self._assignment_model = None
        self._assignment_device = None


    # ============================================================
    # K-Medoids
    # ============================================================

    @torch.no_grad()
    def _fit_kmedoids_cosine(
        self,
        x_raw: torch.Tensor,
        k: int,
        seed: int,
    ):
        x = F.normalize(
            x_raw.float(),
            p=2,
            dim=1,
            eps=self.eps,
        )

        n = int(
            x.shape[0]
        )

        if n < 1:
            raise ValueError(
                "Empty clustering pool."
            )

        k = min(
            max(
                int(k),
                1,
            ),
            n,
        )

        if k == 1:
            direction = (
                x.sum(dim=0)
            )

            medoid_id = (
                (x @ direction)
                .argmax()
                .view(1)
            )

            labels = torch.zeros(
                n,
                dtype=torch.long,
                device=x.device,
            )

            return (
                medoid_id,
                labels,
            )

        generator = torch.Generator(
            device=x.device
        )

        generator.manual_seed(
            int(seed)
        )

        first = int(
            torch.randint(
                low=0,
                high=n,
                size=(1,),
                generator=generator,
                device=x.device,
            ).item()
        )

        selected = [
            first
        ]

        min_dist = (
            1.0
            - (
                x
                @ x[
                    first:
                    first + 1
                ].t()
            ).squeeze(1)
        )

        for _ in range(
            1,
            k,
        ):
            idx = int(
                min_dist
                .argmax()
                .item()
            )

            selected.append(
                idx
            )

            dist = (
                1.0
                - (
                    x
                    @ x[
                        idx:
                        idx + 1
                    ].t()
                ).squeeze(1)
            )

            min_dist = (
                torch.minimum(
                    min_dist,
                    dist,
                )
            )

        medoid_ids = torch.tensor(
            selected,
            device=x.device,
            dtype=torch.long,
        )

        old_labels = None

        for _ in range(
            self.kmedoids_iterations
        ):
            medoids = (
                x.index_select(
                    dim=0,
                    index=medoid_ids,
                )
            )

            similarity = (
                x
                @ medoids.t()
            )

            labels = (
                similarity
                .argmax(dim=1)
            )

            if (
                old_labels is not None
                and torch.equal(
                    labels,
                    old_labels,
                )
            ):
                break

            old_labels = (
                labels.clone()
            )

            new_ids = []

            for cluster_id in range(k):
                members = (
                    labels
                    .eq(cluster_id)
                    .nonzero(
                        as_tuple=False
                    )
                    .squeeze(1)
                )

                if members.numel() == 0:
                    candidate = int(
                        similarity
                        .max(dim=1)
                        .values
                        .argmin()
                        .item()
                    )

                    new_ids.append(
                        candidate
                    )

                    continue

                member_x = (
                    x.index_select(
                        dim=0,
                        index=members,
                    )
                )

                direction = (
                    member_x.sum(
                        dim=0
                    )
                )

                local_id = (
                    member_x
                    @ direction
                ).argmax()

                new_ids.append(
                    int(
                        members[
                            local_id
                        ].item()
                    )
                )

            new_ids = torch.tensor(
                new_ids,
                device=x.device,
                dtype=torch.long,
            )

            if torch.equal(
                new_ids,
                medoid_ids,
            ):
                medoid_ids = (
                    new_ids
                )

                break

            medoid_ids = (
                new_ids
            )

        final_medoids = (
            x.index_select(
                dim=0,
                index=medoid_ids,
            )
        )

        final_labels = (
            x
            @ final_medoids.t()
        ).argmax(
            dim=1
        )

        return (
            medoid_ids,
            final_labels,
        )


    # ============================================================
    # Cosine Silhouette
    # ============================================================

    @torch.no_grad()
    def _silhouette_cosine(
        self,
        x_raw: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
    ) -> float:
        """
        Cosine Silhouette score.

        Uses a deterministic subset if the pool is too large.
        """
        n = int(
            x_raw.shape[0]
        )

        if n < 3:
            return -1.0

        if labels.unique().numel() < 2:
            return -1.0

        sample_size = min(
            n,
            self.silhouette_sample_size,
        )

        if sample_size < n:
            generator = torch.Generator(
                device=x_raw.device
            )

            generator.manual_seed(
                int(seed)
            )

            sample_ids = torch.randperm(
                n,
                generator=generator,
                device=x_raw.device,
            )[:sample_size]

            x = x_raw.index_select(
                dim=0,
                index=sample_ids,
            )

            y = labels.index_select(
                dim=0,
                index=sample_ids,
            )

        else:
            x = x_raw
            y = labels

        x = F.normalize(
            x.float(),
            p=2,
            dim=1,
            eps=self.eps,
        )

        distance = (
            1.0
            - x @ x.t()
        ).clamp_min(
            0.0
        )

        silhouette = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        clusters = (
            y.unique()
        )

        for i in range(
            x.shape[0]
        ):
            own = (
                y[i]
            )

            own_mask = (
                y.eq(own)
            )

            own_count = int(
                own_mask
                .sum()
                .item()
            )

            if own_count <= 1:
                silhouette[i] = 0.0
                continue

            a = (
                distance[
                    i,
                    own_mask
                ].sum()
                /
                float(
                    own_count - 1
                )
            )

            b = None

            for cluster_id in clusters:
                if (
                    int(cluster_id.item())
                    == int(own.item())
                ):
                    continue

                mask = (
                    y.eq(
                        cluster_id
                    )
                )

                if not bool(
                    mask.any().item()
                ):
                    continue

                mean_distance = (
                    distance[
                        i,
                        mask
                    ].mean()
                )

                if (
                    b is None
                    or mean_distance < b
                ):
                    b = mean_distance

            if b is None:
                silhouette[i] = 0.0
                continue

            denominator = (
                torch.maximum(
                    a,
                    b,
                )
                .clamp_min(
                    self.eps
                )
            )

            silhouette[i] = (
                b - a
            ) / denominator

        return float(
            silhouette
            .mean()
            .item()
        )


    # ============================================================
    # Automatic prototype K search
    # ============================================================

    @torch.no_grad()
    def _find_best_prototype_k(
        self,
        pool: torch.Tensor,
        pool_name: str,
        seed_offset: int,
        verbose: bool,
    ):
        """
        Example:
            num_classes = 8
            prototype_k_factors = [1,2,3,4]

        candidate K:
            [8,16,24,32]

        Nuisance/reference pools choose independently.
        """
        n = int(
            pool.shape[0]
        )

        if n < 2:
            raise RuntimeError(
                "{} pool is too small.".format(
                    pool_name
                )
            )

        # --------------------------------------------------------
        # Fixed-K fallback
        # --------------------------------------------------------
        if self.prototype_k_factors is None:
            fixed_k = (
                self.num_nuisance_prototypes
                if pool_name == "nuisance"
                else self.num_reference_prototypes
            )

            fixed_k = min(
                max(
                    int(fixed_k),
                    1,
                ),
                n,
            )

            medoid_ids, labels = (
                self._fit_kmedoids_cosine(
                    pool,
                    fixed_k,
                    self.random_seed
                    + seed_offset
                    + fixed_k,
                )
            )

            if fixed_k >= 2:
                silhouette = (
                    self._silhouette_cosine(
                        pool,
                        labels,
                        self.random_seed
                        + seed_offset
                        + 1000
                        + fixed_k,
                    )
                )
            else:
                silhouette = -1.0

            return (
                int(fixed_k),
                medoid_ids,
                float(silhouette),
            )

        # --------------------------------------------------------
        # Auto-K candidates
        # --------------------------------------------------------
        candidate_k = []

        for factor in (
            self.prototype_k_factors
        ):
            k = (
                self.num_classes
                * int(factor)
            )

            # Silhouette needs:
            # 2 <= K < N
            if (
                k >= 2
                and k < n
            ):
                candidate_k.append(
                    int(k)
                )

        candidate_k = sorted(
            set(
                candidate_k
            )
        )

        if len(candidate_k) == 0:
            fallback_k = min(
                max(
                    2,
                    self.num_classes,
                ),
                n - 1,
            )

            candidate_k = [
                fallback_k
            ]

        if verbose:
            print(
                "---------- {} prototype K search ----------".format(
                    pool_name.upper()
                )
            )

            print(
                "candidate K = {}".format(
                    candidate_k
                )
            )

        best_k = None
        best_medoid_ids = None
        best_score = -float("inf")

        for k in candidate_k:
            medoid_ids, labels = (
                self._fit_kmedoids_cosine(
                    pool,
                    k,
                    self.random_seed
                    + seed_offset
                    + k,
                )
            )

            score = (
                self._silhouette_cosine(
                    pool,
                    labels,
                    self.random_seed
                    + seed_offset
                    + 1000
                    + k,
                )
            )

            if verbose:
                print(
                    "K={:3d} | silhouette={:.6f}".format(
                        k,
                        score,
                    )
                )

            if score > best_score:
                best_score = float(
                    score
                )

                best_k = int(
                    k
                )

                best_medoid_ids = (
                    medoid_ids
                    .detach()
                    .clone()
                )

        if best_medoid_ids is None:
            raise RuntimeError(
                "Automatic K search failed for {} pool.".format(
                    pool_name
                )
            )

        if verbose:
            print(
                "BEST {} K={} | silhouette={:.6f}".format(
                    pool_name.upper(),
                    best_k,
                    best_score,
                )
            )

            print(
                "-------------------------------------------"
            )

        return (
            best_k,
            best_medoid_ids,
            best_score,
        )


    # ============================================================
    # Counterfactual feature intervention
    # ============================================================

    def _counterfactual_pooled_features(
        self,
        regions: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        regions:
            [B,R,D]

        output:
            [B,R,D]

        output[:,r,:]:
            global pooled feature after intervention on patch r
        """
        batch_size, num_regions, dim = (
            regions.shape
        )

        pooled = (
            regions.mean(
                dim=1
            )
        )

        pooled_expand = (
            pooled[:, None, :]
            .expand(
                batch_size,
                num_regions,
                dim,
            )
        )

        if mode == "zero":
            return (
                pooled_expand
                - regions
                / float(
                    num_regions
                )
            )

        if mode == "mean":
            return (
                pooled_expand
                + (
                    pooled_expand
                    - regions
                )
                /
                float(
                    num_regions
                )
            )

        if mode == "loo":
            if num_regions <= 1:
                return (
                    pooled_expand.clone()
                )

            total = (
                regions.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            return (
                total
                - regions
            ) / float(
                num_regions - 1
            )

        raise ValueError(
            "Unknown intervention: "
            "{}".format(mode)
        )


    @torch.no_grad()
    def _batch_patch_evidence(
        self,
        logits: torch.Tensor,
        regions: torch.Tensor,
        labels: torch.Tensor,
        classifier: nn.Linear,
    ):
        """
        Returns
        -------
        reliance:
            [B,R]

        harmful_ratio:
            [B,R]

        instability:
            [B,R]
        """
        batch_size, num_regions, dim = (
            regions.shape
        )

        if (
            classifier.weight.shape[1]
            != dim
        ):
            raise ValueError(
                "Classifier input dim "
                "!= layer4 region dim."
            )

        original_margin = (
            _true_class_margin(
                logits,
                labels,
            )
        )

        original_ce = (
            F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        margin_effects = []
        ce_effects = []

        for mode in (
            self.intervention_modes
        ):
            pooled_cf = (
                self._counterfactual_pooled_features(
                    regions,
                    mode,
                )
            )

            cf_logits = (
                classifier(
                    pooled_cf.reshape(
                        batch_size
                        * num_regions,
                        dim,
                    )
                )
                .view(
                    batch_size,
                    num_regions,
                    self.num_classes,
                )
            )

            labels_flat = (
                labels[:, None]
                .expand(
                    batch_size,
                    num_regions,
                )
                .reshape(-1)
            )

            cf_margin = (
                _true_class_margin(
                    cf_logits.reshape(
                        batch_size
                        * num_regions,
                        self.num_classes,
                    ),
                    labels_flat,
                )
                .view(
                    batch_size,
                    num_regions,
                )
            )

            cf_ce = (
                F.cross_entropy(
                    cf_logits.reshape(
                        batch_size
                        * num_regions,
                        self.num_classes,
                    ),
                    labels_flat,
                    reduction="none",
                )
                .view(
                    batch_size,
                    num_regions,
                )
            )

            # >0: original patch supports true-class margin
            # <0: removing it improves margin
            delta_margin = (
                original_margin[:, None]
                - cf_margin
            )

            # >0: removing patch worsens CE -> helpful patch
            # <0: removing patch improves CE -> harmful patch
            delta_ce = (
                cf_ce
                - original_ce[:, None]
            )

            margin_effects.append(
                delta_margin
            )

            ce_effects.append(
                delta_ce
            )

        margin_effects = (
            torch.stack(
                margin_effects,
                dim=-1,
            )
        )

        ce_effects = (
            torch.stack(
                ce_effects,
                dim=-1,
            )
        )

        # --------------------------------------------------------
        # R_i: model reliance
        # --------------------------------------------------------
        reliance = (
            margin_effects
            .abs()
            .mean(
                dim=-1
            )
        )

        # --------------------------------------------------------
        # H_i: harmful ratio
        # --------------------------------------------------------
        harmful_gain = (
            F.relu(
                -ce_effects
            )
            .mean(
                dim=-1
            )
        )

        helpful_gain = (
            F.relu(
                ce_effects
            )
            .mean(
                dim=-1
            )
        )

        harmful_ratio = (
            harmful_gain
            /
            (
                harmful_gain
                + helpful_gain
                + self.eps
            )
        )

        # --------------------------------------------------------
        # U_i: intervention instability
        # --------------------------------------------------------
        if (
            len(
                self.intervention_modes
            )
            == 1
        ):
            instability = (
                torch.zeros_like(
                    reliance
                )
            )
        else:
            instability = (
                margin_effects.std(
                    dim=-1,
                    unbiased=False,
                )
                /
                (
                    margin_effects
                    .abs()
                    .mean(
                        dim=-1
                    )
                    + self.eps
                )
            ).clamp(
                min=0.0,
                max=1.0,
            )

        return (
            reliance,
            harmful_ratio,
            instability,
        )


    # ============================================================
    # Candidate masks
    # ============================================================

    def _extreme_masks(
        self,
        score: torch.Tensor,
        image_labels: torch.Tensor,
        regions_per_image: int,
    ):
        # --------------------------------------------------------
        # Global
        # --------------------------------------------------------
        if not self.class_balanced_discovery:
            high_threshold = (
                torch.quantile(
                    score,
                    1.0
                    - self.nuisance_top_fraction,
                )
            )

            low_threshold = (
                torch.quantile(
                    score,
                    self.reference_bottom_fraction,
                )
            )

            nuisance_mask = (
                score
                >= high_threshold
            )

            reference_mask = (
                score
                <= low_threshold
            )

            overlap = (
                nuisance_mask
                & reference_mask
            )

            reference_mask[
                overlap
            ] = False

            return (
                nuisance_mask,
                reference_mask,
                float(
                    high_threshold.item()
                ),
                float(
                    low_threshold.item()
                ),
            )

        # --------------------------------------------------------
        # Class-balanced
        # --------------------------------------------------------
        patch_labels = (
            image_labels[:, None]
            .expand(
                image_labels.shape[0],
                regions_per_image,
            )
            .reshape(-1)
        )

        nuisance_mask = (
            torch.zeros_like(
                score,
                dtype=torch.bool,
            )
        )

        reference_mask = (
            torch.zeros_like(
                score,
                dtype=torch.bool,
            )
        )

        high_thresholds = []
        low_thresholds = []

        for class_id in range(
            self.num_classes
        ):
            class_ids = (
                patch_labels
                .eq(
                    class_id
                )
                .nonzero(
                    as_tuple=False
                )
                .squeeze(1)
            )

            if class_ids.numel() == 0:
                continue

            class_score = (
                score.index_select(
                    dim=0,
                    index=class_ids,
                )
            )

            high_threshold = (
                torch.quantile(
                    class_score,
                    1.0
                    - self.nuisance_top_fraction,
                )
            )

            low_threshold = (
                torch.quantile(
                    class_score,
                    self.reference_bottom_fraction,
                )
            )

            nuisance_local = (
                class_score
                >= high_threshold
            )

            reference_local = (
                class_score
                <= low_threshold
            )

            nuisance_mask[
                class_ids[
                    nuisance_local
                ]
            ] = True

            reference_mask[
                class_ids[
                    reference_local
                ]
            ] = True

            high_thresholds.append(
                float(
                    high_threshold.item()
                )
            )

            low_thresholds.append(
                float(
                    low_threshold.item()
                )
            )

        overlap = (
            nuisance_mask
            & reference_mask
        )

        reference_mask[
            overlap
        ] = False

        return (
            nuisance_mask,
            reference_mask,

            sum(
                high_thresholds
            )
            /
            max(
                len(
                    high_thresholds
                ),
                1,
            ),

            sum(
                low_thresholds
            )
            /
            max(
                len(
                    low_thresholds
                ),
                1,
            ),
        )


    def _subsample(
        self,
        features: torch.Tensor,
        max_num: Optional[int],
        seed: int,
    ):
        if (
            max_num is None
            or features.shape[0]
            <= int(max_num)
        ):
            return features

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            int(seed)
        )

        ids = torch.randperm(
            features.shape[0],
            generator=generator,
        )[
            : int(max_num)
        ]

        return (
            features.index_select(
                dim=0,
                index=ids,
            )
        )


    # ============================================================
    # Stage 1: Discovery
    # ============================================================

    @torch.no_grad()
    def discover(
        self,
        model: nn.Module,
        reference_loader,
        device=None,
        verbose: bool = True,
        make_assignment_snapshot: bool = True,
    ) -> NuisanceDiscoveryResult:
        """
        Run once after Stage-1 ordinary classifier training.
        """
        if device is None:
            device = next(
                model.parameters()
            ).device

        base = (
            _unwrap_model(
                model
            )
        )

        was_training = (
            base.training
        )

        base.eval()

        classifier = (
            _get_classifier(
                model
            )
        )

        all_regions = []
        all_labels = []

        all_reliance = []
        all_harmful = []
        all_instability = []

        try:
            for batch in reference_loader:
                if (
                    not isinstance(
                        batch,
                        (tuple, list),
                    )
                    or len(batch) < 2
                ):
                    raise TypeError(
                        "reference_loader must return "
                        "(inputs, labels, ...)"
                    )

                inputs = (
                    batch[0]
                )

                labels = (
                    batch[1]
                )

                if isinstance(
                    inputs,
                    (tuple, list),
                ):
                    inputs = (
                        inputs[0]
                    )

                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )

                labels = (
                    labels
                    .to(
                        device,
                        non_blocking=True,
                    )
                    .view(-1)
                    .long()
                )

                logits, features = (
                    _capture_layer4_and_forward(
                        model,
                        inputs,
                    )
                )

                if (
                    logits.shape[1]
                    != self.num_classes
                ):
                    raise ValueError(
                        "Expected {} classes, "
                        "got {}.".format(
                            self.num_classes,
                            logits.shape[1],
                        )
                    )

                regions = (
                    _regions_from_features(
                        features
                    )
                )

                (
                    reliance,
                    harmful,
                    instability,
                ) = (
                    self._batch_patch_evidence(
                        logits=logits,
                        regions=regions,
                        labels=labels,
                        classifier=classifier,
                    )
                )

                all_regions.append(
                    regions
                    .detach()
                    .cpu()
                    .half()
                )

                all_labels.append(
                    labels
                    .detach()
                    .cpu()
                )

                all_reliance.append(
                    reliance
                    .detach()
                    .cpu()
                    .float()
                )

                all_harmful.append(
                    harmful
                    .detach()
                    .cpu()
                    .float()
                )

                all_instability.append(
                    instability
                    .detach()
                    .cpu()
                    .float()
                )

        finally:
            base.train(
                was_training
            )

        if len(all_regions) == 0:
            raise RuntimeError(
                "Empty reference_loader."
            )

        regions_img = (
            torch.cat(
                all_regions,
                dim=0,
            )
            .float()
        )

        labels_img = (
            torch.cat(
                all_labels,
                dim=0,
            )
            .long()
        )

        reliance = (
            torch.cat(
                all_reliance,
                dim=0,
            )
            .reshape(-1)
        )

        harmful = (
            torch.cat(
                all_harmful,
                dim=0,
            )
            .reshape(-1)
        )

        instability = (
            torch.cat(
                all_instability,
                dim=0,
            )
            .reshape(-1)
        )

        (
            num_images,
            num_regions,
            dim,
        ) = regions_img.shape

        regions_flat = (
            regions_img.reshape(
                num_images
                * num_regions,
                dim,
            )
        )

        # --------------------------------------------------------
        # Normalize reliance
        # --------------------------------------------------------
        reliance_scale = (
            torch.quantile(
                reliance,
                self.reliance_quantile,
            )
            .clamp_min(
                self.eps
            )
        )

        reliance_normalized = (
            reliance
            / reliance_scale
        ).clamp(
            min=0.0,
            max=1.0,
        )

        # --------------------------------------------------------
        # Harm gate
        # --------------------------------------------------------
        harmful_gate = (
            torch.sigmoid(
                (
                    harmful
                    - self.harmful_ratio_center
                )
                /
                self.harmful_gate_temperature
            )
        )

        # --------------------------------------------------------
        # Instability
        # --------------------------------------------------------
        instability_factor = (
            self.instability_floor
            +
            (
                1.0
                - self.instability_floor
            )
            * instability
        )

        # --------------------------------------------------------
        # rho_i
        # --------------------------------------------------------
        nuisance_score = (
            reliance_normalized
            * harmful_gate
            * instability_factor
        ).clamp(
            min=0.0,
            max=1.0,
        )

        # --------------------------------------------------------
        # Split candidate pools
        # --------------------------------------------------------
        (
            nuisance_mask,
            reference_mask,
            nuisance_threshold,
            reference_threshold,
        ) = (
            self._extreme_masks(
                score=nuisance_score,
                image_labels=labels_img,
                regions_per_image=num_regions,
            )
        )

        nuisance_pool = (
            regions_flat[
                nuisance_mask
            ]
        )

        reference_pool = (
            regions_flat[
                reference_mask
            ]
        )

        if (
            nuisance_pool.shape[0]
            < 3
        ):
            raise RuntimeError(
                "Too few nuisance "
                "candidate patches."
            )

        if (
            reference_pool.shape[0]
            < 3
        ):
            raise RuntimeError(
                "Too few reference "
                "candidate patches."
            )

        nuisance_pool = (
            self._subsample(
                nuisance_pool,
                self.max_cluster_regions,
                self.random_seed + 11,
            )
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        reference_pool = (
            self._subsample(
                reference_pool,
                self.max_cluster_regions,
                self.random_seed + 17,
            )
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------------
        # AUTO K SEARCH
        # --------------------------------------------------------
        (
            k_nuisance,
            nuisance_ids,
            nuisance_silhouette,
        ) = (
            self._find_best_prototype_k(
                pool=nuisance_pool,
                pool_name="nuisance",
                seed_offset=101,
                verbose=verbose,
            )
        )

        (
            k_reference,
            reference_ids,
            reference_silhouette,
        ) = (
            self._find_best_prototype_k(
                pool=reference_pool,
                pool_name="reference",
                seed_offset=151,
                verbose=verbose,
            )
        )

        nuisance_prototypes = (
            nuisance_pool
            .index_select(
                dim=0,
                index=nuisance_ids,
            )
            .detach()
        )

        reference_prototypes = (
            reference_pool
            .index_select(
                dim=0,
                index=reference_ids,
            )
            .detach()
        )

        # --------------------------------------------------------
        # Store prototype banks
        # --------------------------------------------------------
        self.nuisance_prototypes_raw = (
            nuisance_prototypes
            .cpu()
        )

        self.nuisance_prototypes_norm = (
            F.normalize(
                nuisance_prototypes,
                p=2,
                dim=1,
                eps=self.eps,
            )
            .cpu()
        )

        self.reference_prototypes_raw = (
            reference_prototypes
            .cpu()
        )

        self.reference_prototypes_norm = (
            F.normalize(
                reference_prototypes,
                p=2,
                dim=1,
                eps=self.eps,
            )
            .cpu()
        )

        result = (
            NuisanceDiscoveryResult(
                nuisance_threshold=float(
                    nuisance_threshold
                ),

                reference_threshold=float(
                    reference_threshold
                ),

                num_images=int(
                    num_images
                ),

                num_regions=int(
                    num_images
                    * num_regions
                ),

                num_nuisance_candidates=int(
                    nuisance_mask
                    .sum()
                    .item()
                ),

                num_reference_candidates=int(
                    reference_mask
                    .sum()
                    .item()
                ),

                num_nuisance_prototypes=int(
                    k_nuisance
                ),

                num_reference_prototypes=int(
                    k_reference
                ),

                nuisance_silhouette=float(
                    nuisance_silhouette
                ),

                reference_silhouette=float(
                    reference_silhouette
                ),

                mean_reliance=float(
                    reliance
                    .mean()
                    .item()
                ),

                mean_harmful_ratio=float(
                    harmful
                    .mean()
                    .item()
                ),

                mean_instability=float(
                    instability
                    .mean()
                    .item()
                ),

                mean_nuisance_score=float(
                    nuisance_score
                    .mean()
                    .item()
                ),
            )
        )

        self.discovery_result = (
            result
        )

        if make_assignment_snapshot:
            self.prepare_stage2_assignment_model(
                model=model,
                device=device,
            )

        if verbose:
            print(
                "========== "
                "Nuisance-CnC Discovery "
                "=========="
            )

            print(
                "images={} | "
                "regions/image={} | "
                "total={}".format(
                    num_images,
                    num_regions,
                    num_images
                    * num_regions,
                )
            )

            print(
                "mean R={:.6f} | "
                "mean H={:.6f} | "
                "mean U={:.6f} | "
                "mean rho={:.6f}".format(
                    result.mean_reliance,
                    result.mean_harmful_ratio,
                    result.mean_instability,
                    result.mean_nuisance_score,
                )
            )

            print(
                "nuisance candidates={} | "
                "reference candidates={}".format(
                    result.num_nuisance_candidates,
                    result.num_reference_candidates,
                )
            )

            print(
                "P_N={} (sil={:.6f}) | "
                "P_C={} (sil={:.6f})".format(
                    k_nuisance,
                    nuisance_silhouette,
                    k_reference,
                    reference_silhouette,
                )
            )

            print(
                "==========================================="
            )

        return result


    # ============================================================
    # Frozen Stage-1 assignment model
    # ============================================================

    @torch.no_grad()
    def prepare_stage2_assignment_model(
        self,
        model: nn.Module,
        device=None,
    ):
        """
        This frozen model is NOT a teacher.

        It only provides a fixed Stage-1 coordinate system for:
            patch -> prototype matching
        """
        base = (
            _unwrap_model(
                model
            )
        )

        if device is None:
            device = next(
                base.parameters()
            ).device

        frozen = (
            copy.deepcopy(
                base
            )
            .to(
                device
            )
        )

        frozen.eval()

        for parameter in (
            frozen.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self._assignment_model = (
            frozen
        )

        self._assignment_device = (
            device
        )


    # ============================================================
    # Patch -> prototype
    # ============================================================

    @torch.no_grad()
    def _match_regions(
        self,
        regions: torch.Tensor,
    ):
        if (
            self.nuisance_prototypes_norm
            is None
            or
            self.reference_prototypes_norm
            is None
        ):
            raise RuntimeError(
                "Run discover(...) first."
            )

        (
            batch_size,
            num_regions,
            dim,
        ) = regions.shape

        flat = (
            regions.reshape(
                -1,
                dim,
            )
        )

        flat = F.normalize(
            flat,
            p=2,
            dim=1,
            eps=self.eps,
        )

        nuisance_bank = (
            self.nuisance_prototypes_norm
            .to(
                device=flat.device,
                dtype=flat.dtype,
            )
        )

        reference_bank = (
            self.reference_prototypes_norm
            .to(
                device=flat.device,
                dtype=flat.dtype,
            )
        )

        nuisance_similarity = []
        reference_similarity = []

        for start in range(
            0,
            flat.shape[0],
            self.assignment_chunk_size,
        ):
            end = min(
                start
                + self.assignment_chunk_size,
                flat.shape[0],
            )

            x = (
                flat[
                    start:end
                ]
            )

            nuisance_similarity.append(
                x
                @ nuisance_bank.t()
            )

            reference_similarity.append(
                x
                @ reference_bank.t()
            )

        nuisance_similarity = (
            torch.cat(
                nuisance_similarity,
                dim=0,
            )
        )

        reference_similarity = (
            torch.cat(
                reference_similarity,
                dim=0,
            )
        )

        (
            max_nuisance_similarity,
            nuisance_id,
        ) = (
            nuisance_similarity
            .max(
                dim=1
            )
        )

        max_reference_similarity = (
            reference_similarity
            .max(
                dim=1
            )
            .values
        )

        nuisance_weight = (
            torch.sigmoid(
                (
                    max_nuisance_similarity
                    - max_reference_similarity
                    - self.nuisance_margin
                )
                /
                self.profile_temperature
            )
        )

        if (
            self.min_nuisance_similarity
            is not None
        ):
            absolute_gate = (
                torch.sigmoid(
                    (
                        max_nuisance_similarity
                        - float(
                            self.min_nuisance_similarity
                        )
                    )
                    /
                    self.absolute_gate_temperature
                )
            )

            nuisance_weight = (
                nuisance_weight
                * absolute_gate
            )

        if (
            self.hard_patch_assignment
        ):
            nuisance_probability = (
                F.one_hot(
                    nuisance_id,
                    num_classes=(
                        nuisance_bank.shape[0]
                    ),
                )
                .to(
                    nuisance_similarity.dtype
                )
            )
        else:
            nuisance_probability = (
                F.softmax(
                    nuisance_similarity
                    /
                    self.prototype_temperature,
                    dim=1,
                )
            )

        return (
            nuisance_weight.view(
                batch_size,
                num_regions,
            ),

            nuisance_id.view(
                batch_size,
                num_regions,
            ),

            nuisance_probability.view(
                batch_size,
                num_regions,
                -1,
            ),
        )


    # ============================================================
    # Image-level nuisance profile
    # ============================================================

    @torch.no_grad()
    def nuisance_profile_from_inputs(
        self,
        inputs: torch.Tensor,
    ):
        if (
            self._assignment_model
            is None
        ):
            raise RuntimeError(
                "Run discover(...) or "
                "prepare_stage2_assignment_model(...)."
            )

        assignment_inputs = (
            inputs.to(
                self._assignment_device,
                non_blocking=True,
            )
        )

        _, features = (
            _capture_layer4_and_forward(
                self._assignment_model,
                assignment_inputs,
            )
        )

        regions = (
            _regions_from_features(
                features
            )
        )

        (
            nuisance_weight,
            nuisance_id,
            nuisance_probability,
        ) = (
            self._match_regions(
                regions
            )
        )

        nuisance_mass = (
            nuisance_weight
            .sum(
                dim=1
            )
        )

        nuisance_distribution = (
            nuisance_weight
            .unsqueeze(-1)
            * nuisance_probability
        ).sum(
            dim=1
        )

        nuisance_distribution = (
            nuisance_distribution
            /
            nuisance_mass[
                :,
                None,
            ].clamp_min(
                self.eps
            )
        )

        zero_mass = (
            nuisance_mass
            <= self.eps
        )

        if bool(
            zero_mass
            .any()
            .item()
        ):
            nuisance_distribution[
                zero_mass
            ] = (
                1.0
                /
                nuisance_distribution.shape[1]
            )

        nuisance_strength = (
            nuisance_weight
            .mean(
                dim=1
            )
            .clamp(
                min=0.0,
                max=1.0,
            )
        )

        if (
            self.include_clean_state
        ):
            nuisance_profile = (
                torch.cat(
                    [
                        nuisance_strength[
                            :,
                            None,
                        ]
                        * nuisance_distribution,

                        (
                            1.0
                            - nuisance_strength
                        )[
                            :,
                            None,
                        ],
                    ],
                    dim=1,
                )
            )
        else:
            nuisance_profile = (
                nuisance_distribution
            )

        nuisance_profile = (
            nuisance_profile
            /
            nuisance_profile
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(
                self.eps
            )
        )

        nuisance_mask = (
            nuisance_weight
            >= self.nuisance_patch_threshold
        )

        return {
            "nuisance_profile":
                nuisance_profile,

            "nuisance_strength":
                nuisance_strength,

            "patch_nuisance_weight":
                nuisance_weight,

            "patch_nuisance_id":
                nuisance_id,

            "nuisance_mask":
                nuisance_mask,
        }


    # ============================================================
    # Profile distance
    # ============================================================

    def _pairwise_profile_distance(
        self,
        profile: torch.Tensor,
    ):
        if (
            self.profile_distance
            == "cosine"
        ):
            q = F.normalize(
                profile,
                p=2,
                dim=1,
                eps=self.eps,
            )

            distance = (
                1.0
                - q
                @ q.t()
            )

            return (
                distance.clamp(
                    min=0.0,
                    max=2.0,
                )
            )

        if (
            self.profile_distance
            == "l1"
        ):
            return (
                profile[
                    :,
                    None,
                    :
                ]
                -
                profile[
                    None,
                    :,
                    :
                ]
            ).abs().sum(
                dim=2
            )

        # Jensen-Shannon
        p = (
            profile
            .clamp_min(
                self.eps
            )
        )

        p_i = (
            p[
                :,
                None,
                :
            ]
        )

        p_j = (
            p[
                None,
                :,
                :
            ]
        )

        m = (
            0.5
            * (
                p_i
                + p_j
            )
        )

        kl_i = (
            p_i
            * (
                p_i.log()
                - m.log()
            )
        ).sum(
            dim=2
        )

        kl_j = (
            p_j
            * (
                p_j.log()
                - m.log()
            )
        ).sum(
            dim=2
        )

        return (
            0.5
            * (
                kl_i
                + kl_j
            )
        )


    # ============================================================
    # Pair cap
    # ============================================================

    def _cap_mask(
        self,
        mask: torch.Tensor,
        distance: torch.Tensor,
        max_pairs: Optional[int],
        largest: bool,
    ):
        if max_pairs is None:
            return mask

        max_pairs = int(
            max_pairs
        )

        output = (
            torch.zeros_like(
                mask
            )
        )

        if max_pairs <= 0:
            return output

        for i in range(
            mask.shape[0]
        ):
            ids = (
                mask[i]
                .nonzero(
                    as_tuple=False
                )
                .squeeze(1)
            )

            if ids.numel() == 0:
                continue

            if (
                ids.numel()
                <= max_pairs
            ):
                output[
                    i,
                    ids
                ] = True

                continue

            values = (
                distance[i]
                .index_select(
                    dim=0,
                    index=ids,
                )
            )

            local_ids = (
                torch.topk(
                    values,
                    k=max_pairs,
                    largest=largest,
                    sorted=False,
                )
                .indices
            )

            selected = (
                ids.index_select(
                    dim=0,
                    index=local_ids,
                )
            )

            output[
                i,
                selected
            ] = True

        return output


    # ============================================================
    # Pair mining
    # ============================================================

    def _build_pair_masks(
        self,
        labels: torch.Tensor,
        distance: torch.Tensor,
    ):
        batch_size = (
            labels.shape[0]
        )

        same_class = (
            labels[
                :,
                None
            ]
            .eq(
                labels[
                    None,
                    :
                ]
            )
        )

        identity = torch.eye(
            batch_size,
            device=labels.device,
            dtype=torch.bool,
        )

        positive_mask = (
            same_class
            & (~identity)
            & (
                distance
                >= self.delta_pos
            )
        )

        negative_mask = (
            (~same_class)
            & (
                distance
                <= self.delta_neg
            )
        )

        # --------------------------------------------------------
        # Optional fallback
        # --------------------------------------------------------
        if (
            self.fallback_to_nearest
        ):
            for i in range(
                batch_size
            ):
                same_ids = (
                    (
                        same_class[i]
                        & (~identity[i])
                    )
                    .nonzero(
                        as_tuple=False
                    )
                    .squeeze(1)
                )

                different_ids = (
                    (~same_class[i])
                    .nonzero(
                        as_tuple=False
                    )
                    .squeeze(1)
                )

                if (
                    not bool(
                        positive_mask[
                            i
                        ]
                        .any()
                        .item()
                    )
                    and
                    same_ids.numel() > 0
                ):
                    local_distance = (
                        distance[i]
                        .index_select(
                            dim=0,
                            index=same_ids,
                        )
                    )

                    farthest = (
                        local_distance
                        .argmax()
                    )

                    selected = (
                        same_ids[
                            farthest
                        ]
                    )

                    positive_mask[
                        i,
                        selected
                    ] = True

                if (
                    not bool(
                        negative_mask[
                            i
                        ]
                        .any()
                        .item()
                    )
                    and
                    different_ids.numel() > 0
                ):
                    local_distance = (
                        distance[i]
                        .index_select(
                            dim=0,
                            index=different_ids,
                        )
                    )

                    nearest = (
                        local_distance
                        .argmin()
                    )

                    selected = (
                        different_ids[
                            nearest
                        ]
                    )

                    negative_mask[
                        i,
                        selected
                    ] = True

        positive_mask = (
            self._cap_mask(
                mask=positive_mask,
                distance=distance,
                max_pairs=self.max_positives,
                largest=True,
            )
        )

        negative_mask = (
            self._cap_mask(
                mask=negative_mask,
                distance=distance,
                max_pairs=self.max_negatives,
                largest=False,
            )
        )

        return (
            positive_mask,
            negative_mask,
        )


    # ============================================================
    # CnC loss
    # ============================================================

    def _cnc_loss(
        self,
        representation: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ):
        representation = (
            F.normalize(
                representation,
                p=2,
                dim=1,
                eps=self.eps,
            )
        )

        similarity = (
            representation
            @ representation.t()
        ) / self.temperature

        losses = []

        for i in range(
            representation.shape[0]
        ):
            positive_ids = (
                positive_mask[i]
                .nonzero(
                    as_tuple=False
                )
                .squeeze(1)
            )

            negative_ids = (
                negative_mask[i]
                .nonzero(
                    as_tuple=False
                )
                .squeeze(1)
            )

            if (
                positive_ids.numel() == 0
                or
                negative_ids.numel() == 0
            ):
                continue

            denominator_ids = (
                torch.cat(
                    [
                        positive_ids,
                        negative_ids,
                    ],
                    dim=0,
                )
            )

            positive_logits = (
                similarity[i]
                .index_select(
                    dim=0,
                    index=positive_ids,
                )
            )

            denominator_logits = (
                similarity[i]
                .index_select(
                    dim=0,
                    index=denominator_ids,
                )
            )

            loss_i = (
                -positive_logits.mean()
                +
                torch.logsumexp(
                    denominator_logits,
                    dim=0,
                )
            )

            losses.append(
                loss_i
            )

        if len(losses) == 0:
            zero = (
                representation.sum()
                * 0.0
            )

            return (
                zero,
                0,
            )

        loss = (
            torch.stack(
                losses
            )
            .mean()
        )

        return (
            loss,
            len(losses),
        )


    # ============================================================
    # Stage 2 forward
    # ============================================================

    def forward(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> NuisanceCnCOutput:
        if (
            self.nuisance_prototypes_norm
            is None
        ):
            raise RuntimeError(
                "Run discover(...) first."
            )

        logits, features = (
            _capture_layer4_and_forward(
                student_model,
                inputs,
            )
        )

        labels = (
            labels
            .to(
                logits.device,
                non_blocking=True,
            )
            .view(-1)
            .long()
        )

        if (
            logits.shape[1]
            != self.num_classes
        ):
            raise ValueError(
                "Expected {} classes, "
                "got {}.".format(
                    self.num_classes,
                    logits.shape[1],
                )
            )

        regions = (
            _regions_from_features(
                features
            )
        )

        representation = (
            regions.mean(
                dim=1
            )
        )

        profile_output = (
            self.nuisance_profile_from_inputs(
                inputs
            )
        )

        nuisance_profile = (
            profile_output[
                "nuisance_profile"
            ]
            .to(
                device=representation.device,
                dtype=representation.dtype,
            )
            .detach()
        )

        nuisance_strength = (
            profile_output[
                "nuisance_strength"
            ]
            .to(
                device=representation.device,
                dtype=representation.dtype,
            )
        )

        patch_nuisance_weight = (
            profile_output[
                "patch_nuisance_weight"
            ]
            .to(
                device=representation.device,
                dtype=representation.dtype,
            )
        )

        patch_nuisance_id = (
            profile_output[
                "patch_nuisance_id"
            ]
            .to(
                device=representation.device,
            )
        )

        nuisance_mask = (
            profile_output[
                "nuisance_mask"
            ]
            .to(
                device=representation.device,
            )
        )

        profile_distance = (
            self._pairwise_profile_distance(
                nuisance_profile
            )
        )

        (
            positive_mask,
            negative_mask,
        ) = (
            self._build_pair_masks(
                labels=labels,
                distance=profile_distance,
            )
        )

        (
            loss_cnc,
            num_valid_anchors,
        ) = (
            self._cnc_loss(
                representation=representation,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
            )
        )

        return (
            NuisanceCnCOutput(
                loss_cnc=loss_cnc,

                logits=logits,

                nuisance_profile=(
                    nuisance_profile.detach()
                ),

                nuisance_strength=(
                    nuisance_strength.detach()
                ),

                patch_nuisance_weight=(
                    patch_nuisance_weight.detach()
                ),

                patch_nuisance_id=(
                    patch_nuisance_id.detach()
                ),

                nuisance_mask=(
                    nuisance_mask.detach()
                ),

                positive_mask=(
                    positive_mask.detach()
                ),

                negative_mask=(
                    negative_mask.detach()
                ),

                profile_distance=(
                    profile_distance.detach()
                ),

                num_positive_pairs=int(
                    positive_mask
                    .sum()
                    .item()
                ),

                num_negative_pairs=int(
                    negative_mask
                    .sum()
                    .item()
                ),

                num_valid_anchors=int(
                    num_valid_anchors
                ),

                num_nuisance_patches=int(
                    nuisance_mask
                    .sum()
                    .item()
                ),

                num_total_patches=int(
                    nuisance_mask
                    .numel()
                ),
            )
        )

    __call__ = forward


    # ============================================================
    # Final loss
    # ============================================================

    def combine_with_classification_loss(
        self,
        classification_loss: torch.Tensor,
        output: NuisanceCnCOutput,
        lambda_cnc: Optional[
            float
        ] = None,
    ) -> torch.Tensor:
        if lambda_cnc is None:
            lambda_cnc = (
                self.lambda_cnc
            )

        return (
            classification_loss
            +
            float(
                lambda_cnc
            )
            * output.loss_cnc
        )


    # ============================================================
    # Save / load
    # ============================================================

    def save_discovery(
        self,
        path: str,
    ):
        if (
            self.nuisance_prototypes_raw
            is None
            or
            self.reference_prototypes_raw
            is None
        ):
            raise RuntimeError(
                "Nothing to save. "
                "Run discover(...) first."
            )

        torch.save(
            {
                "num_classes":
                    self.num_classes,

                "prototype_k_factors":
                    self.prototype_k_factors,

                "silhouette_sample_size":
                    self.silhouette_sample_size,

                "nuisance_prototypes_raw":
                    self.nuisance_prototypes_raw,

                "nuisance_prototypes_norm":
                    self.nuisance_prototypes_norm,

                "reference_prototypes_raw":
                    self.reference_prototypes_raw,

                "reference_prototypes_norm":
                    self.reference_prototypes_norm,

                "discovery_result":
                    self.discovery_result,
            },
            path,
        )


    def load_discovery(
        self,
        path: str,
        model_for_assignment: Optional[
            nn.Module
        ] = None,
        device=None,
    ):
        state = torch.load(
            path,
            map_location="cpu",
        )

        if (
            int(
                state[
                    "num_classes"
                ]
            )
            != self.num_classes
        ):
            raise ValueError(
                "num_classes mismatch."
            )

        self.nuisance_prototypes_raw = (
            state[
                "nuisance_prototypes_raw"
            ].float()
        )

        self.nuisance_prototypes_norm = (
            state[
                "nuisance_prototypes_norm"
            ].float()
        )

        self.reference_prototypes_raw = (
            state[
                "reference_prototypes_raw"
            ].float()
        )

        self.reference_prototypes_norm = (
            state[
                "reference_prototypes_norm"
            ].float()
        )

        self.discovery_result = (
            state.get(
                "discovery_result",
                None,
            )
        )

        if (
            model_for_assignment
            is not None
        ):
            self.prepare_stage2_assignment_model(
                model=model_for_assignment,
                device=device,
            )


# ================================================================
# Minimal usage
# ================================================================
#
# from losses.nuisance_cnc_resnet import NuisanceCnCResNet
#
#
# ------------------------------------------------
# 1. num_classes
# ------------------------------------------------
#
# base_model = (
#     model.module
#     if hasattr(model, "module")
#     else model
# )
#
# num_classes = base_model.fc.out_features
#
#
# ------------------------------------------------
# 2. Initialize
# ------------------------------------------------
#
# cnc_model = NuisanceCnCResNet(
#     num_classes=num_classes,
#
#     temperature=0.07,
#     lambda_cnc=0.5,
#
#     profile_distance="cosine",
#     delta_pos=0.20,
#     delta_neg=0.05,
#
#     max_positives=8,
#     max_negatives=32,
#
#     fallback_to_nearest=True,
#
#     profile_temperature=0.10,
#     prototype_temperature=0.10,
#
#     nuisance_margin=0.0,
#     nuisance_patch_threshold=0.50,
#
#     include_clean_state=True,
#
#     intervention_modes=(
#         "zero",
#         "mean",
#         "loo",
#     ),
#
#     reliance_quantile=0.90,
#     harmful_ratio_center=0.50,
#     harmful_gate_temperature=0.10,
#
#     instability_floor=0.50,
#
#     nuisance_top_fraction=0.20,
#     reference_bottom_fraction=0.30,
#
#     class_balanced_discovery=True,
#
#     # ==========================================================
#     # IMPORTANT:
#     # num_classes=8
#     # -> K = [8,16,24,32]
#     # ==========================================================
#     prototype_k_factors=[
#         1,
#         2,
#         3,
#         4,
#     ],
#
#     silhouette_sample_size=3000,
#
#     kmedoids_iterations=30,
#     max_cluster_regions=20000,
#
#     random_seed=0,
# )
#
#
# ------------------------------------------------
# 3. Run discovery ONCE
# ------------------------------------------------
#
# result = cnc_model.discover(
#     model=model,
#     reference_loader=val_loader,
#     device=device,
#     verbose=True,
#     make_assignment_snapshot=True,
# )
#
# print(
#     result.summary()
# )
#
# cnc_model.save_discovery(
#     "./nuisance_cnc_discovery.pth"
# )
#
#
# ------------------------------------------------
# 4. Stage-2 training
# ------------------------------------------------
#
# for batch_idx, data_tuple in enumerate(
#     trainloader
# ):
#
#     inputs = data_tuple[0].to(
#         device,
#         non_blocking=True,
#     )
#
#     labels = data_tuple[1].to(
#         device,
#         non_blocking=True,
#     ).long()
#
#     output = cnc_model(
#         student_model=model,
#         inputs=inputs,
#         labels=labels,
#     )
#
#     loss_cls = F.cross_entropy(
#         output.logits,
#         labels,
#     )
#
#     loss = (
#         cnc_model
#         .combine_with_classification_loss(
#             classification_loss=loss_cls,
#             output=output,
#         )
#     )
#
#     optimizer.zero_grad(
#         set_to_none=True
#     )
#
#     loss.backward()
#
#     optimizer.step()
#
#     if batch_idx % 50 == 0:
#
#         print(
#             "loss={:.4f} | "
#             "cls={:.4f} | "
#             "cnc={:.4f} | "
#             "pos={} | "
#             "neg={} | "
#             "anchors={}".format(
#                 loss.item(),
#                 loss_cls.item(),
#                 output.loss_cnc.item(),
#                 output.num_positive_pairs,
#                 output.num_negative_pairs,
#                 output.num_valid_anchors,
#             )
#         )
#
#
# ------------------------------------------------
# 5. Load in a later run
# ------------------------------------------------
#
# IMPORTANT:
# model_for_assignment must be the Stage-1 model snapshot
# corresponding to the feature space used for discovery.
#
# cnc_model.load_discovery(
#     "./nuisance_cnc_discovery.pth",
#     model_for_assignment=model,
#     device=device,
# )
#
