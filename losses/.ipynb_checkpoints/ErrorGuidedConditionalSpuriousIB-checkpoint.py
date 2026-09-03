from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CSIBOutput:
    """Output of the RaVL-guided conditional spurious information bottleneck."""

    loss: torch.Tensor
    logits: torch.Tensor
    misclassified_mask: torch.Tensor              # [B]
    valid_mask: torch.Tensor                      # [B]
    spurious_features: torch.Tensor               # [B, C_l]
    hard_spurious_labels: torch.Tensor             # [B], invalid=-1
    soft_spurious_targets: torch.Tensor            # [B, K]
    rho_maps: torch.Tensor                         # [B, H, W]
    error_maps: torch.Tensor                       # [B, H, W]
    selected_region_counts: torch.Tensor           # [B]
    region_features: torch.Tensor                  # [N_r, C_l]
    region_cluster_labels: torch.Tensor            # [N_r]
    region_sample_indices: torch.Tensor             # [N_r]
    region_weights: torch.Tensor                   # [N_r]
    region_similarities: torch.Tensor              # [N_r]
    influence_scores: torch.Tensor                 # [K]
    gap_scores: torch.Tensor                       # [K]
    spurious_cluster_mask: torch.Tensor            # [K], bool

    def statistics(self) -> Dict[str, float]:
        valid_targets = self.soft_spurious_targets[self.valid_mask]
        active_clusters = (
            int((valid_targets.sum(dim=0) > 0).sum().item())
            if valid_targets.numel() > 0
            else 0
        )
        return {
            "loss_csib": float(self.loss.detach().item()),
            "num_misclassified": float(self.misclassified_mask.sum().item()),
            "num_valid_samples": float(self.valid_mask.sum().item()),
            "num_selected_regions": float(self.region_cluster_labels.numel()),
            "num_active_clusters_in_batch": float(active_clusters),
            "num_ravl_spurious_clusters": float(
                self.spurious_cluster_mask.sum().item()
            ),
            "mean_selected_regions": (
                float(
                    self.selected_region_counts[self.valid_mask]
                    .float()
                    .mean()
                    .item()
                )
                if self.valid_mask.any()
                else 0.0
            ),
            "mean_region_weight": (
                float(self.region_weights.mean().item())
                if self.region_weights.numel() > 0
                else 0.0
            ),
            "mean_prototype_similarity": (
                float(self.region_similarities.mean().item())
                if self.region_similarities.numel() > 0
                else 0.0
            ),
            "max_influence_score": (
                float(self.influence_scores.max().item())
                if self.influence_scores.numel() > 0
                else 0.0
            ),
            "max_gap_score": (
                float(self.gap_scores.max().item())
                if self.gap_scores.numel() > 0
                else 0.0
            ),
        }


def _unwrap_tensor_output(output: object, name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    raise TypeError(
        f"{name} must be a Tensor or a tuple/list whose first item is a Tensor."
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _resolve_resnet_layer(model: nn.Module, layer: int) -> nn.Module:
    if layer not in (1, 2, 3, 4):
        raise ValueError("layer must be one of {1, 2, 3, 4}.")
    base_model = _unwrap_model(model)
    name = f"layer{layer}"
    if not hasattr(base_model, name):
        raise AttributeError(
            f"The supplied model has no attribute '{name}'. "
            "A timm/torchvision ResNet-like model is required."
        )
    module = getattr(base_model, name)
    if not isinstance(module, nn.Module):
        raise TypeError(f"model.{name} is not an nn.Module.")
    return module


def _get_classifier_weight(model: nn.Module) -> torch.Tensor:
    """Locate final linear classifier weight [num_classes, feature_dim]."""
    base_model = _unwrap_model(model)
    candidates = []

    if hasattr(base_model, "get_classifier"):
        try:
            candidates.append(base_model.get_classifier())
        except Exception:
            pass

    for name in ("fc", "head", "classifier"):
        if hasattr(base_model, name):
            candidates.append(getattr(base_model, name))

    for candidate in candidates:
        if isinstance(candidate, nn.Linear):
            return candidate.weight
        if isinstance(candidate, nn.Sequential):
            for submodule in reversed(candidate):
                if isinstance(submodule, nn.Linear):
                    return submodule.weight
        weight = getattr(candidate, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return weight

    raise AttributeError("Unable to locate the final linear classifier weight.")


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def grad_reverse(x: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, coefficient)


class OnlineRegionClusterBank(nn.Module):
    """One-time spherical K-Means prototypes followed by slow EMA updates."""

    def __init__(
        self,
        num_clusters: int,
        feature_dim: int,
        momentum: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1).")

        self.num_clusters = int(num_clusters)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer(
            "prototypes", torch.zeros(num_clusters, feature_dim), persistent=True
        )
        self.register_buffer(
            "initialized", torch.zeros(num_clusters, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "assignment_counts", torch.zeros(num_clusters, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "kmeans_initialized", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "influence_scores", torch.zeros(num_clusters), persistent=True
        )
        self.register_buffer(
            "gap_scores", torch.zeros(num_clusters), persistent=True
        )
        self.register_buffer(
            "spurious_cluster_mask",
            torch.zeros(num_clusters, dtype=torch.bool),
            persistent=True,
        )

    @torch.no_grad()
    def reset_state(self) -> None:
        self.prototypes.zero_()
        self.initialized.zero_()
        self.assignment_counts.zero_()
        self.kmeans_initialized.fill_(False)
        self.influence_scores.zero_()
        self.gap_scores.zero_()
        self.spurious_cluster_mask.zero_()

    @torch.no_grad()
    def initialize_from_features(
        self,
        region_features: torch.Tensor,
        num_iterations: int = 50,
    ) -> torch.Tensor:
        """Run deterministic spherical K-Means and return labels [N]."""
        if region_features.ndim != 2:
            raise ValueError("region_features must have shape [N_region, C].")
        if region_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected region feature dim {self.feature_dim}, got "
                f"{region_features.shape[1]}."
            )
        if region_features.shape[0] < self.num_clusters:
            raise RuntimeError(
                "The warm-up collection epoch produced fewer candidate regions "
                f"({region_features.shape[0]}) than num_clusters "
                f"({self.num_clusters}). Reduce num_clusters or relax "
                "region_threshold."
            )
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

        normalized = F.normalize(
            region_features.detach().to(dtype=torch.float32),
            p=2,
            dim=1,
            eps=self.eps,
        )

        selected_indices = [0]
        first_center = normalized[0:1]
        min_distance = 1.0 - (
            normalized @ first_center.transpose(0, 1)
        ).squeeze(1)

        for _ in range(1, self.num_clusters):
            next_index = int(min_distance.argmax().item())
            selected_indices.append(next_index)
            next_center = normalized[next_index : next_index + 1]
            next_distance = 1.0 - (
                normalized @ next_center.transpose(0, 1)
            ).squeeze(1)
            min_distance = torch.minimum(min_distance, next_distance)

        center_indices = torch.tensor(
            selected_indices, device=normalized.device, dtype=torch.long
        )
        centers = normalized.index_select(0, center_indices).clone()
        previous_labels: Optional[torch.Tensor] = None

        for _ in range(num_iterations):
            similarity = normalized @ centers.transpose(0, 1)
            labels = similarity.argmax(dim=1)

            if previous_labels is not None and torch.equal(labels, previous_labels):
                break
            previous_labels = labels

            new_centers = []
            max_similarity = similarity.max(dim=1).values
            for cluster_id in range(self.num_clusters):
                mask = labels.eq(cluster_id)
                if mask.any():
                    center = normalized[mask].mean(dim=0)
                else:
                    replacement_index = int(max_similarity.argmin().item())
                    center = normalized[replacement_index]
                new_centers.append(
                    F.normalize(center, p=2, dim=0, eps=self.eps)
                )
            centers = torch.stack(new_centers, dim=0)

        final_similarity = normalized @ centers.transpose(0, 1)
        final_labels = final_similarity.argmax(dim=1)
        final_counts = torch.bincount(final_labels, minlength=self.num_clusters)

        self.prototypes.copy_(
            centers.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        )
        self.initialized.fill_(True)
        self.assignment_counts.copy_(final_counts.to(self.assignment_counts.device))
        self.kmeans_initialized.fill_(True)
        return final_labels

    @torch.no_grad()
    def assign_with_similarity(
        self,
        region_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if region_features.ndim != 2:
            raise ValueError("region_features must have shape [N_region, C].")
        if region_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected region feature dim {self.feature_dim}, got "
                f"{region_features.shape[1]}."
            )
        if region_features.shape[0] == 0:
            empty_labels = torch.empty(
                0, device=region_features.device, dtype=torch.long
            )
            empty_similarity = region_features.new_zeros((0,))
            return empty_labels, empty_similarity
        if not bool(self.kmeans_initialized.item()):
            raise RuntimeError(
                "The prototype bank has not been initialized by one-time K-Means."
            )

        normalized = F.normalize(
            region_features.detach(), p=2, dim=1, eps=self.eps
        )
        prototypes = F.normalize(
            self.prototypes.to(device=normalized.device, dtype=normalized.dtype),
            p=2,
            dim=1,
            eps=self.eps,
        )
        similarity = normalized @ prototypes.transpose(0, 1)
        max_similarity, labels = similarity.max(dim=1)
        return labels, max_similarity

    @torch.no_grad()
    def assign(self, region_features: torch.Tensor) -> torch.Tensor:
        labels, _ = self.assign_with_similarity(region_features)
        return labels

    @torch.no_grad()
    def set_ravl_statistics(
        self,
        influence_scores: torch.Tensor,
        gap_scores: torch.Tensor,
        spurious_cluster_mask: torch.Tensor,
    ) -> None:
        if influence_scores.shape != (self.num_clusters,):
            raise ValueError("influence_scores must have shape [K].")
        if gap_scores.shape != (self.num_clusters,):
            raise ValueError("gap_scores must have shape [K].")
        if spurious_cluster_mask.shape != (self.num_clusters,):
            raise ValueError("spurious_cluster_mask must have shape [K].")
        self.influence_scores.copy_(
            influence_scores.to(self.influence_scores.device)
        )
        self.gap_scores.copy_(gap_scores.to(self.gap_scores.device))
        self.spurious_cluster_mask.copy_(
            spurious_cluster_mask.to(self.spurious_cluster_mask.device)
        )

    @torch.no_grad()
    def update(
        self,
        region_features: torch.Tensor,
        region_labels: torch.Tensor,
        region_similarities: Optional[torch.Tensor] = None,
        similarity_threshold: float = -1.0,
    ) -> None:
        if region_features.shape[0] == 0:
            return
        if not bool(self.kmeans_initialized.item()):
            raise RuntimeError(
                "EMA prototype updates require one-time K-Means initialization."
            )

        normalized = F.normalize(
            region_features.detach(), p=2, dim=1, eps=self.eps
        )
        update_mask = torch.ones(
            region_features.shape[0], device=region_features.device, dtype=torch.bool
        )
        if region_similarities is not None:
            update_mask &= region_similarities.detach().ge(similarity_threshold)

        for cluster_id_tensor in region_labels.unique():
            cluster_id = int(cluster_id_tensor.item())
            mask = region_labels.eq(cluster_id) & update_mask
            if not mask.any():
                continue
            mean_feature = F.normalize(
                normalized[mask].mean(dim=0), p=2, dim=0, eps=self.eps
            )
            new_prototype = (
                self.momentum * self.prototypes[cluster_id]
                + (1.0 - self.momentum) * mean_feature.to(
                    device=self.prototypes.device,
                    dtype=self.prototypes.dtype,
                )
            )
            self.prototypes[cluster_id] = F.normalize(
                new_prototype, p=2, dim=0, eps=self.eps
            )
            self.assignment_counts[cluster_id] += int(mask.sum().item())


class ConditionalSpuriousDiscriminator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_clusters: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.network = nn.Sequential(
            nn.Linear(feature_dim + num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
        )

    def forward(
        self,
        spurious_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        condition = F.one_hot(
            labels, num_classes=self.num_classes
        ).to(dtype=spurious_features.dtype)
        return self.network(torch.cat([spurious_features, condition], dim=1))


class ErrorGuidedConditionalSpuriousIB(nn.Module):
    """
    RaVL-guided conditional spurious information bottleneck.

    Training schedule for 1-based epoch indices and warm_up=20:
      epochs 1--19: classification representation learning only;
      epoch 20: collect a full epoch of candidate local regions and metadata;
      first call in epoch 21: spherical K-Means + RaVL-style H/G validation;
      epoch 21 onward: use only validated spurious prototypes to construct S,
      optimize the conditional adversarial surrogate of I(Z_spur; S | Y),
      and slowly update prototypes by EMA.

    The public training path is a single call to ``forward_from_model``.
    """

    def __init__(
        self,
        num_classes: int,
        layer_dims: Sequence[int],
        num_clusters: int = 8,
        cluster_momentum: float = 0.99,
        warm_up: int = 20,
        discriminator_hidden_dim: int = 256,
        discriminator_dropout: float = 0.1,
        error_threshold: float = 0.0,
        gate_temperature: float = 0.2,
        region_threshold: float = 0.5,
        max_regions_per_sample: Optional[int] = 32,
        target_mode: str = "soft",
        class_direction_mode: str = "auto",
        min_valid_samples: int = 2,
        min_active_clusters: int = 2,
        detach_region_weights: bool = True,
        influence_threshold: float = 0.25,
        gap_threshold: float = 0.0,
        num_spurious_clusters: int = 3,
        prototype_similarity_threshold: float = 0.2,
        min_spurious_clusters: int = 2,
        kmeans_iterations: int = 50,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if len(layer_dims) != 4:
            raise ValueError("layer_dims must contain four stage dimensions.")
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2.")
        if warm_up <= 1:
            raise ValueError("warm_up must be greater than 1.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if not 0.0 < region_threshold < 1.0:
            raise ValueError("region_threshold must lie in (0, 1).")
        if max_regions_per_sample is not None and max_regions_per_sample <= 0:
            raise ValueError("max_regions_per_sample must be positive or None.")
        if target_mode not in {"hard", "soft"}:
            raise ValueError("target_mode must be 'hard' or 'soft'.")
        if class_direction_mode not in {"auto", "classifier", "gradient"}:
            raise ValueError(
                "class_direction_mode must be 'auto', 'classifier', or 'gradient'."
            )
        if min_valid_samples <= 0:
            raise ValueError("min_valid_samples must be positive.")
        if min_active_clusters < 2:
            raise ValueError("min_active_clusters must be at least 2.")
        if not 0.0 <= influence_threshold <= 1.0:
            raise ValueError("influence_threshold must lie in [0, 1].")
        if gap_threshold < 0:
            raise ValueError("gap_threshold must be non-negative.")
        if not 1 <= num_spurious_clusters <= num_clusters:
            raise ValueError("num_spurious_clusters must lie in [1, num_clusters].")
        if not 2 <= min_spurious_clusters <= num_clusters:
            raise ValueError("min_spurious_clusters must lie in [2, num_clusters].")
        if kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive.")

        self.num_classes = int(num_classes)
        self.layer_dims = tuple(int(dim) for dim in layer_dims)
        self.num_clusters = int(num_clusters)
        self.warm_up = int(warm_up)
        self.error_threshold = float(error_threshold)
        self.gate_temperature = float(gate_temperature)
        self.region_threshold = float(region_threshold)
        self.max_regions_per_sample = max_regions_per_sample
        self.target_mode = target_mode
        self.class_direction_mode = class_direction_mode
        self.min_valid_samples = int(min_valid_samples)
        self.min_active_clusters = int(min_active_clusters)
        self.detach_region_weights = bool(detach_region_weights)
        self.influence_threshold = float(influence_threshold)
        self.gap_threshold = float(gap_threshold)
        self.num_spurious_clusters = int(num_spurious_clusters)
        self.prototype_similarity_threshold = float(
            prototype_similarity_threshold
        )
        self.min_spurious_clusters = int(min_spurious_clusters)
        self.kmeans_iterations = int(kmeans_iterations)
        self.eps = float(eps)

        self.cluster_banks = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

        self._warmup_region_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_region_weight_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_region_error_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_region_sample_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_sample_label_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_sample_correct_memory: Dict[str, List[torch.Tensor]] = {
            str(i): [] for i in range(1, 5)
        }
        self._warmup_sample_counter: Dict[str, int] = {
            str(i): 0 for i in range(1, 5)
        }

        for layer_index, feature_dim in enumerate(self.layer_dims, start=1):
            key = str(layer_index)
            self.cluster_banks[key] = OnlineRegionClusterBank(
                num_clusters=num_clusters,
                feature_dim=feature_dim,
                momentum=cluster_momentum,
                eps=eps,
            )
            self.discriminators[key] = ConditionalSpuriousDiscriminator(
                feature_dim=feature_dim,
                num_classes=num_classes,
                num_clusters=num_clusters,
                hidden_dim=discriminator_hidden_dim,
                dropout=discriminator_dropout,
            )

    @torch.no_grad()
    def reset_state(self) -> None:
        for bank in self.cluster_banks.values():
            bank.reset_state()
        for key in self._warmup_region_memory:
            self._warmup_region_memory[key].clear()
            self._warmup_region_weight_memory[key].clear()
            self._warmup_region_error_memory[key].clear()
            self._warmup_region_sample_memory[key].clear()
            self._warmup_sample_label_memory[key].clear()
            self._warmup_sample_correct_memory[key].clear()
            self._warmup_sample_counter[key] = 0

    def _capture_layer_and_forward(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_module = _resolve_resnet_layer(student_model, layer)
        holder: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            holder["features"] = _unwrap_tensor_output(
                output, f"student_model.layer{layer} output"
            )

        handle = selected_module.register_forward_hook(hook)
        try:
            logits = _unwrap_tensor_output(
                student_model(inputs), "student_model(inputs)"
            )
        finally:
            handle.remove()

        if "features" not in holder:
            raise RuntimeError(f"Failed to capture layer{layer} features.")
        features = holder["features"]
        if features.ndim != 4:
            raise ValueError(
                f"layer{layer} features must be [B,C,H,W], got "
                f"{tuple(features.shape)}."
            )
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}.")
        expected_dim = self.layer_dims[layer - 1]
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"layer{layer} channel dim is {features.shape[1]}, but "
                f"layer_dims specifies {expected_dim}."
            )
        return logits, features

    def _compute_class_directions(
        self,
        student_model: nn.Module,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        mis_indices: torch.Tensor,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Return ground-truth and competitor directions for selected samples.

        The argument names are preserved from the earlier implementation.
        Here ``predictions`` can contain the strongest non-ground-truth class
        for correctly classified samples, allowing RaVL-style presence/gap
        statistics to use every sample in the collection epoch.
        """
        mode = self.class_direction_mode
        classifier_weight: Optional[torch.Tensor] = None

        if mode in {"auto", "classifier"}:
            try:
                candidate = _get_classifier_weight(student_model)
                if candidate.shape[0] == self.num_classes and (
                    candidate.shape[1] == features.shape[1]
                ):
                    classifier_weight = candidate
            except AttributeError:
                classifier_weight = None

        use_classifier = classifier_weight is not None and mode != "gradient"
        if mode == "classifier" and not use_classifier:
            raise ValueError(
                "The final classifier weight dimension does not match the "
                f"selected layer{layer}. Use layer=4 or "
                "class_direction_mode='gradient'."
            )

        if use_classifier:
            weight = classifier_weight.to(
                device=features.device, dtype=features.dtype
            )
            gt_direction = weight.index_select(0, labels[mis_indices])
            err_direction = weight.index_select(0, predictions[mis_indices])
            return gt_direction.detach(), err_direction.detach(), "classifier"

        true_scores = logits[mis_indices, labels[mis_indices]]
        error_scores = logits[mis_indices, predictions[mis_indices]]

        gt_gradient = torch.autograd.grad(
            outputs=true_scores.sum(),
            inputs=features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].index_select(0, mis_indices)
        err_gradient = torch.autograd.grad(
            outputs=error_scores.sum(),
            inputs=features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].index_select(0, mis_indices)

        gt_direction = gt_gradient.mean(dim=(2, 3)).detach()
        err_direction = err_gradient.mean(dim=(2, 3)).detach()
        return gt_direction, err_direction, "gradient"

    def _compute_error_and_rho_maps(
        self,
        local_features: torch.Tensor,
        gt_directions: torch.Tensor,
        err_directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        regions = local_features.flatten(2).transpose(1, 2)
        regions_norm = F.normalize(regions, p=2, dim=2, eps=self.eps)
        gt_norm = F.normalize(gt_directions, p=2, dim=1, eps=self.eps)
        err_norm = F.normalize(err_directions, p=2, dim=1, eps=self.eps)

        a_gt = torch.sum(regions_norm * gt_norm.unsqueeze(1), dim=2)
        a_err = torch.sum(regions_norm * err_norm.unsqueeze(1), dim=2)
        error_scores = a_err - a_gt
        rho = torch.sigmoid(
            (error_scores - self.error_threshold) / self.gate_temperature
        )
        height, width = local_features.shape[2:]
        return (
            error_scores.view(-1, height, width),
            rho.view(-1, height, width),
        )

    def _collect_candidate_regions(
        self,
        mis_features: torch.Tensor,
        rho_maps: torch.Tensor,
        mis_indices: torch.Tensor,
        error_maps: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Collect local regions satisfying rho > region_threshold."""
        num_samples, channels, _, _ = mis_features.shape
        flat_features = mis_features.flatten(2).transpose(1, 2)
        flat_rho = rho_maps.flatten(1)
        flat_error = (
            error_maps.flatten(1)
            if error_maps is not None
            else torch.zeros_like(flat_rho)
        )

        feature_list = []
        weight_list = []
        error_list = []
        local_sample_list = []
        full_sample_list = []

        for local_sample_index in range(num_samples):
            sample_rho = flat_rho[local_sample_index]
            candidate_indices = sample_rho.gt(self.region_threshold).nonzero(
                as_tuple=False
            ).squeeze(1)
            if candidate_indices.numel() == 0:
                continue

            if (
                self.max_regions_per_sample is not None
                and candidate_indices.numel() > self.max_regions_per_sample
            ):
                candidate_rho = sample_rho.index_select(0, candidate_indices)
                top_relative = torch.topk(
                    candidate_rho,
                    k=self.max_regions_per_sample,
                    largest=True,
                    sorted=False,
                ).indices
                candidate_indices = candidate_indices.index_select(0, top_relative)

            selected_features = flat_features[local_sample_index].index_select(
                0, candidate_indices
            )
            selected_rho = sample_rho.index_select(0, candidate_indices)
            selected_error = flat_error[local_sample_index].index_select(
                0, candidate_indices
            )
            if self.detach_region_weights:
                selected_rho = selected_rho.detach()

            num_selected = int(candidate_indices.numel())
            feature_list.append(selected_features)
            weight_list.append(selected_rho)
            error_list.append(selected_error.detach())
            local_sample_list.append(
                torch.full(
                    (num_selected,),
                    local_sample_index,
                    device=mis_features.device,
                    dtype=torch.long,
                )
            )
            full_sample_list.append(
                torch.full(
                    (num_selected,),
                    int(mis_indices[local_sample_index].item()),
                    device=mis_features.device,
                    dtype=torch.long,
                )
            )

        if not feature_list:
            return (
                mis_features.new_zeros((0, channels)),
                mis_features.new_zeros((0,)),
                mis_features.new_zeros((0,)),
                torch.empty(0, device=mis_features.device, dtype=torch.long),
                torch.empty(0, device=mis_features.device, dtype=torch.long),
            )

        return (
            torch.cat(feature_list, dim=0),
            torch.cat(weight_list, dim=0),
            torch.cat(error_list, dim=0),
            torch.cat(local_sample_list, dim=0),
            torch.cat(full_sample_list, dim=0),
        )

    @torch.no_grad()
    def _append_warmup_regions(
        self,
        layer_key: str,
        region_features: torch.Tensor,
        region_weights: torch.Tensor,
        region_errors: torch.Tensor,
        region_full_sample_indices: torch.Tensor,
        labels: torch.Tensor,
        correct_mask: torch.Tensor,
    ) -> None:
        batch_size = int(labels.shape[0])
        sample_offset = self._warmup_sample_counter[layer_key]

        self._warmup_sample_label_memory[layer_key].append(
            labels.detach().to(device="cpu", dtype=torch.long)
        )
        self._warmup_sample_correct_memory[layer_key].append(
            correct_mask.detach().to(device="cpu", dtype=torch.bool)
        )

        if region_features.shape[0] > 0:
            normalized = F.normalize(
                region_features.detach().to(dtype=torch.float32),
                p=2,
                dim=1,
                eps=self.eps,
            ).to(device="cpu", dtype=torch.float16)
            self._warmup_region_memory[layer_key].append(normalized)
            self._warmup_region_weight_memory[layer_key].append(
                region_weights.detach().to(device="cpu", dtype=torch.float32)
            )
            self._warmup_region_error_memory[layer_key].append(
                region_errors.detach().to(device="cpu", dtype=torch.float32)
            )
            self._warmup_region_sample_memory[layer_key].append(
                region_full_sample_indices.detach().to(device="cpu", dtype=torch.long)
                + sample_offset
            )

        self._warmup_sample_counter[layer_key] += batch_size

    @staticmethod
    def _safe_minmax(x: torch.Tensor, eps: float) -> torch.Tensor:
        if x.numel() == 0:
            return x
        xmin = x.min()
        xmax = x.max()
        if float((xmax - xmin).abs().item()) <= eps:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin).clamp_min(eps)

    @torch.no_grad()
    def _compute_ravl_statistics(
        self,
        region_cluster_labels: torch.Tensor,
        region_errors: torch.Tensor,
        region_sample_indices: torch.Tensor,
        sample_labels: torch.Tensor,
        sample_correct: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RaVL-inspired influence H_k and presence-gap G_k."""
        device = region_cluster_labels.device
        num_samples = int(sample_labels.shape[0])
        K = self.num_clusters
        C = self.num_classes

        presence = torch.zeros(num_samples, K, device=device, dtype=torch.bool)
        if region_cluster_labels.numel() > 0:
            flat = region_sample_indices * K + region_cluster_labels
            presence.view(-1)[flat.unique()] = True

        # H_k^c: among misclassified samples of class c, which cluster owns
        # the strongest local error-supporting region?
        influence_by_class = torch.zeros(C, K, device=device)
        for class_id in range(C):
            class_mis = sample_labels.eq(class_id) & (~sample_correct)
            mis_ids = class_mis.nonzero(as_tuple=False).squeeze(1)
            denominator = int(mis_ids.numel())
            if denominator == 0:
                continue

            counts = torch.zeros(K, device=device)
            for sample_id_tensor in mis_ids:
                sample_id = int(sample_id_tensor.item())
                region_mask = region_sample_indices.eq(sample_id)
                if not region_mask.any():
                    continue
                sample_errors = region_errors[region_mask]
                sample_clusters = region_cluster_labels[region_mask]
                top_cluster = sample_clusters[sample_errors.argmax()]
                counts[top_cluster] += 1.0
            influence_by_class[class_id] = counts / float(denominator)

        influence_scores = influence_by_class.max(dim=0).values

        # G_k: class-conditional accuracy difference between images containing
        # cluster k and images not containing cluster k, with RaVL balancing.
        gap_scores = torch.zeros(K, device=device)
        for cluster_id in range(K):
            total_gap = torch.tensor(0.0, device=device)
            for class_id in range(C):
                class_mask = sample_labels.eq(class_id)
                in_mask = class_mask & presence[:, cluster_id]
                out_mask = class_mask & (~presence[:, cluster_id])
                n_in = int(in_mask.sum().item())
                n_out = int(out_mask.sum().item())
                if n_in == 0 or n_out == 0:
                    continue
                acc_in = sample_correct[in_mask].float().mean()
                acc_out = sample_correct[out_mask].float().mean()
                balance = 2.0 * min(n_in, n_out) / float(n_in + n_out)
                total_gap = total_gap + balance * (acc_in - acc_out).abs()
            gap_scores[cluster_id] = total_gap

        threshold_mask = (
            influence_scores.ge(self.influence_threshold)
            & gap_scores.ge(self.gap_threshold)
        )

        h_norm = self._safe_minmax(influence_scores, self.eps)
        g_norm = self._safe_minmax(gap_scores, self.eps)
        combined = h_norm * g_norm
        if float(combined.max().item()) <= self.eps:
            combined = h_norm + g_norm

        target_count = min(self.num_spurious_clusters, K)
        selected = threshold_mask.clone()

        if int(selected.sum().item()) > target_count:
            candidate_ids = selected.nonzero(as_tuple=False).squeeze(1)
            candidate_scores = combined.index_select(0, candidate_ids)
            keep_local = torch.topk(
                candidate_scores, k=target_count, largest=True
            ).indices
            new_selected = torch.zeros_like(selected)
            new_selected[candidate_ids.index_select(0, keep_local)] = True
            selected = new_selected

        required = min(self.min_spurious_clusters, target_count)
        if int(selected.sum().item()) < required:
            fallback_ids = torch.topk(
                combined, k=required, largest=True
            ).indices
            selected[fallback_ids] = True

        return influence_scores, gap_scores, selected

    @torch.no_grad()
    def _initialize_cluster_bank_from_memory(self, layer_key: str) -> None:
        cluster_bank = self.cluster_banks[layer_key]
        if bool(cluster_bank.kmeans_initialized.item()):
            return

        if not self._warmup_sample_label_memory[layer_key]:
            raise RuntimeError(
                f"No warm-up samples were collected for layer{layer_key}. "
                "Pass a 1-based epoch index and run the full warm-up epoch."
            )
        if not self._warmup_region_memory[layer_key]:
            raise RuntimeError(
                f"No candidate regions were collected in epoch {self.warm_up} "
                f"for layer{layer_key}. Relax region_threshold or "
                "error_threshold."
            )

        all_regions = torch.cat(
            self._warmup_region_memory[layer_key], dim=0
        ).to(dtype=torch.float32)
        all_weights = torch.cat(
            self._warmup_region_weight_memory[layer_key], dim=0
        )
        all_errors = torch.cat(
            self._warmup_region_error_memory[layer_key], dim=0
        )
        all_region_samples = torch.cat(
            self._warmup_region_sample_memory[layer_key], dim=0
        )
        all_sample_labels = torch.cat(
            self._warmup_sample_label_memory[layer_key], dim=0
        )
        all_sample_correct = torch.cat(
            self._warmup_sample_correct_memory[layer_key], dim=0
        )

        del all_weights  # weights are not needed for H/G validation.

        region_labels = cluster_bank.initialize_from_features(
            region_features=all_regions,
            num_iterations=self.kmeans_iterations,
        )
        influence, gap, spurious_mask = self._compute_ravl_statistics(
            region_cluster_labels=region_labels,
            region_errors=all_errors,
            region_sample_indices=all_region_samples,
            sample_labels=all_sample_labels,
            sample_correct=all_sample_correct,
        )
        cluster_bank.set_ravl_statistics(
            influence_scores=influence,
            gap_scores=gap,
            spurious_cluster_mask=spurious_mask,
        )

        self._warmup_region_memory[layer_key].clear()
        self._warmup_region_weight_memory[layer_key].clear()
        self._warmup_region_error_memory[layer_key].clear()
        self._warmup_region_sample_memory[layer_key].clear()
        self._warmup_sample_label_memory[layer_key].clear()
        self._warmup_sample_correct_memory[layer_key].clear()
        self._warmup_sample_counter[layer_key] = 0

    def _aggregate_sample_spurious_variable(
        self,
        region_features: torch.Tensor,
        region_weights: torch.Tensor,
        region_local_sample_indices: torch.Tensor,
        region_cluster_labels: torch.Tensor,
        num_misclassified: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = region_features.device
        dtype = region_features.dtype
        feature_dim = region_features.shape[1]

        cluster_mass = torch.zeros(
            num_misclassified,
            self.num_clusters,
            device=device,
            dtype=dtype,
        )
        flat_indices = (
            region_local_sample_indices * self.num_clusters
            + region_cluster_labels
        )
        cluster_mass.view(-1).scatter_add_(
            0, flat_indices, region_weights.to(dtype)
        )

        mass_sum = cluster_mass.sum(dim=1, keepdim=True)
        valid_local_mask = mass_sum.squeeze(1).gt(self.eps)
        soft_targets = cluster_mass / mass_sum.clamp_min(self.eps)
        hard_labels = soft_targets.argmax(dim=1)
        hard_labels = hard_labels.masked_fill(~valid_local_mask, -1)

        weighted_regions = region_features * region_weights.unsqueeze(1)
        weighted_sum = torch.zeros(
            num_misclassified,
            feature_dim,
            device=device,
            dtype=dtype,
        )
        weighted_sum.index_add_(
            0, region_local_sample_indices, weighted_regions
        )
        weight_sum = torch.zeros(
            num_misclassified, device=device, dtype=dtype
        )
        weight_sum.index_add_(
            0, region_local_sample_indices, region_weights.to(dtype)
        )
        spurious_features = weighted_sum / weight_sum.unsqueeze(1).clamp_min(
            self.eps
        )

        selected_counts = torch.zeros(
            num_misclassified, device=device, dtype=torch.long
        )
        selected_counts.index_add_(
            0,
            region_local_sample_indices,
            torch.ones_like(region_local_sample_indices),
        )
        return (
            spurious_features,
            hard_labels,
            soft_targets,
            valid_local_mask,
            selected_counts,
        )

    def _empty_output(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        misclassified_mask: torch.Tensor,
        rho_maps: Optional[torch.Tensor] = None,
        error_maps: Optional[torch.Tensor] = None,
        selected_region_counts: Optional[torch.Tensor] = None,
        region_features: Optional[torch.Tensor] = None,
        region_sample_indices: Optional[torch.Tensor] = None,
        region_weights: Optional[torch.Tensor] = None,
    ) -> CSIBOutput:
        B, C_l, H, W = features.shape
        device = features.device
        dtype = features.dtype
        layer_key = None
        for key, dim in enumerate(self.layer_dims, start=1):
            if dim == C_l:
                layer_key = str(key)
                break
        bank = self.cluster_banks[layer_key] if layer_key is not None else None

        return CSIBOutput(
            loss=logits.sum() * 0.0,
            logits=logits,
            misclassified_mask=misclassified_mask,
            valid_mask=torch.zeros(B, device=device, dtype=torch.bool),
            spurious_features=torch.zeros(B, C_l, device=device, dtype=dtype),
            hard_spurious_labels=torch.full(
                (B,), -1, device=device, dtype=torch.long
            ),
            soft_spurious_targets=torch.zeros(
                B, self.num_clusters, device=device, dtype=dtype
            ),
            rho_maps=(
                rho_maps
                if rho_maps is not None
                else torch.zeros(B, H, W, device=device, dtype=dtype)
            ),
            error_maps=(
                error_maps
                if error_maps is not None
                else torch.zeros(B, H, W, device=device, dtype=dtype)
            ),
            selected_region_counts=(
                selected_region_counts
                if selected_region_counts is not None
                else torch.zeros(B, device=device, dtype=torch.long)
            ),
            region_features=(
                region_features.detach()
                if region_features is not None
                else torch.zeros(0, C_l, device=device, dtype=dtype)
            ),
            region_cluster_labels=torch.empty(
                0, device=device, dtype=torch.long
            ),
            region_sample_indices=(
                region_sample_indices.detach()
                if region_sample_indices is not None
                else torch.empty(0, device=device, dtype=torch.long)
            ),
            region_weights=(
                region_weights.detach()
                if region_weights is not None
                else torch.empty(0, device=device, dtype=dtype)
            ),
            region_similarities=torch.empty(0, device=device, dtype=dtype),
            influence_scores=(
                bank.influence_scores.detach().to(device=device, dtype=dtype)
                if bank is not None
                else torch.zeros(self.num_clusters, device=device, dtype=dtype)
            ),
            gap_scores=(
                bank.gap_scores.detach().to(device=device, dtype=dtype)
                if bank is not None
                else torch.zeros(self.num_clusters, device=device, dtype=dtype)
            ),
            spurious_cluster_mask=(
                bank.spurious_cluster_mask.detach().to(device=device)
                if bank is not None
                else torch.zeros(
                    self.num_clusters, device=device, dtype=torch.bool
                )
            ),
        )

    def forward_from_model(
        self,
        student_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        layer: int = 4,
        grl_coefficient: float = 1.0,
        update_clusters: bool = True,
        epoch: int = 1,
    ) -> CSIBOutput:
        if labels.ndim != 1:
            raise ValueError("labels must have shape [B].")
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError("inputs and labels batch sizes differ.")
        if grl_coefficient < 0:
            raise ValueError("grl_coefficient must be non-negative.")
        if epoch <= 0:
            raise ValueError(
                "epoch must be a positive 1-based index. For a 0-based loop, "
                "pass epoch=epoch_index + 1."
            )

        logits, features = self._capture_layer_and_forward(
            student_model=student_model,
            inputs=inputs,
            layer=layer,
        )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} logits, got {logits.shape[1]}."
            )

        B, _, H, W = features.shape
        device = features.device
        dtype = features.dtype
        layer_key = str(layer)
        cluster_bank = self.cluster_banks[layer_key]

        predictions = logits.detach().argmax(dim=1)
        misclassified_mask = predictions.ne(labels)

        if epoch > self.warm_up and not bool(
            cluster_bank.kmeans_initialized.item()
        ):
            self._initialize_cluster_bank_from_memory(layer_key)

        if epoch < self.warm_up:
            return self._empty_output(
                logits, features, labels, misclassified_mask
            )

        # For every sample, use the strongest non-ground-truth class as the
        # competitor. On a misclassified sample it equals the predicted class.
        competitor_logits = logits.detach().clone()
        competitor_logits.scatter_(
            1, labels.view(-1, 1), torch.finfo(competitor_logits.dtype).min
        )
        competitor_labels = competitor_logits.argmax(dim=1)
        all_indices = torch.arange(B, device=device, dtype=torch.long)

        gt_directions, err_directions, _ = self._compute_class_directions(
            student_model=student_model,
            logits=logits,
            features=features,
            labels=labels,
            predictions=competitor_labels,
            mis_indices=all_indices,
            layer=layer,
        )
        full_error, full_rho = self._compute_error_and_rho_maps(
            local_features=features,
            gt_directions=gt_directions,
            err_directions=err_directions,
        )

        (
            region_features,
            region_weights,
            region_errors,
            region_local_sample_indices,
            region_full_sample_indices,
        ) = self._collect_candidate_regions(
            mis_features=features,
            rho_maps=full_rho,
            mis_indices=all_indices,
            error_maps=full_error,
        )

        counts = torch.zeros(B, device=device, dtype=torch.long)
        if region_full_sample_indices.numel() > 0:
            counts.index_add_(
                0,
                region_full_sample_indices,
                torch.ones_like(region_full_sample_indices),
            )

        if epoch == self.warm_up:
            self._append_warmup_regions(
                layer_key=layer_key,
                region_features=region_features,
                region_weights=region_weights,
                region_errors=region_errors,
                region_full_sample_indices=region_full_sample_indices,
                labels=labels,
                correct_mask=~misclassified_mask,
            )
            return self._empty_output(
                logits=logits,
                features=features,
                labels=labels,
                misclassified_mask=misclassified_mask,
                rho_maps=full_rho.detach().to(dtype),
                error_maps=full_error.detach().to(dtype),
                selected_region_counts=counts,
                region_features=region_features,
                region_sample_indices=region_full_sample_indices,
                region_weights=region_weights,
            )

        if region_features.shape[0] == 0:
            return self._empty_output(
                logits=logits,
                features=features,
                labels=labels,
                misclassified_mask=misclassified_mask,
                rho_maps=full_rho.detach().to(dtype),
                error_maps=full_error.detach().to(dtype),
                selected_region_counts=counts,
            )

        all_cluster_labels, all_similarities = (
            cluster_bank.assign_with_similarity(region_features)
        )

        ravl_spurious = cluster_bank.spurious_cluster_mask.to(device=device)
        keep_mask = (
            ravl_spurious.index_select(0, all_cluster_labels)
            & all_similarities.ge(self.prototype_similarity_threshold)
        )

        if update_clusters:
            cluster_bank.update(
                region_features=region_features,
                region_labels=all_cluster_labels,
                region_similarities=all_similarities,
                similarity_threshold=self.prototype_similarity_threshold,
            )

        if not keep_mask.any():
            return self._empty_output(
                logits=logits,
                features=features,
                labels=labels,
                misclassified_mask=misclassified_mask,
                rho_maps=full_rho.detach().to(dtype),
                error_maps=full_error.detach().to(dtype),
                selected_region_counts=torch.zeros_like(counts),
            )

        spur_region_features = region_features[keep_mask]
        spur_region_weights = region_weights[keep_mask]
        spur_region_local_indices = region_local_sample_indices[keep_mask]
        spur_region_full_indices = region_full_sample_indices[keep_mask]
        spur_cluster_labels = all_cluster_labels[keep_mask]
        spur_similarities = all_similarities[keep_mask]

        (
            local_spurious_features,
            local_hard_labels,
            local_soft_targets,
            valid_local_mask,
            local_selected_counts,
        ) = self._aggregate_sample_spurious_variable(
            region_features=spur_region_features,
            region_weights=spur_region_weights,
            region_local_sample_indices=spur_region_local_indices,
            region_cluster_labels=spur_cluster_labels,
            num_misclassified=B,
        )

        full_valid = valid_local_mask
        full_spur = local_spurious_features
        full_hard = local_hard_labels
        full_soft = local_soft_targets
        full_counts = local_selected_counts

        num_valid = int(full_valid.sum().item())
        active_clusters = int(
            (full_soft[full_valid].sum(dim=0) > 0).sum().item()
        )

        if (
            num_valid < self.min_valid_samples
            or active_clusters < self.min_active_clusters
        ):
            loss = logits.sum() * 0.0
        else:
            valid_spur = full_spur[full_valid]
            valid_labels = labels[full_valid]
            discriminator_logits = self.discriminators[layer_key](
                spurious_features=grad_reverse(
                    valid_spur, coefficient=grl_coefficient
                ),
                labels=valid_labels,
            )

            inactive = ~cluster_bank.spurious_cluster_mask.to(device=device)
            discriminator_logits = discriminator_logits.masked_fill(
                inactive.unsqueeze(0),
                torch.finfo(discriminator_logits.dtype).min,
            )

            if self.target_mode == "hard":
                loss = F.cross_entropy(
                    discriminator_logits, full_hard[full_valid]
                )
            else:
                soft_target = full_soft[full_valid].detach()
                log_probability = F.log_softmax(discriminator_logits, dim=1)
                loss = -torch.sum(
                    soft_target * log_probability, dim=1
                ).mean()

        return CSIBOutput(
            loss=loss,
            logits=logits,
            misclassified_mask=misclassified_mask,
            valid_mask=full_valid,
            spurious_features=full_spur,
            hard_spurious_labels=full_hard,
            soft_spurious_targets=full_soft,
            rho_maps=full_rho.detach().to(dtype),
            error_maps=full_error.detach().to(dtype),
            selected_region_counts=full_counts,
            region_features=spur_region_features.detach(),
            region_cluster_labels=spur_cluster_labels.detach(),
            region_sample_indices=spur_region_full_indices.detach(),
            region_weights=spur_region_weights.detach(),
            region_similarities=spur_similarities.detach(),
            influence_scores=cluster_bank.influence_scores.detach().to(
                device=device, dtype=dtype
            ),
            gap_scores=cluster_bank.gap_scores.detach().to(
                device=device, dtype=dtype
            ),
            spurious_cluster_mask=cluster_bank.spurious_cluster_mask.detach().to(
                device=device
            ),
        )


def ravl_guided_ibp(
    module: ErrorGuidedConditionalSpuriousIB,
    student_model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    layer: int = 3,
    grl_coefficient: float = 1.0,
    update_clusters: bool = True,
    epoch: int = 1,
) -> CSIBOutput:
    """
    Single public function for direct training-loop use.

    The stateful ``module`` must be constructed once and included in the
    optimizer because its conditional discriminator is trainable.
    """
    return module.forward_from_model(
        student_model=student_model,
        inputs=inputs,
        labels=labels,
        layer=layer,
        grl_coefficient=grl_coefficient,
        update_clusters=update_clusters,
        epoch=epoch,
    )

