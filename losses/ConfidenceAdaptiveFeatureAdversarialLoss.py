
# from __future__ import annotations

# from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def confidence_adversarial_loss(
#     model, criterion,  inputs,
#     labels, perturb_power=1.0,
#     include_clean_loss=False):
#     if perturb_power <= 0:
#         raise ValueError(
#             "perturb_power must be positive."
#         )
#     if labels.ndim != 1:
#         raise ValueError(
#             "labels must have shape [B]."
#         )
#     student_features = model.forward_features(inputs)
#     clean_logits = model.forward_head(student_features, pre_logits=False)
#     if isinstance(clean_logits, (tuple, list)):
#         clean_logits = clean_logits[0]
#     if clean_logits.ndim != 2:
#         raise ValueError(
#             "model.forward_head(...) must return logits "
#             "with shape [B, num_classes]."
#         )
#     if clean_logits.shape[0] != labels.shape[0]:
#         raise ValueError(
#             "The batch sizes of logits and labels differ."
#         )
#     with torch.no_grad():
#         probability = F.softmax(
#             clean_logits.detach(),
#             dim=1,
#         )
#         true_class_probability = probability.gather(
#             dim=1,
#             index=labels.unsqueeze(1),
#         ).squeeze(1)
#         perturbation_scale = (
#             1.0 - true_class_probability
#         ).clamp(
#             min=0.0,
#             max=1.0,
#         ).pow(
#             perturb_power
#         )
#     attack_loss = criterion(clean_logits, labels)
#     feature_gradient = torch.autograd.grad(
#         outputs=attack_loss,
#         inputs=student_features,
#         retain_graph=True,
#         create_graph=False,
#         only_inputs=True,
#     )[0]

#     flattened_gradient = feature_gradient.flatten(start_dim=1)
#     gradient_norm = flattened_gradient.norm(
#         p=2,
#         dim=1,
#         keepdim=True,
#     ).clamp_min(1e-12)

#     gradient_direction = (
#         flattened_gradient / gradient_norm
#     ).view_as(feature_gradient)
#     scale_shape = [perturbation_scale.shape[0]] + [1] * (student_features.ndim - 1)
#     perturbation_scale = perturbation_scale.view(
#         *scale_shape
#     )
#     perturbation = (perturbation_scale * gradient_direction).detach()
#     perturbed_features = (student_features + perturbation)
#     return perturbed_features

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _unwrap_tensor(output, name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"{name} must return a Tensor or contain a Tensor.")


def _get_classifier(model: nn.Module) -> nn.Module:
    """Get the final classifier that receives the pooled embedding."""
    if hasattr(model, "get_classifier"):
        try:
            classifier = model.get_classifier()
            if isinstance(classifier, nn.Module) and not isinstance(classifier, nn.Identity):
                return classifier
        except Exception:
            pass

    for name in ("fc", "head", "classifier"):
        if hasattr(model, name):
            classifier = getattr(model, name)
            if isinstance(classifier, nn.Module) and not isinstance(classifier, nn.Identity):
                return classifier

    raise AttributeError(
        "Cannot find the final classifier. Check model.fc, model.head, "
        "model.classifier, or model.get_classifier()."
    )


def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss if loss.ndim == 0 else loss.mean()


def gaussian_embedding_perturbation_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    noise_strength: float = 0.1,
    include_clean_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to the pooled embedding immediately before the
    final classifier.

    Flow:
        feature_map = model.forward_features(inputs)
        embedding = model.forward_head(feature_map, pre_logits=True)
        noisy_embedding = embedding + noise_strength * std(embedding_i) * N(0, I)
        noisy_logits = classifier(noisy_embedding)

    Args:
        model: Must provide forward_features(), forward_head(), and a final
            classifier exposed by get_classifier(), fc, head, or classifier.
        inputs: Input batch.
        labels: Ground-truth labels, shape [B].
        noise_strength: Relative Gaussian noise standard deviation. A value
            of 0.1 means approximately 10% of each sample embedding's std.
        include_clean_loss: If True, return clean CE + noisy CE; otherwise
            return noisy CE only.

    Returns:
        loss: Scalar loss.
        perturbed_embedding: Noisy pooled embedding, shape [B, D].
    """
    if noise_strength < 0:
        raise ValueError("noise_strength must be non-negative.")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B].")

    feature_map = _unwrap_tensor(
        model.forward_features(inputs),
        "model.forward_features(inputs)",
    )

    embedding = _unwrap_tensor(
        model.forward_head(feature_map, pre_logits=True),
        "model.forward_head(feature_map, pre_logits=True)",
    )

    if embedding.ndim != 2:
        raise ValueError(
            "The pre-logits embedding must have shape [B, D], "
            f"but got {tuple(embedding.shape)}."
        )
    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The batch sizes of embedding and labels differ.")

    with torch.no_grad():
        embedding_std = embedding.detach().std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)

        perturbation = (
            noise_strength
            * torch.randn_like(embedding)
        ).detach()

    perturbed_embedding = embedding + perturbation
    return embedding, perturbed_embedding
