
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unwrap_tensor(output, name: str) -> torch.Tensor:
    """Return the first Tensor when a model returns Tensor/tuple/list."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

    raise TypeError(
        f"{name} must return a Tensor or contain a Tensor."
    )


def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Support criterion with reduction='none'."""
    return loss if loss.ndim == 0 else loss.mean()


def get_timm_classifier(model: nn.Module) -> nn.Module:
    """
    Get the final classifier receiving the pooled embedding.

    For timm ResNet, this normally returns model.fc.
    """
    if hasattr(model, "get_classifier"):
        try:
            classifier = model.get_classifier()

            if (
                isinstance(classifier, nn.Module)
                and not isinstance(classifier, nn.Identity)
            ):
                return classifier
        except Exception:
            pass

    for name in ("fc", "classifier", "head"):
        if hasattr(model, name):
            classifier = getattr(model, name)

            if (
                isinstance(classifier, nn.Module)
                and not isinstance(classifier, nn.Identity)
            ):
                return classifier

    raise AttributeError(
        "Cannot find the final classifier. Check model.get_classifier(), "
        "model.fc, model.classifier, or model.head."
    )


def extract_timm_embedding_and_logits(
    model: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract spatial features, pooled embedding, and logits with one backbone
    forward pass.

    Returns:
        feature_map:
            Output of model.forward_features(inputs).

        embedding:
            Pooled feature before the final classifier, shape [B, D].

        logits:
            Classifier output, shape [B, C].
    """
    feature_map = _unwrap_tensor(
        model.forward_features(inputs),
        "model.forward_features(inputs)",
    )

    embedding = _unwrap_tensor(
        model.forward_head(
            feature_map,
            pre_logits=True,
        ),
        "model.forward_head(feature_map, pre_logits=True)",
    )

    if embedding.ndim != 2:
        raise ValueError(
            "The pooled embedding must have shape [B, D], "
            f"but got {tuple(embedding.shape)}."
        )

    classifier = get_timm_classifier(model)

    logits = _unwrap_tensor(
        classifier(embedding),
        "classifier(embedding)",
    )

    if logits.ndim != 2:
        raise ValueError(
            "The classifier output must have shape [B, C], "
            f"but got {tuple(logits.shape)}."
        )

    return feature_map, embedding, logits


class LongTailedConfusionAwareJSDInfoMax(nn.Module):
    """
    Long-Tailed Confusion-Aware JSD InfoMax.

    Positive pair:
        (z_i, y_i)

    Confusion-aware negative pair:
        (z_i, y_i^-)

    where:
        y_i^- ~ q(Y | z_i, Y != y_i)

    The negative label is sampled from the current classifier probabilities
    after excluding the ground-truth class. Therefore, labels that the model
    currently confuses with the ground truth are sampled more frequently.

    Important:
        1. This module does NOT use GRL.
        2. The encoder and the JSD critic minimize the same loss.
        3. Because negatives are confusion-conditioned rather than sampled
           from p(Y), this is a confusion-aware JSD surrogate, not an exact
           mutual-information estimator.
        4. Use it together with ordinary classification CE:

               total_loss = ce_loss + lambda_jsd * jsd_loss

    Args:
        feature_dim:
            Dimension D of the pooled embedding.

        num_classes:
            Number of classes.

        cls_num_list:
            Training sample count of every class. When provided, the JSD loss
            uses tail-aware anchor weights:

                w_c = (n_max / n_c) ** class_weight_power

            and normalizes them to mean 1.

            Set cls_num_list=None or class_weight_power=0.0 to disable
            long-tail weighting.

        projection_dim:
            Dimension of projected image and label features.

        hidden_dim:
            Hidden dimension of the pair discriminator.

        negative_temperature:
            Sampling temperature for q(Y | z_i, Y != y_i).

            temperature < 1:
                More concentrated on the most confusing class.

            temperature > 1:
                More diverse negative classes.

        negative_mode:
            "sample":
                Sample from q(Y | z_i, Y != y_i). Recommended.

            "top1":
                Always use the highest-logit wrong class.

        class_weight_power:
            Strength of long-tail anchor weighting.
            Recommended initial value: 0.5.

        dropout:
            Dropout in the pair discriminator.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        cls_num_list: Optional[
            Sequence[int] | torch.Tensor
        ] = None,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        negative_temperature: float = 1.0,
        negative_mode: str = "sample",
        class_weight_power: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(
                "feature_dim must be positive."
            )

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if projection_dim <= 0:
            raise ValueError(
                "projection_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if negative_temperature <= 0:
            raise ValueError(
                "negative_temperature must be positive."
            )

        if negative_mode not in {"sample", "top1"}:
            raise ValueError(
                'negative_mode must be "sample" or "top1".'
            )

        if class_weight_power < 0:
            raise ValueError(
                "class_weight_power must be non-negative."
            )

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.negative_temperature = float(
            negative_temperature
        )
        self.negative_mode = negative_mode

        class_weights = self._build_class_weights(
            cls_num_list=cls_num_list,
            num_classes=num_classes,
            class_weight_power=class_weight_power,
        )

        self.register_buffer(
            "class_weights",
            class_weights,
        )

        self.feature_projector = nn.Sequential(
            nn.Linear(
                feature_dim,
                projection_dim,
                bias=False,
            ),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(
                projection_dim,
                projection_dim,
                bias=False,
            ),
        )

        self.label_embedding = nn.Embedding(
            num_classes,
            projection_dim,
        )

        # Input:
        #   projected image embedding z,
        #   label embedding e_y,
        #   element-wise relation z * e_y.
        self.pair_discriminator = nn.Sequential(
            nn.Linear(
                projection_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.reset_parameters()

    @staticmethod
    def _build_class_weights(
        cls_num_list: Optional[
            Sequence[int] | torch.Tensor
        ],
        num_classes: int,
        class_weight_power: float,
    ) -> torch.Tensor:
        if (
            cls_num_list is None
            or class_weight_power == 0.0
        ):
            return torch.ones(
                num_classes,
                dtype=torch.float32,
            )

        class_counts = torch.as_tensor(
            cls_num_list,
            dtype=torch.float32,
        )

        if class_counts.ndim != 1:
            raise ValueError(
                "cls_num_list must be one-dimensional."
            )

        if class_counts.numel() != num_classes:
            raise ValueError(
                "The length of cls_num_list must equal "
                "num_classes."
            )

        if torch.any(class_counts <= 0):
            raise ValueError(
                "Every class count must be positive."
            )

        class_weights = (
            class_counts.max() / class_counts
        ).pow(class_weight_power)

        class_weights = (
            class_weights
            / class_weights.mean().clamp_min(1e-12)
        )

        return class_weights

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.label_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    @torch.no_grad()
    def select_confusion_negative_labels(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select confusion-aware wrong labels.

        sample mode:
            y_i^- ~ q(Y | z_i, Y != y_i)

        top1 mode:
            y_i^- = argmax_{c != y_i} q(c | z_i)

        Args:
            logits:
                Current classifier logits, shape [B, C].

            labels:
                Ground-truth labels, shape [B].

        Returns:
            negative_labels:
                Wrong labels, shape [B].
        """
        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [B, C]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [B]."
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "The batch sizes of logits and labels differ."
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "The class dimension of logits does not match "
                "num_classes."
            )

        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                "labels contain invalid class indices."
            )

        # Detach because discrete negative-label selection should not
        # propagate gradients through logits.
        confusion_logits = (
            logits.detach()
            / self.negative_temperature
        ).clone()

        # Exclude the ground-truth class.
        confusion_logits.scatter_(
            dim=1,
            index=labels.unsqueeze(1),
            value=-torch.inf,
        )

        if self.negative_mode == "top1":
            negative_labels = (
                confusion_logits.argmax(dim=1)
            )

        else:
            confusion_probabilities = F.softmax(
                confusion_logits,
                dim=1,
            )

            negative_labels = torch.multinomial(
                confusion_probabilities,
                num_samples=1,
                replacement=True,
            ).squeeze(1)

        # Defensive check.
        if torch.any(negative_labels == labels):
            raise RuntimeError(
                "A sampled negative label equals its ground-truth label."
            )

        return negative_labels

    def _compute_pair_scores(
        self,
        projected_embedding: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score whether each (z, y) pair comes from the positive joint pairs.
        """
        label_features = self.label_embedding(
            pair_labels
        )

        label_features = F.normalize(
            label_features,
            p=2,
            dim=1,
            eps=1e-8,
        )

        pair_features = torch.cat(
            [
                projected_embedding,
                label_features,
                projected_embedding
                * label_features,
            ],
            dim=1,
        )

        pair_scores = self.pair_discriminator(
            pair_features
        ).squeeze(1)

        return pair_scores

    def forward(
        self,
        embedding: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Compute the long-tailed confusion-aware JSD surrogate.

        Args:
            embedding:
                Pooled embedding before the classifier, shape [B, D].

            logits:
                Current classifier logits, shape [B, C].

            labels:
                Ground-truth labels, shape [B].

        Returns:
            loss:
                Scalar JSD surrogate loss.

            diagnostics:
                Dictionary of detached monitoring values.
        """
        if embedding.ndim != 2:
            raise ValueError(
                "embedding must have shape [B, D]."
            )

        if embedding.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"but got {embedding.shape[1]}."
            )

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [B, C]."
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected num_classes={self.num_classes}, "
                f"but got {logits.shape[1]}."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [B]."
            )

        if (
            embedding.shape[0] != logits.shape[0]
            or embedding.shape[0] != labels.shape[0]
        ):
            raise ValueError(
                "The batch sizes of embedding, logits, and labels differ."
            )

        negative_labels = (
            self.select_confusion_negative_labels(
                logits=logits,
                labels=labels,
            )
        )

        projected_embedding = (
            self.feature_projector(
                embedding
            )
        )

        projected_embedding = F.normalize(
            projected_embedding,
            p=2,
            dim=1,
            eps=1e-8,
        )

        # Positive pairs: (z_i, y_i)
        positive_scores = (
            self._compute_pair_scores(
                projected_embedding=
                    projected_embedding,
                pair_labels=labels,
            )
        )

        # Negative pairs: (z_i, y_i^-)
        negative_scores = (
            self._compute_pair_scores(
                projected_embedding=
                    projected_embedding,
                pair_labels=negative_labels,
            )
        )

        # Stable BCE in logit form:
        #   positive: -log sigmoid(s+)
        #   negative: -log sigmoid(-s-)
        positive_loss_per_sample = F.softplus(
            -positive_scores
        )

        negative_loss_per_sample = F.softplus(
            negative_scores
        )

        # Tail-aware anchor weighting.
        sample_weights = self.class_weights[
            labels
        ]

        # Normalize within the current batch to keep the loss scale stable.
        sample_weights = (
            sample_weights
            / sample_weights.mean().clamp_min(
                1e-12
            )
        )

        positive_loss = (
            sample_weights
            * positive_loss_per_sample
        ).sum() / sample_weights.sum().clamp_min(
            1e-12
        )

        negative_loss = (
            sample_weights
            * negative_loss_per_sample
        ).sum() / sample_weights.sum().clamp_min(
            1e-12
        )

        # Balanced positive/negative pair objective.
        loss = 0.5 * (
            positive_loss + negative_loss
        )

        with torch.no_grad():
            positive_accuracy = (
                positive_scores > 0
            ).float().mean()

            negative_accuracy = (
                negative_scores < 0
            ).float().mean()

            pair_accuracy = 0.5 * (
                positive_accuracy
                + negative_accuracy
            )

            # Only a monitoring surrogate. Since confusion-aware negatives
            # are not sampled from p(Y), do not report this as an exact MI.
            jsd_surrogate = (
                loss.detach().new_tensor(
                    math.log(2.0)
                )
                - loss.detach()
            )

            negative_confidence = F.softmax(
                logits.detach(),
                dim=1,
            ).gather(
                dim=1,
                index=negative_labels.unsqueeze(1),
            ).mean()

            diagnostics = {
                "negative_labels":
                    negative_labels.detach(),
                "positive_loss":
                    positive_loss.detach(),
                "negative_loss":
                    negative_loss.detach(),
                "positive_accuracy":
                    positive_accuracy,
                "negative_accuracy":
                    negative_accuracy,
                "pair_accuracy":
                    pair_accuracy,
                "jsd_surrogate":
                    jsd_surrogate,
                "mean_negative_confidence":
                    negative_confidence,
            }

        return loss, diagnostics


def compute_lt_confusion_jsd_training_loss(
    model: nn.Module,
    jsd_informax: LongTailedConfusionAwareJSDInfoMax,
    criterion: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    lambda_jsd: float = 0.5,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
]:
    """
    One-call training helper for timm models.

    Returns:
        total_loss
        outputs dictionary
    """
    if lambda_jsd < 0:
        raise ValueError(
            "lambda_jsd must be non-negative."
        )

    feature_map, embedding, logits = (
        extract_timm_embedding_and_logits(
            model=model,
            inputs=inputs,
        )
    )

    ce_loss = _reduce_loss(criterion(logits, labels))
    jsd_loss, jsd_info = jsd_informax(
        embedding=embedding,
        logits=logits,
        labels=labels,
    )
    total_loss = (ce_loss + lambda_jsd * jsd_loss)
    outputs = {
        "feature_map": feature_map,
        "embedding": embedding,
        "logits": logits,
        "ce_loss": ce_loss.detach(),
        "jsd_loss": jsd_loss.detach(),
        **jsd_info,
    }

    return total_loss, outputs