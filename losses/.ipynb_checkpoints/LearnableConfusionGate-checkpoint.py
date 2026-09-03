
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrNone = Optional[torch.Tensor]


class _GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """前向恒等、反向乘以 -alpha 的梯度反转层。"""
    return _GradientReverseFunction.apply(x, alpha)


def _normalize_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对每个样本的 CAM 做 [0, 1] 归一化。"""
    flat = cam.flatten(start_dim=1)
    flat_min = flat.amin(dim=1, keepdim=True)
    flat_max = flat.amax(dim=1, keepdim=True)
    flat = (flat - flat_min) / (flat_max - flat_min + eps)
    return flat.view_as(cam)


@torch.no_grad()
def compute_class_cam(
    spatial_features: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_indices: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    使用线性分类器权重计算 CAM。

    Args:
        spatial_features: [B, C, H, W]
        classifier_weight: [num_classes, C]
        class_indices: [B]

    Returns:
        cam: [B, 1, H, W]
    """
    if spatial_features.ndim != 4:
        raise ValueError(
            "spatial_features must have shape [B, C, H, W], "
            f"but got {tuple(spatial_features.shape)}."
        )

    if classifier_weight.ndim != 2:
        raise ValueError(
            "classifier_weight must have shape [num_classes, C]."
        )

    if spatial_features.shape[1] != classifier_weight.shape[1]:
        raise ValueError(
            "Feature channels do not match classifier input dimension: "
            f"{spatial_features.shape[1]} vs {classifier_weight.shape[1]}."
        )

    sample_weight = classifier_weight[class_indices].to(
        device=spatial_features.device,
        dtype=spatial_features.dtype,
    )
    raw_cam = torch.einsum("bc,bchw->bhw", sample_weight, spatial_features)
    cam = F.relu(raw_cam)

    # ReLU 后完全为零时，回退到平移后的原始响应，避免失去定位信息。
    zero_mask = cam.flatten(1).amax(dim=1, keepdim=True) <= eps
    fallback = raw_cam - raw_cam.flatten(1).amin(dim=1, keepdim=True).view(-1, 1, 1)
    cam = torch.where(zero_mask.view(-1, 1, 1), fallback, cam)

    return _normalize_cam(cam, eps=eps).unsqueeze(1)


def get_classifier_weight(model: nn.Module) -> torch.Tensor:
    """
    获取常见分类模型最后一个线性分类器的权重 [num_classes, feature_dim]。
    支持 timm 的 get_classifier()、fc、classifier 和 head。
    """
    classifier = None

    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, str):
            classifier = getattr(model, classifier)

    if classifier is None:
        for name in ("fc", "classifier", "head"):
            if hasattr(model, name):
                classifier = getattr(model, name)
                break

    if isinstance(classifier, nn.Linear):
        return classifier.weight

    if classifier is not None:
        linear_layers = [
            module for module in classifier.modules()
            if isinstance(module, nn.Linear)
        ]
        if linear_layers:
            return linear_layers[-1].weight

    raise AttributeError(
        "Cannot find a supported linear classifier. "
        "Pass classifier_weight explicitly if the model uses a custom head."
    )


def _validate_binary_group_mapping(
    class_to_group: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    class_to_group = torch.as_tensor(class_to_group, dtype=torch.long)

    if class_to_group.ndim != 1 or class_to_group.numel() != num_classes:
        raise ValueError(
            "class_to_group must be a 1D tensor with one entry per class. "
            f"Expected {num_classes}, got {tuple(class_to_group.shape)}."
        )

    unique_groups = set(class_to_group.detach().cpu().tolist())
    if not unique_groups.issubset({0, 1}):
        raise ValueError(
            "This AP implementation uses a binary group mapping: "
            "0=head/non-tail and 1=tail."
        )

    return class_to_group


class LearnableConfusionGate(nn.Module):
    """
    可学习空间门控。

    以学生空间特征和两个 CAM 提示为输入，为每个位置输出可学习分数。
    forward 中可使用 straight-through top-k，使前向真正选择 top-ratio
    区域，同时保留可微梯度。
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive.")

        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.cue_proj = nn.Conv2d(3, hidden_dim, kernel_size=1)
        self.score_head = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        reversed_features: torch.Tensor,
        true_cam: torch.Tensor,
        confusing_cam: torch.Tensor,
        temperature: float = 0.2,
        topk_ratio: Optional[float] = 0.2,
        straight_through: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate: [B, 1, H, W], 每个样本空间权重和为 1。
            score_map: [B, 1, H, W]
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        cue = torch.cat(
            [
                true_cam,
                confusing_cam,
                confusing_cam - true_cam,
            ],
            dim=1,
        )

        score_map = self.score_head(
            self.feature_proj(reversed_features)
            + self.cue_proj(cue)
        )

        batch_size, _, height, width = score_map.shape
        flat_score = score_map.flatten(start_dim=1)
        soft_gate = F.softmax(flat_score / temperature, dim=1)

        if topk_ratio is None:
            gate = soft_gate
        else:
            if not 0 < topk_ratio <= 1:
                raise ValueError("topk_ratio must be in (0, 1].")

            num_positions = height * width
            k = max(1, int(ceil(topk_ratio * num_positions)))
            topk_indices = flat_score.topk(k=k, dim=1, largest=True).indices

            hard_gate = torch.zeros_like(soft_gate)
            hard_gate.scatter_(
                dim=1,
                index=topk_indices,
                value=1.0 / float(k),
            )

            if straight_through:
                # 前向使用 hard top-k，反向使用 soft gate 的梯度。
                gate = hard_gate - soft_gate.detach() + soft_gate
            else:
                gate = hard_gate

        return gate.view(batch_size, 1, height, width), score_map


class CrossGroupDiscriminator(nn.Module):
    """二分类跨组混淆方向判别器。"""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class APResult:
    loss: Optional[torch.Tensor]
    num_selected: int
    selected_ratio: float
    gate_entropy: float
    true_groups: Optional[torch.Tensor] = None
    predicted_groups: Optional[torch.Tensor] = None
    confusing_classes: Optional[torch.Tensor] = None
    confusion_targets: Optional[torch.Tensor] = None


class GatedAdversarialPurifier(nn.Module):
    """
    可学习门控 + 二分类判别器 + GRL 的跨组对抗净化模块。

    约定:
        class_to_group[c] = 0 表示 head/non-tail
        class_to_group[c] = 1 表示 tail
    """

    def __init__(
        self,
        feature_dim: int,
        gate_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        discriminator_dropout: float = 0.1,
        grl_alpha: float = 1.0,
        gate_temperature: float = 0.2,
        topk_ratio: Optional[float] = 0.2,
        straight_through_topk: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if grl_alpha < 0:
            raise ValueError("grl_alpha must be non-negative.")

        self.gate = LearnableConfusionGate(
            feature_dim=feature_dim,
            hidden_dim=gate_hidden_dim,
        )
        self.discriminator = CrossGroupDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=discriminator_hidden_dim,
            dropout=discriminator_dropout,
        )

        self.grl_alpha = float(grl_alpha)
        self.gate_temperature = float(gate_temperature)
        self.topk_ratio = topk_ratio
        self.straight_through_topk = bool(straight_through_topk)
        self.eps = float(eps)

    def forward(
        self,
        spatial_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_to_group: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> APResult:
        """
        Args:
            spatial_features:
                学生 forward_features 输出，[B, C, H, W]。
            logits:
                学生分类 logits，[B, num_classes]。
                仅用于无梯度地确定预测组和最混淆异组类别。
            labels:
                全局类别标签，[B]。
            class_to_group:
                每个类别对应的二元组标签，[num_classes]。
                0=head/non-tail，1=tail。
            classifier_weight:
                学生线性分类器权重，[num_classes, C]。

        Returns:
            APResult。没有跨组误分类样本时 loss=None。
        """
        if spatial_features.ndim != 4:
            raise ValueError(
                "spatial_features must have shape [B, C, H, W]."
            )
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B, num_classes].")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [B].")
        if spatial_features.shape[0] != logits.shape[0] or labels.shape[0] != logits.shape[0]:
            raise ValueError("Batch sizes of features, logits, and labels must match.")

        batch_size, feature_dim, _, _ = spatial_features.shape
        num_classes = logits.shape[1]

        if classifier_weight.shape != (num_classes, feature_dim):
            raise ValueError(
                "classifier_weight must have shape "
                f"[{num_classes}, {feature_dim}], got "
                f"{tuple(classifier_weight.shape)}."
            )

        class_to_group = _validate_binary_group_mapping(
            class_to_group=class_to_group,
            num_classes=num_classes,
        ).to(device=labels.device)

        with torch.no_grad():
            detached_logits = logits.detach()
            predicted_classes = detached_logits.argmax(dim=1)
            true_groups = class_to_group[labels]
            predicted_groups = class_to_group[predicted_classes]
            sample_mask = predicted_groups.ne(true_groups)

            num_selected = int(sample_mask.sum().item())
            selected_ratio = num_selected / max(batch_size, 1)

            if num_selected == 0:
                return APResult(
                    loss=None,
                    num_selected=0,
                    selected_ratio=0.0,
                    gate_entropy=0.0,
                )

            selected_logits = detached_logits[sample_mask]
            selected_true_groups = true_groups[sample_mask]
            selected_predicted_groups = predicted_groups[sample_mask]
            selected_labels = labels[sample_mask]

            # 仅在真实组的对立组中寻找最高分类别。
            opposite_groups = 1 - selected_true_groups
            allowed_mask = (
                class_to_group.unsqueeze(0)
                == opposite_groups.unsqueeze(1)
            )
            masked_logits = selected_logits.masked_fill(
                ~allowed_mask,
                torch.finfo(selected_logits.dtype).min,
            )
            confusing_classes = masked_logits.argmax(dim=1)

            # 0=head, 1=tail，因此预测组本身正好对应：
            # 1: head -> tail, 0: tail -> head。
            confusion_targets = selected_predicted_groups.long()

        selected_features = spatial_features[sample_mask]

        # CAM 仅作为无梯度的类别定位提示。
        true_cam = compute_class_cam(
            spatial_features=selected_features.detach(),
            classifier_weight=classifier_weight.detach(),
            class_indices=selected_labels,
            eps=self.eps,
        )
        confusing_cam = compute_class_cam(
            spatial_features=selected_features.detach(),
            classifier_weight=classifier_weight.detach(),
            class_indices=confusing_classes,
            eps=self.eps,
        )

        # GRL 只反转传回学生特征提取器的梯度；
        # gate 与 discriminator 参数仍按正常 CE 方向更新。
        reversed_features = grad_reverse(
            selected_features,
            alpha=self.grl_alpha,
        )

        gate, _ = self.gate(
            reversed_features=reversed_features,
            true_cam=true_cam,
            confusing_cam=confusing_cam,
            temperature=self.gate_temperature,
            topk_ratio=self.topk_ratio,
            straight_through=self.straight_through_topk,
        )

        confusion_features = (
            reversed_features * gate
        ).sum(dim=(2, 3))

        ap_logits = self.discriminator(confusion_features)
        loss = F.cross_entropy(
            ap_logits,
            confusion_targets,
        )

        with torch.no_grad():
            flat_gate = gate.detach().flatten(start_dim=1).clamp_min(self.eps)
            gate_entropy = float(
                (-(flat_gate * flat_gate.log()).sum(dim=1)).mean().item()
            )

        return APResult(
            loss=loss,
            num_selected=num_selected,
            selected_ratio=selected_ratio,
            gate_entropy=gate_entropy,
            true_groups=selected_true_groups.detach(),
            predicted_groups=selected_predicted_groups.detach(),
            confusing_classes=confusing_classes.detach(),
            confusion_targets=confusion_targets.detach(),
        )


def _project_ap_gradient(
    task_grads: List[TensorOrNone],
    ap_grads: List[TensorOrNone],
    eps: float = 1e-12,
) -> tuple[List[TensorOrNone], Dict[str, float]]:
    """
    当 AP 梯度与任务梯度冲突时，投影 AP 梯度。
    """
    device = None
    dtype = None

    for grad in task_grads + ap_grads:
        if grad is not None:
            device = grad.device
            dtype = grad.dtype
            break

    if device is None:
        return ap_grads, {
            "gradient_conflict": 0.0,
            "gradient_cosine": 0.0,
            "task_grad_norm": 0.0,
            "ap_grad_norm": 0.0,
        }

    dot_product = torch.zeros((), device=device, dtype=dtype)
    task_norm_sq = torch.zeros_like(dot_product)
    ap_norm_sq = torch.zeros_like(dot_product)
    has_shared_gradient = False

    for g_task, g_ap in zip(task_grads, ap_grads):
        if g_task is None or g_ap is None:
            continue

        has_shared_gradient = True
        dot_product = dot_product + torch.sum(g_task * g_ap)
        task_norm_sq = task_norm_sq + torch.sum(g_task * g_task)
        ap_norm_sq = ap_norm_sq + torch.sum(g_ap * g_ap)

    if not has_shared_gradient:
        return ap_grads, {
            "gradient_conflict": 0.0,
            "gradient_cosine": 0.0,
            "task_grad_norm": 0.0,
            "ap_grad_norm": 0.0,
        }

    cosine = dot_product / (
        torch.sqrt(task_norm_sq + eps)
        * torch.sqrt(ap_norm_sq + eps)
    )

    if dot_product.detach().item() >= 0:
        return ap_grads, {
            "gradient_conflict": 0.0,
            "gradient_cosine": cosine.detach().item(),
            "task_grad_norm": torch.sqrt(task_norm_sq).detach().item(),
            "ap_grad_norm": torch.sqrt(ap_norm_sq).detach().item(),
        }

    projection_coefficient = dot_product / (task_norm_sq + eps)
    projected_ap_grads: List[TensorOrNone] = []

    for g_task, g_ap in zip(task_grads, ap_grads):
        if g_ap is None:
            projected_ap_grads.append(None)
        elif g_task is None:
            projected_ap_grads.append(g_ap)
        else:
            projected_ap_grads.append(
                g_ap - projection_coefficient * g_task
            )

    return projected_ap_grads, {
        "gradient_conflict": 1.0,
        "gradient_cosine": cosine.detach().item(),
        "task_grad_norm": torch.sqrt(task_norm_sq).detach().item(),
        "ap_grad_norm": torch.sqrt(ap_norm_sq).detach().item(),
    }


def gradient_project(
    model: nn.Module,
    ap_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_task: torch.Tensor,
    loss_ap: Optional[torch.Tensor],
    lambda_ap: float = 1.0,
    max_grad_norm: Optional[float] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    完成一次带任务梯度保护的参数更新。

    注意:
        1. AP gate 与 discriminator 必须封装在 ap_model 中。
        2. GRL alpha 推荐设为 1.0，通过 lambda_ap 控制 AP 强度。
        3. 本函数内部已执行 zero_grad 和 optimizer.step，
           外部不要再调用 backward/step。
    """
    if not loss_task.requires_grad:
        raise ValueError("loss_task must require gradients.")
    if lambda_ap < 0:
        raise ValueError("lambda_ap must be non-negative.")

    model_params = [p for p in model.parameters() if p.requires_grad]
    ap_params = [p for p in ap_model.parameters() if p.requires_grad]

    model_param_ids = {id(p) for p in model_params}
    if any(id(p) in model_param_ids for p in ap_params):
        raise ValueError("model and ap_model contain overlapping parameters.")

    optimizer.zero_grad(set_to_none=True)

    use_ap = (
        loss_ap is not None
        and loss_ap.requires_grad
        and lambda_ap > 0
    )

    task_grads = torch.autograd.grad(
        outputs=loss_task,
        inputs=model_params,
        retain_graph=use_ap,
        create_graph=False,
        allow_unused=True,
    )

    if use_ap:
        all_ap_params = model_params + ap_params
        all_ap_grads = torch.autograd.grad(
            outputs=loss_ap,
            inputs=all_ap_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        num_model_params = len(model_params)
        ap_model_grads = list(all_ap_grads[:num_model_params])
        ap_module_grads = list(all_ap_grads[num_model_params:])

        protected_ap_grads, statistics = _project_ap_gradient(
            task_grads=list(task_grads),
            ap_grads=ap_model_grads,
            eps=eps,
        )
    else:
        protected_ap_grads = [None for _ in model_params]
        ap_module_grads = [None for _ in ap_params]
        statistics = {
            "gradient_conflict": 0.0,
            "gradient_cosine": 0.0,
            "task_grad_norm": 0.0,
            "ap_grad_norm": 0.0,
        }

    for param, g_task, g_ap in zip(
        model_params,
        task_grads,
        protected_ap_grads,
    ):
        total_grad = None

        if g_task is not None:
            total_grad = g_task.detach().clone()

        if g_ap is not None:
            weighted_ap_grad = lambda_ap * g_ap.detach()
            if total_grad is None:
                total_grad = weighted_ap_grad.clone()
            else:
                total_grad.add_(weighted_ap_grad)

        param.grad = total_grad

    # gate 与 discriminator 不做冲突投影，正常最小化 AP CE。
    for param, g_ap_module in zip(ap_params, ap_module_grads):
        if g_ap_module is None:
            param.grad = None
        else:
            param.grad = lambda_ap * g_ap_module.detach().clone()

    if max_grad_norm is not None:
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model_params + ap_params,
            max_norm=max_grad_norm,
        )
        statistics["total_grad_norm"] = float(total_grad_norm)

    optimizer.step()

    statistics["ap_enabled"] = float(use_ap)
    statistics["loss_task"] = float(loss_task.detach().item())
    statistics["loss_ap"] = (
        float(loss_ap.detach().item()) if use_ap else 0.0
    )

    return statistics


class _TinyStudent(nn.Module):
    """用于直接运行本文件的最小示例模型。"""

    def __init__(self, num_classes: int = 8, feature_dim: int = 32) -> None:
        super().__init__()
        self.num_features = feature_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_head(
        self,
        features: torch.Tensor,
        pre_logits: bool = False,
    ) -> torch.Tensor:
        pooled = self.global_pool(features).flatten(1)
        return pooled if pre_logits else self.fc(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(
            self.forward_features(x),
            pre_logits=False,
        )


def _demo() -> None:
    """
    直接执行:
        python gated_adversarial_purification.py

    用随机数据完成一次可学习 gate + AP + 梯度保护更新。
    """
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 8
    feature_dim = 32

    # 0=head/non-tail, 1=tail
    class_to_group = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1],
        device=device,
        dtype=torch.long,
    )

    model = _TinyStudent(
        num_classes=num_classes,
        feature_dim=feature_dim,
    ).to(device)

    ap_model = GatedAdversarialPurifier(
        feature_dim=feature_dim,
        gate_hidden_dim=32,
        discriminator_hidden_dim=64,
        grl_alpha=1.0,
        gate_temperature=0.2,
        topk_ratio=0.2,
        straight_through_topk=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(ap_model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    inputs = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.tensor(
        [0, 1, 2, 3, 5, 6, 7, 5],
        device=device,
        dtype=torch.long,
    )

    spatial_features = model.forward_features(inputs)
    outputs = model.forward_head(spatial_features, pre_logits=False)
    loss_task = F.cross_entropy(outputs, labels)

    # 仅用于 demo：人为调整一份“用于 AP 选择的 logits”，确保有跨组错误。
    # 实际训练直接传 outputs。
    ap_selection_logits = outputs.detach().clone()
    ap_selection_logits[0, 5] = ap_selection_logits[0].max() + 5.0  # head -> tail
    ap_selection_logits[4, 0] = ap_selection_logits[4].max() + 5.0  # tail -> head

    ap_result = ap_model(
        spatial_features=spatial_features,
        logits=ap_selection_logits,
        labels=labels,
        class_to_group=class_to_group,
        classifier_weight=get_classifier_weight(model),
    )

    stats = gradient_project(
        model=model,
        ap_model=ap_model,
        optimizer=optimizer,
        loss_task=loss_task,
        loss_ap=ap_result.loss,
        lambda_ap=0.2,
        max_grad_norm=5.0,
    )

    print("Demo finished successfully.")
    print(f"Selected AP samples: {ap_result.num_selected}")
    print(f"Selected ratio: {ap_result.selected_ratio:.4f}")
    print(f"Gate entropy: {ap_result.gate_entropy:.4f}")
    print(f"Task loss: {stats['loss_task']:.4f}")
    print(f"AP loss: {stats['loss_ap']:.4f}")
    print(f"Gradient conflict: {stats['gradient_conflict']:.0f}")
    print(f"Gradient cosine: {stats['gradient_cosine']:.4f}")


if __name__ == "__main__":
    _demo()
