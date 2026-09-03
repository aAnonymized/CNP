# import math
# from typing import Optional, Sequence, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def _get_classifier_weight(model: nn.Module) -> torch.Tensor:
#     """
#     获取分类器权重。

#     返回
#     ----
#     weight:
#         [num_classes, feature_dim]

#     支持：
#         model.get_classifier()
#         model.fc
#         model.classifier
#         model.head
#         Sequential 中的最后一个 Linear
#         1x1 Conv2d 分类器
#     """
#     classifier = None

#     if hasattr(model, "get_classifier"):
#         classifier = model.get_classifier()

#         if isinstance(classifier, str):
#             classifier = getattr(model, classifier)

#     if classifier is None:
#         for name in ("fc", "classifier", "head"):
#             if hasattr(model, name):
#                 classifier = getattr(model, name)
#                 break

#     if classifier is None:
#         raise AttributeError(
#             "Cannot find classifier in teacher_model. "
#             "Expected get_classifier(), fc, classifier, or head."
#         )

#     if isinstance(classifier, nn.Linear):
#         return classifier.weight

#     if isinstance(classifier, nn.Conv2d):
#         if classifier.kernel_size != (1, 1):
#             raise ValueError(
#                 "Only a 1x1 Conv2d classifier is supported."
#             )

#         return classifier.weight[:, :, 0, 0]

#     linear_layers = [
#         module
#         for module in classifier.modules()
#         if isinstance(module, nn.Linear)
#     ]

#     if linear_layers:
#         return linear_layers[-1].weight

#     conv_layers = [
#         module
#         for module in classifier.modules()
#         if isinstance(module, nn.Conv2d)
#         and module.kernel_size == (1, 1)
#     ]

#     if conv_layers:
#         return conv_layers[-1].weight[:, :, 0, 0]

#     raise TypeError(
#         "No supported classifier layer was found in teacher_model."
#     )


# def _map_global_to_teacher_labels(
#     global_labels: torch.Tensor,
#     teacher_classes: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     将全局类别标签映射为教师模型内部的局部标签。

#     例如：
#         teacher_classes = [5, 6, 7]

#         global 5 -> teacher local 0
#         global 6 -> teacher local 1
#         global 7 -> teacher local 2

#     注意：
#         teacher_classes 的排列顺序必须与教师模型训练时的
#         局部标签顺序一致。
#     """
#     teacher_classes = teacher_classes.to(
#         device=global_labels.device,
#         dtype=torch.long,
#     )

#     matches = (
#         global_labels.unsqueeze(1)
#         == teacher_classes.unsqueeze(0)
#     )  # [B_tail, num_teacher_classes]

#     valid_mask = matches.any(dim=1)

#     if not valid_mask.all():
#         invalid_labels = (
#             global_labels[~valid_mask]
#             .detach()
#             .cpu()
#             .tolist()
#         )

#         raise ValueError(
#             "The following labels are not contained in "
#             f"teacher_classes: {invalid_labels}"
#         )

#     teacher_local_labels = matches.long().argmax(dim=1)

#     return teacher_local_labels


# def _resize_teacher_features(
#     teacher_features: torch.Tensor,
#     target_spatial_shape: Sequence[int],
# ) -> torch.Tensor:
#     """
#     将教师特征的空间尺寸调整为学生特征空间尺寸。

#     支持：
#         2D 特征：[B, C, H, W]
#         3D 特征：[B, C, T, H, W]
#     """
#     current_spatial_shape = teacher_features.shape[2:]

#     if tuple(current_spatial_shape) == tuple(target_spatial_shape):
#         return teacher_features

#     if teacher_features.ndim == 4:
#         return F.interpolate(
#             teacher_features,
#             size=tuple(target_spatial_shape),
#             mode="bilinear",
#             align_corners=False,
#         )

#     if teacher_features.ndim == 5:
#         return F.interpolate(
#             teacher_features,
#             size=tuple(target_spatial_shape),
#             mode="trilinear",
#             align_corners=False,
#         )

#     raise ValueError(
#         "Only 4D or 5D spatial feature maps are supported, "
#         f"but received shape {tuple(teacher_features.shape)}."
#     )


# @torch.no_grad()
# def _compute_teacher_cam(
#     teacher_features: torch.Tensor,
#     classifier_weight: torch.Tensor,
#     teacher_labels: torch.Tensor,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     计算教师模型 CAM。

#     参数
#     ----
#     teacher_features:
#         [B_tail, C, H, W]
#         或 [B_tail, C, T, H, W]

#     classifier_weight:
#         [num_teacher_classes, C]

#     teacher_labels:
#         [B_tail]

#     返回
#     ----
#     teacher_cam:
#         [B_tail, S]

#         S = H * W
#         或 S = T * H * W
#     """
#     if teacher_features.ndim not in (4, 5):
#         raise ValueError(
#             "teacher_features must have shape [B,C,H,W] "
#             "or [B,C,T,H,W], but received "
#             f"{tuple(teacher_features.shape)}."
#         )

#     batch_size, channels = teacher_features.shape[:2]

#     if classifier_weight.ndim != 2:
#         raise ValueError(
#             "classifier_weight must have shape "
#             "[num_classes, channels]."
#         )

#     if classifier_weight.shape[1] != channels:
#         raise ValueError(
#             "Teacher classifier input dimension does not match "
#             "teacher feature channels: "
#             f"{classifier_weight.shape[1]} vs {channels}."
#         )

#     if teacher_labels.shape[0] != batch_size:
#         raise ValueError(
#             "teacher_labels batch size does not match "
#             "teacher_features batch size."
#         )

#     if teacher_labels.min() < 0:
#         raise ValueError(
#             "teacher_labels must be non-negative."
#         )

#     if teacher_labels.max() >= classifier_weight.shape[0]:
#         raise ValueError(
#             "teacher_labels exceed teacher classifier output size."
#         )

#     # 每个样本对应类别的分类器权重：[B_tail, C]
#     sample_classifier_weight = classifier_weight[
#         teacher_labels
#     ].to(
#         device=teacher_features.device,
#         dtype=teacher_features.dtype,
#     )

#     # [B_tail, C, S]
#     flat_teacher_features = teacher_features.flatten(
#         start_dim=2
#     )

#     # [B_tail, S]
#     raw_cam = torch.einsum(
#         "bc,bcs->bs",
#         sample_classifier_weight,
#         flat_teacher_features,
#     )

#     teacher_cam = F.relu(raw_cam)

#     # 某个样本的 CAM 全为 0 时，使用平移后的原始响应，
#     # 避免 TopK 在全零 CAM 上完全随机选择。
#     zero_cam_mask = (
#         teacher_cam.amax(dim=1, keepdim=True)
#         <= eps
#     )

#     fallback_cam = (
#         raw_cam
#         - raw_cam.amin(dim=1, keepdim=True)
#     )

#     teacher_cam = torch.where(
#         zero_cam_mask,
#         fallback_cam,
#         teacher_cam,
#     )

#     # 每个样本单独归一化到 [0, 1]
#     cam_min = teacher_cam.amin(
#         dim=1,
#         keepdim=True,
#     )

#     cam_max = teacher_cam.amax(
#         dim=1,
#         keepdim=True,
#     )

#     teacher_cam = (
#         teacher_cam - cam_min
#     ) / (
#         cam_max - cam_min + eps
#     )

#     return teacher_cam


# def _gather_teacher_cam_regions(
#     student_tail_features: torch.Tensor,
#     teacher_features: torch.Tensor,
#     teacher_cam: torch.Tensor,
#     top_ratio: float,
# ):
#     """
#     根据教师 CAM 选取前 top_ratio 的位置，并在相同位置提取
#     学生特征和教师特征。

#     返回
#     ----
#     student_selected:
#         [B_tail, K, C]

#     teacher_selected:
#         [B_tail, K, C]

#     top_indices:
#         [B_tail, K]
#     """
#     if not 0.0 < top_ratio <= 1.0:
#         raise ValueError(
#             f"top_ratio must be in (0, 1], but got {top_ratio}."
#         )

#     if student_tail_features.shape != teacher_features.shape:
#         raise ValueError(
#             "After spatial resizing, student_tail_features and "
#             "teacher_features must have the same shape, but got "
#             f"{tuple(student_tail_features.shape)} and "
#             f"{tuple(teacher_features.shape)}."
#         )

#     batch_size, channels = student_tail_features.shape[:2]

#     # [B, C, ...] -> [B, C, S] -> [B, S, C]
#     flat_student_features = (
#         student_tail_features
#         .flatten(start_dim=2)
#         .transpose(1, 2)
#     )

#     flat_teacher_features = (
#         teacher_features
#         .flatten(start_dim=2)
#         .transpose(1, 2)
#     )

#     num_positions = flat_student_features.shape[1]

#     if teacher_cam.shape != (
#         batch_size,
#         num_positions,
#     ):
#         raise ValueError(
#             "teacher_cam shape does not match flattened "
#             "feature positions: "
#             f"{tuple(teacher_cam.shape)} vs "
#             f"{(batch_size, num_positions)}."
#         )

#     top_k = max(
#         1,
#         math.ceil(top_ratio * num_positions),
#     )

#     # 教师 CAM 只用于位置选择，因此索引不参与反向传播
#     top_indices = torch.topk(
#         teacher_cam.detach(),
#         k=top_k,
#         dim=1,
#         largest=True,
#         sorted=False,
#     ).indices  # [B_tail, K]

#     gather_indices = top_indices.unsqueeze(-1).expand(
#         batch_size,
#         top_k,
#         channels,
#     )

#     # 使用相同坐标提取学生和教师特征
#     student_selected = torch.gather(
#         flat_student_features,
#         dim=1,
#         index=gather_indices,
#     )  # [B_tail, K, C]

#     teacher_selected = torch.gather(
#         flat_teacher_features,
#         dim=1,
#         index=gather_indices,
#     )  # [B_tail, K, C]

#     return (
#         student_selected,
#         teacher_selected,
#         top_indices,
#     )


# def camGuided_IBP_Loss(
#     teacher_model: nn.Module,
#     student_features: torch.Tensor,
#     teacher_features: Optional[torch.Tensor],
#     labels: torch.Tensor,
#     tail_mask: torch.Tensor,
#     teacher_classes: Union[torch.Tensor, Sequence[int]],
#     top_ratio: float = 0.2,
#     noise_std: float = 0.1,
#     relative_noise: bool = True,
#     normalize_features: bool = False,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     只使用教师 CAM 的 IBP 前景特征对齐损失。

#     完整流程
#     --------
#     1. 教师模型计算尾类 CAM；
#     2. 选择教师 CAM 最大的前 top_ratio 位置；
#     3. 在相同坐标提取教师特征和学生特征；
#     4. 对学生前景特征加入高斯扰动；
#     5. 对学生和教师的逐位置特征进行 MSE 对齐。

#     参数
#     ----
#     teacher_model: 仅使用 teacher_classes 训练的教师模型。
#     student_features: 学生模型完整 batch 的空间特征：
#         [B, C, H, W]
#         或 [B, C, T, H, W]
#     teacher_features: 教师模型对尾部样本提取的空间特征：
#         [B_tail, C, H, W]
#         或 [B_tail, C, T, H, W]
#     labels: 全局类别标签，例如 [0, ..., 7]。
#     tail_mask: teacher_classes 对应样本的布尔掩码。
#     teacher_classes: 教师输出局部标签与全局标签的对应关系。
#         例如：
#             teacher_classes = [5, 6, 7]
#         必须表示：
#             teacher output 0 对应 global class 5
#             teacher output 1 对应 global class 6
#             teacher output 2 对应 global class 7

#     top_ratio: 教师 CAM 前景比例，默认 0.2，即前 20%。

#     noise_std: 学生前景特征的高斯扰动强度, 设置为 0 表示不加入扰动。
#     relative_noise:
#         True： 噪声大小根据当前学生前景特征标准差调整。
#         False： 直接加入 noise_std * N(0, I)。
#     normalize_features:
#         False：直接对齐原始特征值，更符合平方距离推导。
#         True：对每个空间位置的特征向量进行 L2 归一化后对齐。

#     返回
#     ----
#     loss:
#         标量 IBP 损失。
#     """
#     if student_features.ndim not in (4, 5):
#         raise ValueError(
#             "student_features must have shape [B,C,H,W] "
#             "or [B,C,T,H,W], but received "
#             f"{tuple(student_features.shape)}."
#         )

#     tail_mask = tail_mask.to(
#         device=labels.device,
#         dtype=torch.bool,
#     )

#     if not tail_mask.any():
#         return student_features.sum() * 0.0

#     if teacher_features is None:
#         raise ValueError(
#             "teacher_features cannot be None when "
#             "tail_mask contains tail-class samples."
#         )

#     teacher_classes = torch.as_tensor(
#         teacher_classes,
#         device=labels.device,
#         dtype=torch.long,
#     )

#     student_tail_features = student_features[
#         tail_mask
#     ]

#     tail_global_labels = labels[
#         tail_mask
#     ].long()

#     teacher_features = teacher_features.detach()

#     if (
#         student_tail_features.shape[0]
#         != teacher_features.shape[0]
#     ):
#         raise ValueError(
#             "The number of student tail samples must equal "
#             "the number of teacher samples: "
#             f"{student_tail_features.shape[0]} vs "
#             f"{teacher_features.shape[0]}."
#         )

#     if (
#         student_tail_features.ndim
#         != teacher_features.ndim
#     ):
#         raise ValueError(
#             "Student and teacher feature dimensions differ: "
#             f"{student_tail_features.ndim}D vs "
#             f"{teacher_features.ndim}D."
#         )

#     if (
#         student_tail_features.shape[1]
#         != teacher_features.shape[1]
#     ):
#         raise ValueError(
#             "Direct feature alignment requires identical channel "
#             "dimensions, but student and teacher channels are "
#             f"{student_tail_features.shape[1]} and "
#             f"{teacher_features.shape[1]}."
#         )

#     teacher_features = _resize_teacher_features(
#         teacher_features=teacher_features,
#         target_spatial_shape=student_tail_features.shape[2:],
#     )

#     teacher_classifier_weight = _get_classifier_weight(
#         teacher_model
#     ).detach()

#     num_teacher_outputs = (
#         teacher_classifier_weight.shape[0]
#     )

#     if num_teacher_outputs == teacher_classes.numel():
#         teacher_cam_labels = _map_global_to_teacher_labels(
#             global_labels=tail_global_labels,
#             teacher_classes=teacher_classes,
#         )

#     elif (
#         tail_global_labels.max().item()
#         < num_teacher_outputs
#     ):
#         teacher_cam_labels = tail_global_labels

#     else:
#         raise ValueError(
#             "Cannot infer teacher label mapping. "
#             f"Teacher classifier has {num_teacher_outputs} outputs, "
#             f"while teacher_classes contains "
#             f"{teacher_classes.numel()} classes."
#         )

#     teacher_cam = _compute_teacher_cam(
#         teacher_features=teacher_features,
#         classifier_weight=teacher_classifier_weight,
#         teacher_labels=teacher_cam_labels,
#         eps=eps,
#     )

#     (
#         student_selected,
#         teacher_selected,
#         _,
#     ) = _gather_teacher_cam_regions(
#         student_tail_features=student_tail_features,
#         teacher_features=teacher_features,
#         teacher_cam=teacher_cam,
#         top_ratio=top_ratio,
#     )

#     if noise_std > 0.0:
#         gaussian_noise = torch.randn_like(
#             student_selected
#         )

#         if relative_noise:
#             feature_scale = (
#                 student_selected
#                 .detach()
#                 .std(
#                     dim=(1, 2),
#                     keepdim=True,
#                     unbiased=False,
#                 )
#                 .clamp_min(eps)
#             )

#             student_selected = (
#                 student_selected
#                 + noise_std
#                 * feature_scale
#                 * gaussian_noise
#             )

#         else:
#             student_selected = (
#                 student_selected
#                 + noise_std
#                 * gaussian_noise
#             )

#     if normalize_features:
#         student_selected = F.normalize(
#             student_selected,
#             p=2,
#             dim=-1,
#             eps=eps,
#         )

#         teacher_selected = F.normalize(
#             teacher_selected,
#             p=2,
#             dim=-1,
#             eps=eps,
#         )

#     teacher_selected = teacher_selected.detach()
#     loss = F.mse_loss(
#         student_selected,
#         teacher_selected,
#         reduction="mean",
#     )

#     return loss

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_classifier_weight(
    model: nn.Module,
) -> torch.Tensor:
    """
    获取分类器权重，返回 [num_classes, feature_dim]。

    支持常见 timm 模型：
        model.get_classifier()
        model.fc
        model.classifier
        model.head
    """
    classifier = None

    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()

        if isinstance(classifier, str):
            classifier = getattr(
                model,
                classifier,
            )

    if classifier is None:
        for name in (
            "fc",
            "classifier",
            "head",
        ):
            if hasattr(model, name):
                classifier = getattr(
                    model,
                    name,
                )
                break

    if classifier is None:
        raise AttributeError(
            "Cannot find classifier in the model."
        )

    if isinstance(classifier, nn.Linear):
        return classifier.weight

    if isinstance(classifier, nn.Conv2d):
        if classifier.kernel_size != (1, 1):
            raise ValueError(
                "Only a 1x1 Conv2d classifier is supported."
            )

        return classifier.weight[:, :, 0, 0]

    linear_layers = [
        module
        for module in classifier.modules()
        if isinstance(module, nn.Linear)
    ]

    if linear_layers:
        return linear_layers[-1].weight

    conv_layers = [
        module
        for module in classifier.modules()
        if isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
    ]

    if conv_layers:
        return conv_layers[-1].weight[
            :, :, 0, 0
        ]

    raise TypeError(
        "No supported classifier layer was found."
    )


def _map_global_to_teacher_labels(
    global_labels: torch.Tensor,
    teacher_classes: torch.Tensor,
) -> torch.Tensor:
    """
    将全局类别映射为教师模型的局部类别。

    例如：
        teacher_classes = [5, 6, 7]

        global 5 -> local 0
        global 6 -> local 1
        global 7 -> local 2

    teacher_classes 的顺序必须与教师训练时的类别顺序一致。
    """
    teacher_classes = teacher_classes.to(
        device=global_labels.device,
        dtype=torch.long,
    )

    matches = (
        global_labels.unsqueeze(1)
        == teacher_classes.unsqueeze(0)
    )

    valid_mask = matches.any(dim=1)

    if not valid_mask.all():
        invalid_labels = (
            global_labels[~valid_mask]
            .detach()
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Labels not contained in teacher_classes: "
            f"{invalid_labels}"
        )

    return matches.long().argmax(dim=1)


def _resize_teacher_features(
    teacher_features: torch.Tensor,
    target_spatial_shape: Sequence[int],
) -> torch.Tensor:
    """
    将教师特征空间尺寸调整到学生特征尺寸。

    支持：
        [B,C,H,W]
        [B,C,T,H,W]
    """
    if (
        tuple(teacher_features.shape[2:])
        == tuple(target_spatial_shape)
    ):
        return teacher_features

    if teacher_features.ndim == 4:
        return F.interpolate(
            teacher_features,
            size=tuple(target_spatial_shape),
            mode="bilinear",
            align_corners=False,
        )

    if teacher_features.ndim == 5:
        return F.interpolate(
            teacher_features,
            size=tuple(target_spatial_shape),
            mode="trilinear",
            align_corners=False,
        )

    raise ValueError(
        "teacher_features must be 4D or 5D, "
        f"but got {teacher_features.ndim}D."
    )


@torch.no_grad()
def _compute_soft_teacher_cam(
    teacher_features: torch.Tensor,
    classifier_weight: torch.Tensor,
    teacher_labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算 soft teacher CAM。

    输入：
        teacher_features:
            [B,C,H,W] 或 [B,C,T,H,W]

        classifier_weight:
            [num_teacher_classes,C]

        teacher_labels:
            [B]

    返回：
        cam:
            [B,H,W] 或 [B,T,H,W]
    """
    if teacher_features.ndim not in (
        4,
        5,
    ):
        raise ValueError(
            "teacher_features must have shape "
            "[B,C,H,W] or [B,C,T,H,W]."
        )

    batch_size = teacher_features.shape[0]
    channels = teacher_features.shape[1]

    if classifier_weight.ndim != 2:
        raise ValueError(
            "classifier_weight must have shape "
            "[num_classes, channels]."
        )

    if classifier_weight.shape[1] != channels:
        raise ValueError(
            "Teacher classifier feature dimension "
            "does not match teacher feature channels: "
            f"{classifier_weight.shape[1]} vs {channels}."
        )

    if teacher_labels.shape[0] != batch_size:
        raise ValueError(
            "teacher_labels batch size does not match "
            "teacher_features."
        )

    sample_weights = classifier_weight[
        teacher_labels
    ].to(
        device=teacher_features.device,
        dtype=teacher_features.dtype,
    )  # [B,C]

    flat_features = teacher_features.flatten(
        start_dim=2
    )  # [B,C,S]

    raw_cam = torch.einsum(
        "bc,bcs->bs",
        sample_weights,
        flat_features,
    )  # [B,S]

    cam = F.relu(raw_cam)

    # ReLU 后完全为 0 时，使用平移后的 raw CAM 作为回退
    zero_cam_mask = (
        cam.amax(
            dim=1,
            keepdim=True,
        )
        <= eps
    )

    fallback_cam = (
        raw_cam
        - raw_cam.amin(
            dim=1,
            keepdim=True,
        )
    )

    cam = torch.where(
        zero_cam_mask,
        fallback_cam,
        cam,
    )

    # 每个样本归一化至 [0,1]
    cam_min = cam.amin(
        dim=1,
        keepdim=True,
    )

    cam_max = cam.amax(
        dim=1,
        keepdim=True,
    )

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + eps
    )

    spatial_shape = teacher_features.shape[2:]

    return cam.reshape(
        batch_size,
        *spatial_shape,
    )


def _cam_weighted_pool(
    features: torch.Tensor,
    cam: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    CAM 加权平均池化。

    features:
        [B,C,H,W] 或 [B,C,T,H,W]

    cam:
        [B,H,W] 或 [B,T,H,W]

    返回：
        [B,C]
    """
    if features.shape[0] != cam.shape[0]:
        raise ValueError(
            "Feature batch size and CAM batch size differ."
        )

    if tuple(features.shape[2:]) != tuple(
        cam.shape[1:]
    ):
        raise ValueError(
            "Feature spatial shape and CAM shape differ: "
            f"{features.shape[2:]} vs {cam.shape[1:]}."
        )

    flat_features = features.flatten(
        start_dim=2
    )  # [B,C,S]

    flat_cam = cam.flatten(
        start_dim=1
    )  # [B,S]

    numerator = (
        flat_features
        * flat_cam.unsqueeze(1)
    ).sum(dim=2)  # [B,C]

    denominator = flat_cam.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    return numerator / denominator


def camGuided_IBP_Loss(
    student_model: nn.Module,
    teacher_model: nn.Module,
    student_features: torch.Tensor,
    teacher_features: Optional[torch.Tensor],
    labels: torch.Tensor,
    tail_mask: torch.Tensor,
    teacher_classes: Union[
        torch.Tensor,
        Sequence[int],
    ],
    adv_eps: float = 0.1,
    relative_eps: bool = True,
    cam_temperature: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Teacher-CAM-Guided Adversarial Foreground Consistency Loss。

    流程
    ----
    1. 计算冻结教师的 soft CAM；
    2. 使用教师 CAM 对教师和学生特征进行前景池化；
    3. 根据学生分类 CE 计算特征级 FGSM 梯度；
    4. 使用 1 - CAM 将扰动集中到背景；
    5. 计算扰动后的学生前景原型；
    6. 与干净教师前景原型进行 L2 对齐。

    参数
    ----
    student_model:
        学生模型。

    teacher_model:
        仅使用 teacher_classes 训练的尾类教师模型。

    student_features:
        学生完整 batch 的空间特征：
        [B,C,H,W] 或 [B,C,T,H,W]。

    teacher_features:
        教师对尾类样本提取的空间特征：
        [B_tail,C,H,W] 或 [B_tail,C,T,H,W]。

    labels:
        全局标签，例如 0~7。

    tail_mask:
        当前 batch 中 teacher_classes 对应样本的掩码。

    teacher_classes:
        教师局部类别到全局类别的顺序。
        例如 [5,6,7]。

    adv_eps:
        单步对抗扰动强度。

    relative_eps:
        True：按照学生特征尺度调整扰动；
        False：直接使用 adv_eps。

    cam_temperature:
        soft CAM 的温度/锐化系数。
        1.0 表示保持原 CAM；
        >1.0 会更加聚焦于高响应区域。

    返回
    ----
    loss:
        标量 IBP 损失。
    """
    if student_features.ndim not in (4, 5):
        raise ValueError(
            "student_features must be [B,C,H,W] "
            "or [B,C,T,H,W]."
        )

    tail_mask = tail_mask.to(
        device=labels.device,
        dtype=torch.bool,
    )

    if not tail_mask.any():
        return student_features.sum() * 0.0

    if teacher_features is None:
        raise ValueError(
            "teacher_features cannot be None when "
            "tail samples exist."
        )

    if not student_features.requires_grad:
        raise RuntimeError(
            "student_features does not require gradients. "
            "The adversarial perturbation cannot be generated."
        )

    teacher_classes = torch.as_tensor(
        teacher_classes,
        device=labels.device,
        dtype=torch.long,
    )

    student_tail_features = student_features[tail_mask]
    tail_global_labels = labels[tail_mask].long()
    teacher_features = teacher_features.detach()
    if (
        student_tail_features.shape[0]
        != teacher_features.shape[0]
    ):
        raise ValueError(
            "Student tail batch size and teacher batch size "
            "do not match: "
            f"{student_tail_features.shape[0]} vs "
            f"{teacher_features.shape[0]}."
        )

    if (
        student_tail_features.ndim
        != teacher_features.ndim
    ):
        raise ValueError(
            "Student and teacher feature dimensions differ."
        )

    if (
        student_tail_features.shape[1]
        != teacher_features.shape[1]
    ):
        raise ValueError(
            "Direct foreground alignment requires identical "
            "channel dimensions, but got "
            f"{student_tail_features.shape[1]} and "
            f"{teacher_features.shape[1]}."
        )

    # 将教师空间尺寸调整到学生空间尺寸
    teacher_features = _resize_teacher_features(
        teacher_features=teacher_features,
        target_spatial_shape=(
            student_tail_features.shape[2:]
        ),
    ).detach()

    teacher_classifier_weight = (
        _get_classifier_weight(
            teacher_model
        ).detach()
    )

    num_teacher_outputs = (
        teacher_classifier_weight.shape[0]
    )

    # 教师只输出局部尾类
    if (
        num_teacher_outputs
        == teacher_classes.numel()
    ):
        teacher_local_labels = (
            _map_global_to_teacher_labels(
                global_labels=tail_global_labels,
                teacher_classes=teacher_classes,
            )
        )

    # 教师仍然保留全局输出
    elif (
        tail_global_labels.max().item()
        < num_teacher_outputs
    ):
        teacher_local_labels = (
            tail_global_labels
        )

    else:
        raise ValueError(
            "Cannot infer teacher label mapping. "
            f"Teacher has {num_teacher_outputs} outputs, "
            f"while teacher_classes contains "
            f"{teacher_classes.numel()} classes."
        )

    # -------------------------------------------------
    # 1. 计算冻结教师的 soft CAM
    # -------------------------------------------------
    teacher_cam = _compute_soft_teacher_cam(
        teacher_features=teacher_features,
        classifier_weight=teacher_classifier_weight,
        teacher_labels=teacher_local_labels,
        eps=eps,
    ).detach()

    if cam_temperature <= 0:
        raise ValueError(
            "cam_temperature must be positive."
        )

    if cam_temperature != 1.0:
        # >1 时强化高 CAM 区域
        teacher_cam = teacher_cam.pow(
            cam_temperature
        )

    # -------------------------------------------------
    # 2. 计算干净教师前景原型
    # -------------------------------------------------
    with torch.no_grad():
        teacher_prototype = _cam_weighted_pool(
            features=teacher_features,
            cam=teacher_cam,
            eps=eps,
        )

        teacher_prototype = F.normalize(
            teacher_prototype,
            p=2,
            dim=1,
            eps=eps,
        )

    # -------------------------------------------------
    # 3. 根据分类 CE 生成特征级单步对抗扰动
    # -------------------------------------------------
    clean_tail_logits = student_model.forward_head(
        student_tail_features,
        pre_logits=False,
    )

    if isinstance(
        clean_tail_logits,
        (tuple, list),
    ):
        clean_tail_logits = clean_tail_logits[0]

    adversarial_seed_loss = F.cross_entropy(
        clean_tail_logits,
        tail_global_labels,
    )

    feature_gradient = torch.autograd.grad(
        outputs=adversarial_seed_loss,
        inputs=student_tail_features,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]

    # [B, spatial...] -> [B,1,spatial...]
    background_gate = (1.0 - teacher_cam).unsqueeze(1)

    if relative_eps:
        reduce_dims = tuple(
            range(
                1,
                student_tail_features.ndim,
            )
        )

        feature_scale = (
            student_tail_features.detach()
            .std(
                dim=reduce_dims,
                keepdim=True,
                unbiased=False,
            )
            .clamp_min(eps)
        )
    else:
        feature_scale = 1.0

    perturbation = (
        adv_eps
        * feature_scale
        * background_gate
        * feature_gradient.sign()
    ).detach()

    perturbed_student_features = (
        student_tail_features
        + perturbation
    )

    # -------------------------------------------------
    # 4. 计算扰动后的学生前景原型
    # -------------------------------------------------
    student_prototype = _cam_weighted_pool(
        features=perturbed_student_features,
        cam=teacher_cam,
        eps=eps,
    )

    student_prototype = F.normalize(
        student_prototype,
        p=2,
        dim=1,
        eps=eps,
    )

    # -------------------------------------------------
    # 5. 前景对抗一致性
    # -------------------------------------------------
    loss_per_sample = (student_prototype - teacher_prototype.detach()).pow(2).sum(dim=1)
    
    perturbed_tail_logits = student_model.forward_head(
        perturbed_student_features,
        pre_logits=False,
    )
    loss_adv_ce = F.cross_entropy(
        perturbed_tail_logits,
        tail_global_labels,
    )
    
    return loss_per_sample.mean() # + 0.5*loss_adv_ce