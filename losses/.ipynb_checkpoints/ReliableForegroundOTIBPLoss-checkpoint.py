
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# @dataclass
# class ReliableOTIBPOutput:
#     """
#     Reliable foreground OT distillation 的输出。

#     loss:
#         最终可靠性加权 IBP 损失。

#     reliability_weights:
#         完整 batch 的可靠性权重，[B]。非尾类样本为 0。

#     per_sample_ot_loss:
#         完整 batch 的逐样本 OT 损失，[B]。未参与蒸馏的样本为 0。

#     tail_mask:
#         完整 batch 中的尾类样本掩码，[B]。

#     teacher_correct:
#         尾类子 batch 中教师是否预测正确，[B_tail]。

#     teacher_has_advantage:
#         尾类子 batch 中教师是否相对学生具有 margin 优势，[B_tail]。

#     teacher_margin / student_margin:
#         尾类标签空间中的分类 margin，[B_tail]。

#     foreground_counts:
#         完整 batch 中每个样本被 CAM 选中的位置数，[B]。

#     valid_alignment:
#         完整 batch 中是否成功构建 OT 对齐，[B]。
#     """

#     loss: torch.Tensor
#     reliability_weights: torch.Tensor
#     per_sample_ot_loss: torch.Tensor
#     tail_mask: torch.Tensor
#     teacher_correct: torch.Tensor
#     teacher_has_advantage: torch.Tensor
#     teacher_margin: torch.Tensor
#     student_margin: torch.Tensor
#     foreground_counts: torch.Tensor
#     valid_alignment: torch.Tensor

#     def statistics(self) -> Dict[str, float]:
#         num_tail = int(self.tail_mask.sum().item())
#         num_reliable = int(
#             (self.reliability_weights > 0).sum().item()
#         )
#         num_valid = int(self.valid_alignment.sum().item())

#         tail_weights = self.reliability_weights[self.tail_mask]
#         valid_losses = self.per_sample_ot_loss[self.valid_alignment]

#         return {
#             "num_tail": float(num_tail),
#             "num_reliable": float(num_reliable),
#             "num_valid_alignment": float(num_valid),
#             "teacher_correct_ratio": (
#                 float(self.teacher_correct.float().mean().item())
#                 if self.teacher_correct.numel() > 0
#                 else 0.0
#             ),
#             "teacher_advantage_ratio": (
#                 float(
#                     self.teacher_has_advantage.float().mean().item()
#                 )
#                 if self.teacher_has_advantage.numel() > 0
#                 else 0.0
#             ),
#             "mean_reliability_weight": (
#                 float(tail_weights.mean().item())
#                 if tail_weights.numel() > 0
#                 else 0.0
#             ),
#             "mean_foreground_count": (
#                 float(
#                     self.foreground_counts[self.tail_mask]
#                     .float()
#                     .mean()
#                     .item()
#                 )
#                 if num_tail > 0
#                 else 0.0
#             ),
#             "mean_ot_loss": (
#                 float(valid_losses.mean().item())
#                 if valid_losses.numel() > 0
#                 else 0.0
#             ),
#             "loss_ibp": float(self.loss.detach().item()),
#         }


# def _unwrap_tensor_output(output: object, name: str) -> torch.Tensor:
#     """
#     兼容部分模型返回 Tensor 或 (Tensor, ...)。
#     """
#     if torch.is_tensor(output):
#         return output

#     if isinstance(output, (tuple, list)) and len(output) > 0:
#         if torch.is_tensor(output[0]):
#             return output[0]

#     raise TypeError(
#         f"{name} must be a Tensor or a tuple/list whose first item "
#         "is a Tensor."
#     )


# def get_classifier_weight(model: nn.Module) -> torch.Tensor:
#     """
#     尝试从常见 timm / torchvision / 自定义分类模型中读取最终线性分类器权重。

#     支持：
#         model.get_classifier()
#         model.fc
#         model.head
#         model.classifier

#     返回：
#         [num_classes, feature_dim]
#     """

#     candidates: List[object] = []

#     if hasattr(model, "get_classifier"):
#         try:
#             candidates.append(model.get_classifier())
#         except Exception:
#             pass

#     for attr_name in ("fc", "head", "classifier"):
#         if hasattr(model, attr_name):
#             candidates.append(getattr(model, attr_name))

#     for candidate in candidates:
#         if isinstance(candidate, nn.Linear):
#             return candidate.weight

#         if hasattr(candidate, "weight"):
#             weight = getattr(candidate, "weight")
#             if torch.is_tensor(weight) and weight.ndim == 2:
#                 return weight

#         if isinstance(candidate, nn.Sequential):
#             for layer in reversed(candidate):
#                 if isinstance(layer, nn.Linear):
#                     return layer.weight

#     raise AttributeError(
#         "Unable to locate the final linear classifier weight. "
#         "Pass teacher_classifier_weight explicitly to forward()."
#     )


# def _map_global_to_teacher_local(
#     global_labels: torch.Tensor,
#     teacher_classes: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     全局尾类标签映射到教师局部标签。

#     示例：
#         teacher_classes = [5, 6, 7]
#         global 5 -> local 0
#         global 6 -> local 1
#         global 7 -> local 2
#     """
#     match = (
#         global_labels.unsqueeze(1)
#         == teacher_classes.unsqueeze(0)
#     )

#     valid = match.any(dim=1)
#     if not valid.all():
#         invalid_labels = (
#             global_labels[~valid].detach().cpu().tolist()
#         )
#         raise ValueError(
#             "Some labels are not included in teacher_classes: "
#             f"{invalid_labels}"
#         )

#     return match.long().argmax(dim=1)


# def _classification_margin(
#     logits: torch.Tensor,
#     labels: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     真实类相对于最大非真实类的 logit margin：

#         margin_i = logit_{i,y_i} - max_{c != y_i} logit_{i,c}
#     """
#     if logits.ndim != 2:
#         raise ValueError(
#             f"logits must be [B, C], got {tuple(logits.shape)}."
#         )

#     if logits.shape[1] < 2:
#         raise ValueError(
#             "At least two teacher classes are required."
#         )

#     if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
#         raise ValueError(
#             "labels must be [B] and match logits batch size."
#         )

#     true_logit = logits.gather(
#         dim=1,
#         index=labels.unsqueeze(1),
#     ).squeeze(1)

#     true_mask = F.one_hot(
#         labels,
#         num_classes=logits.shape[1],
#     ).bool()

#     max_other = logits.masked_fill(
#         true_mask,
#         float("-inf"),
#     ).max(dim=1).values

#     return true_logit - max_other


# def _normalize_cam(cam: torch.Tensor, eps: float) -> torch.Tensor:
#     """
#     对每个样本的 CAM 做 min-max 归一化。

#     cam:
#         [B, S]，S 为展平后的空间位置数。
#     """
#     cam_min = cam.amin(dim=1, keepdim=True)
#     cam_max = cam.amax(dim=1, keepdim=True)

#     return (cam - cam_min) / (
#         cam_max - cam_min + eps
#     )


# def _resize_student_features(
#     student_features: torch.Tensor,
#     teacher_spatial_shape: Tuple[int, ...],
# ) -> torch.Tensor:
#     """
#     将学生空间特征插值到教师空间尺寸。

#     支持：
#         [B, C, L]
#         [B, C, H, W]
#         [B, C, D, H, W]
#     """
#     student_spatial_shape = tuple(student_features.shape[2:])

#     if student_spatial_shape == teacher_spatial_shape:
#         return student_features

#     spatial_dims = len(teacher_spatial_shape)

#     if spatial_dims == 1:
#         mode = "linear"
#     elif spatial_dims == 2:
#         mode = "bilinear"
#     elif spatial_dims == 3:
#         mode = "trilinear"
#     else:
#         raise ValueError(
#             "Only 1D, 2D, and 3D spatial feature maps are supported."
#         )

#     return F.interpolate(
#         student_features,
#         size=teacher_spatial_shape,
#         mode=mode,
#         align_corners=False,
#     )


# def _log_sinkhorn_transport(
#     cost: torch.Tensor,
#     source_mass: torch.Tensor,
#     target_mass: torch.Tensor,
#     regularization: float,
#     num_iterations: int,
#     eps: float,
# ) -> torch.Tensor:
#     """
#     在 log-domain 中求解熵正则化最优传输计划。

#     cost:
#         [Ns, Nt]

#     source_mass:
#         [Ns]，和为 1。

#     target_mass:
#         [Nt]，和为 1。
#     """
#     if regularization <= 0:
#         raise ValueError(
#             "OT regularization must be positive."
#         )

#     if num_iterations <= 0:
#         raise ValueError(
#             "num_iterations must be positive."
#         )

#     source_mass = source_mass.clamp_min(eps)
#     target_mass = target_mass.clamp_min(eps)

#     source_mass = source_mass / source_mass.sum()
#     target_mass = target_mass / target_mass.sum()

#     log_a = torch.log(source_mass)
#     log_b = torch.log(target_mass)
#     log_kernel = -cost / regularization

#     log_u = torch.zeros_like(log_a)
#     log_v = torch.zeros_like(log_b)

#     for _ in range(num_iterations):
#         log_u = log_a - torch.logsumexp(
#             log_kernel + log_v.unsqueeze(0),
#             dim=1,
#         )

#         log_v = log_b - torch.logsumexp(
#             log_kernel.transpose(0, 1)
#             + log_u.unsqueeze(0),
#             dim=1,
#         )

#     log_plan = (
#         log_u.unsqueeze(1)
#         + log_kernel
#         + log_v.unsqueeze(0)
#     )

#     plan = torch.exp(log_plan)

#     # 数值误差下重新归一化总质量。
#     return plan / plan.sum().clamp_min(eps)


# class ReliableForegroundOTIBPLoss(nn.Module):
#     """
#     三阶段 IBP：

#         Step 1:
#             质量门控。教师预测正确且 margin 优于学生时，
#             返回连续可靠性权重。

#         Step 2:
#             使用教师真实类别 CAM，选择 CAM > 样本均值的位置。

#         Step 3:
#             在教师 CAM 选中的位置，对教师与学生局部特征
#             进行熵正则化最优传输对齐。

#     最终：
#         L_IBP = sum_i weight_i * OT_i / sum_i weight_i

#     重要约定
#     --------
#     1. teacher_classes 顺序必须与教师局部分类器输出顺序一致。
#        例如教师输出 3 类，对应全局类别 [5, 6, 7]。

#     2. teacher_model 必须冻结，不应放入 optimizer。

#     3. student_features 和 teacher_features 应为：
#            [B, C, H, W]
#        或：
#            [B, C, D, H, W]

#     4. 若学生与教师通道数不同，会学习一个 Linear 投影，
#        将学生局部特征投影到教师特征空间。
#     """

#     def __init__(
#         self,
#         teacher_classes: Sequence[int],
#         student_feature_dim: int,
#         teacher_feature_dim: int,
#         tau_weight: float = 0.5,
#         minimum_margin_advantage: float = 0.0,
#         ot_regularization: float = 0.05,
#         sinkhorn_iterations: int = 30,
#         detach_transport_plan: bool = True,
#         normalize_teacher_cam: bool = True,
#         fallback_to_cam_argmax: bool = True,
#         max_foreground_points: Optional[int] = None,
#         eps: float = 1e-6,
#     ) -> None:
#         super().__init__()

#         if len(teacher_classes) < 2:
#             raise ValueError(
#                 "teacher_classes must contain at least two classes."
#             )

#         if student_feature_dim <= 0 or teacher_feature_dim <= 0:
#             raise ValueError(
#                 "Feature dimensions must be positive."
#             )

#         if tau_weight <= 0:
#             raise ValueError(
#                 "tau_weight must be positive."
#             )

#         if max_foreground_points is not None:
#             if max_foreground_points <= 0:
#                 raise ValueError(
#                     "max_foreground_points must be positive or None."
#                 )

#         self.register_buffer(
#             "teacher_classes",
#             torch.as_tensor(
#                 teacher_classes,
#                 dtype=torch.long,
#             ),
#             persistent=True,
#         )

#         self.student_feature_dim = int(student_feature_dim)
#         self.teacher_feature_dim = int(teacher_feature_dim)

#         if self.student_feature_dim == self.teacher_feature_dim:
#             self.student_projection: nn.Module = nn.Identity()
#         else:
#             self.student_projection = nn.Linear(
#                 self.student_feature_dim,
#                 self.teacher_feature_dim,
#                 bias=False,
#             )

#         self.tau_weight = float(tau_weight)
#         self.minimum_margin_advantage = float(
#             minimum_margin_advantage
#         )
#         self.ot_regularization = float(ot_regularization)
#         self.sinkhorn_iterations = int(sinkhorn_iterations)
#         self.detach_transport_plan = bool(
#             detach_transport_plan
#         )
#         self.normalize_teacher_cam = bool(
#             normalize_teacher_cam
#         )
#         self.fallback_to_cam_argmax = bool(
#             fallback_to_cam_argmax
#         )
#         self.max_foreground_points = max_foreground_points
#         self.eps = float(eps)

#     @torch.no_grad()
#     def _compute_reliability_weights(
#         self,
#         student_logits: torch.Tensor,
#         teacher_logits: torch.Tensor,
#         tail_labels_global: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         """
#         返回：
#             weights
#             local_labels
#             teacher_correct
#             teacher_has_advantage
#             teacher_margin
#             student_margin
#         """
#         teacher_classes = self.teacher_classes.to(
#             student_logits.device
#         )

#         local_labels = _map_global_to_teacher_local(
#             global_labels=tail_labels_global,
#             teacher_classes=teacher_classes,
#         )

#         num_teacher_classes = int(
#             teacher_classes.numel()
#         )

#         # 教师可以输出局部尾类空间，也可以输出完整类别空间。
#         if teacher_logits.shape[1] == num_teacher_classes:
#             teacher_tail_logits = teacher_logits
#         elif (
#             teacher_logits.shape[1]
#             > int(teacher_classes.max().item())
#         ):
#             teacher_tail_logits = teacher_logits.index_select(
#                 dim=1,
#                 index=teacher_classes,
#             )
#         else:
#             raise ValueError(
#                 "teacher_logits cannot be aligned with teacher_classes."
#             )

#         if (
#             student_logits.shape[1]
#             <= int(teacher_classes.max().item())
#         ):
#             raise ValueError(
#                 "student_logits does not contain every teacher class."
#             )

#         student_tail_logits = student_logits.index_select(
#             dim=1,
#             index=teacher_classes,
#         )

#         teacher_prediction = teacher_tail_logits.argmax(dim=1)
#         teacher_correct = teacher_prediction.eq(local_labels)

#         teacher_margin = _classification_margin(
#             logits=teacher_tail_logits,
#             labels=local_labels,
#         )

#         student_margin = _classification_margin(
#             logits=student_tail_logits,
#             labels=local_labels,
#         )

#         margin_advantage = (
#             teacher_margin
#             - student_margin
#             - self.minimum_margin_advantage
#         )

#         teacher_has_advantage = margin_advantage.gt(0)

#         soft_advantage = torch.sigmoid(
#             margin_advantage / self.tau_weight
#         )

#         weights = (
#             teacher_correct.to(student_logits.dtype)
#             * teacher_has_advantage.to(student_logits.dtype)
#             * soft_advantage
#         ).detach()

#         return (
#             weights,
#             local_labels,
#             teacher_correct,
#             teacher_has_advantage,
#             teacher_margin,
#             student_margin,
#         )

#     @torch.no_grad()
#     def _compute_teacher_cam(
#         self,
#         teacher_features: torch.Tensor,
#         local_labels: torch.Tensor,
#         classifier_weight: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         返回展平 CAM：
#             [B_tail, S]
#         """
#         if teacher_features.ndim < 3:
#             raise ValueError(
#                 "teacher_features must be [B, C, *spatial]."
#             )

#         batch_size = teacher_features.shape[0]
#         channel_dim = teacher_features.shape[1]
#         spatial_size = int(
#             teacher_features[0, 0].numel()
#         )

#         if classifier_weight.ndim != 2:
#             raise ValueError(
#                 "classifier_weight must be [num_classes, C]."
#             )

#         if classifier_weight.shape[1] != channel_dim:
#             raise ValueError(
#                 "Teacher classifier input dimension does not match "
#                 f"teacher feature channels: {classifier_weight.shape[1]} "
#                 f"vs {channel_dim}."
#             )

#         if classifier_weight.shape[0] <= int(
#             local_labels.max().item()
#         ):
#             raise ValueError(
#                 "Teacher classifier does not contain every local label."
#             )

#         selected_weight = classifier_weight.index_select(
#             dim=0,
#             index=local_labels,
#         )

#         flat_features = teacher_features.reshape(
#             batch_size,
#             channel_dim,
#             spatial_size,
#         )

#         cam = torch.einsum(
#             "bc,bcs->bs",
#             selected_weight,
#             flat_features,
#         )

#         cam = F.relu(cam)

#         if self.normalize_teacher_cam:
#             cam = _normalize_cam(
#                 cam=cam,
#                 eps=self.eps,
#             )

#         return cam

#     def _sample_ot_loss(
#         self,
#         student_flat: torch.Tensor,
#         teacher_flat: torch.Tensor,
#         cam_flat: torch.Tensor,
#     ) -> Tuple[torch.Tensor, int, bool]:
#         """
#         单样本 OT 对齐。

#         student_flat:
#             [S, C_s]

#         teacher_flat:
#             [S, C_t]

#         cam_flat:
#             [S]
#         """
#         mean_activation = cam_flat.mean()
#         selected_mask = cam_flat.gt(mean_activation)
#         selected_indices = selected_mask.nonzero(
#             as_tuple=False
#         ).squeeze(1)

#         # 极端情况下 CAM 可能为常数，严格 > mean 会选不到位置。
#         # 为避免训练崩溃，可退化为 CAM 最大位置。
#         if selected_indices.numel() == 0:
#             if not self.fallback_to_cam_argmax:
#                 zero_loss = student_flat.sum() * 0.0
#                 return zero_loss, 0, False

#             selected_indices = cam_flat.argmax().reshape(1)

#         if (
#             self.max_foreground_points is not None
#             and selected_indices.numel()
#             > self.max_foreground_points
#         ):
#             selected_values = cam_flat.index_select(
#                 dim=0,
#                 index=selected_indices,
#             )

#             top_relative = torch.topk(
#                 selected_values,
#                 k=self.max_foreground_points,
#                 largest=True,
#                 sorted=False,
#             ).indices

#             selected_indices = selected_indices.index_select(
#                 dim=0,
#                 index=top_relative,
#             )

#         student_selected = student_flat.index_select(
#             dim=0,
#             index=selected_indices,
#         )

#         teacher_selected = teacher_flat.index_select(
#             dim=0,
#             index=selected_indices,
#         ).detach()

#         student_projected = self.student_projection(
#             student_selected
#         )

#         student_normalized = F.normalize(
#             student_projected,
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         teacher_normalized = F.normalize(
#             teacher_selected,
#             p=2,
#             dim=1,
#             eps=self.eps,
#         )

#         # 余弦距离，范围通常位于 [0, 2]。
#         cost = 1.0 - (
#             student_normalized
#             @ teacher_normalized.transpose(0, 1)
#         )

#         selected_cam = cam_flat.index_select(
#             dim=0,
#             index=selected_indices,
#         ).detach()

#         if selected_cam.sum().item() <= self.eps:
#             mass = torch.full_like(
#                 selected_cam,
#                 fill_value=1.0 / selected_cam.numel(),
#             )
#         else:
#             mass = selected_cam.clamp_min(self.eps)
#             mass = mass / mass.sum()

#         plan_cost = (
#             cost.detach()
#             if self.detach_transport_plan
#             else cost
#         )

#         transport_plan = _log_sinkhorn_transport(
#             cost=plan_cost,
#             source_mass=mass,
#             target_mass=mass,
#             regularization=self.ot_regularization,
#             num_iterations=self.sinkhorn_iterations,
#             eps=self.eps,
#         )

#         if self.detach_transport_plan:
#             transport_plan = transport_plan.detach()

#         ot_loss = torch.sum(
#             transport_plan * cost
#         )

#         return (
#             ot_loss,
#             int(selected_indices.numel()),
#             True,
#         )

#     def forward(
#         self,
#         student_features: torch.Tensor,
#         student_logits: torch.Tensor,
#         teacher_features: torch.Tensor,
#         teacher_logits: torch.Tensor,
#         labels: torch.Tensor,
#         teacher_classifier_weight: torch.Tensor,
#         tail_mask: Optional[torch.Tensor] = None,
#     ) -> ReliableOTIBPOutput:
#         """
#         基于已经计算好的学生/教师输出计算 IBP。

#         student_features:
#             完整 batch 学生空间特征，[B, C_s, *spatial]。

#         student_logits:
#             完整 batch 学生 logits，[B, C_all]。

#         teacher_features:
#             尾类子 batch 教师空间特征，[B_tail, C_t, *spatial]。

#         teacher_logits:
#             尾类子 batch 教师 logits，[B_tail, C_tail]
#             或 [B_tail, C_all]。

#         labels:
#             完整 batch 全局标签，[B]。

#         teacher_classifier_weight:
#             教师分类器权重。对于尾类子任务教师通常为：
#                 [C_tail, C_t]

#         tail_mask:
#             可选完整 batch 掩码。[B]。
#             未提供时根据 teacher_classes 自动构造。
#         """
#         if student_features.ndim < 3:
#             raise ValueError(
#                 "student_features must be [B, C, *spatial]."
#             )

#         if student_logits.ndim != 2:
#             raise ValueError(
#                 "student_logits must be [B, C_all]."
#             )

#         if labels.ndim != 1:
#             raise ValueError(
#                 "labels must be [B]."
#             )

#         batch_size = labels.shape[0]

#         if student_features.shape[0] != batch_size:
#             raise ValueError(
#                 "student_features and labels batch sizes do not match."
#             )

#         if student_logits.shape[0] != batch_size:
#             raise ValueError(
#                 "student_logits and labels batch sizes do not match."
#             )

#         if student_features.shape[1] != self.student_feature_dim:
#             raise ValueError(
#                 "student_features channel dimension does not match "
#                 f"student_feature_dim: {student_features.shape[1]} "
#                 f"vs {self.student_feature_dim}."
#             )

#         device = student_features.device
#         dtype = student_features.dtype

#         teacher_classes = self.teacher_classes.to(device)

#         if tail_mask is None:
#             tail_mask = torch.isin(
#                 labels,
#                 teacher_classes,
#             )
#         else:
#             tail_mask = tail_mask.to(
#                 device=device,
#                 dtype=torch.bool,
#             )

#         num_tail = int(tail_mask.sum().item())

#         full_weights = torch.zeros(
#             batch_size,
#             device=device,
#             dtype=dtype,
#         )

#         full_ot_losses = torch.zeros(
#             batch_size,
#             device=device,
#             dtype=dtype,
#         )

#         foreground_counts = torch.zeros(
#             batch_size,
#             device=device,
#             dtype=torch.long,
#         )

#         valid_alignment = torch.zeros(
#             batch_size,
#             device=device,
#             dtype=torch.bool,
#         )

#         if num_tail == 0:
#             empty_float = torch.empty(
#                 0,
#                 device=device,
#                 dtype=dtype,
#             )
#             empty_bool = torch.empty(
#                 0,
#                 device=device,
#                 dtype=torch.bool,
#             )

#             zero_loss = (
#                 student_features.sum()
#                 + student_logits.sum()
#             ) * 0.0

#             return ReliableOTIBPOutput(
#                 loss=zero_loss,
#                 reliability_weights=full_weights,
#                 per_sample_ot_loss=full_ot_losses,
#                 tail_mask=tail_mask,
#                 teacher_correct=empty_bool,
#                 teacher_has_advantage=empty_bool,
#                 teacher_margin=empty_float,
#                 student_margin=empty_float,
#                 foreground_counts=foreground_counts,
#                 valid_alignment=valid_alignment,
#             )

#         if teacher_features.shape[0] != num_tail:
#             raise ValueError(
#                 "teacher_features batch size must equal the number "
#                 f"of tail samples: {teacher_features.shape[0]} vs {num_tail}."
#             )

#         if teacher_logits.shape[0] != num_tail:
#             raise ValueError(
#                 "teacher_logits batch size must equal the number "
#                 f"of tail samples: {teacher_logits.shape[0]} vs {num_tail}."
#             )

#         if teacher_features.shape[1] != self.teacher_feature_dim:
#             raise ValueError(
#                 "teacher_features channel dimension does not match "
#                 f"teacher_feature_dim: {teacher_features.shape[1]} "
#                 f"vs {self.teacher_feature_dim}."
#             )

#         tail_labels = labels[tail_mask]
#         student_tail_logits = student_logits[tail_mask]

#         (
#             tail_weights,
#             local_labels,
#             teacher_correct,
#             teacher_has_advantage,
#             teacher_margin,
#             student_margin,
#         ) = self._compute_reliability_weights(
#             student_logits=student_tail_logits,
#             teacher_logits=teacher_logits.detach(),
#             tail_labels_global=tail_labels,
#         )

#         full_weights[tail_mask] = tail_weights.to(dtype)

#         teacher_features = teacher_features.detach().to(
#             device=device,
#             dtype=dtype,
#         )

#         teacher_classifier_weight = (
#             teacher_classifier_weight.detach().to(
#                 device=device,
#                 dtype=dtype,
#             )
#         )

#         teacher_cam = self._compute_teacher_cam(
#             teacher_features=teacher_features,
#             local_labels=local_labels,
#             classifier_weight=teacher_classifier_weight,
#         )

#         student_tail_features = student_features[tail_mask]

#         teacher_spatial_shape = tuple(
#             teacher_features.shape[2:]
#         )

#         student_tail_features = _resize_student_features(
#             student_features=student_tail_features,
#             teacher_spatial_shape=teacher_spatial_shape,
#         )

#         spatial_size = int(
#             teacher_features[0, 0].numel()
#         )

#         student_flat = student_tail_features.reshape(
#             num_tail,
#             self.student_feature_dim,
#             spatial_size,
#         ).transpose(1, 2)

#         teacher_flat = teacher_features.reshape(
#             num_tail,
#             self.teacher_feature_dim,
#             spatial_size,
#         ).transpose(1, 2)

#         tail_indices_in_full_batch = tail_mask.nonzero(
#             as_tuple=False
#         ).squeeze(1)

#         tail_ot_loss_list: List[torch.Tensor] = []
#         tail_valid_list: List[bool] = []
#         tail_count_list: List[int] = []

#         for tail_index in range(num_tail):
#             sample_ot_loss, count, is_valid = (
#                 self._sample_ot_loss(
#                     student_flat=student_flat[tail_index],
#                     teacher_flat=teacher_flat[tail_index],
#                     cam_flat=teacher_cam[tail_index],
#                 )
#             )

#             tail_ot_loss_list.append(sample_ot_loss)
#             tail_count_list.append(count)
#             tail_valid_list.append(is_valid)

#         tail_ot_losses = torch.stack(
#             tail_ot_loss_list,
#             dim=0,
#         )

#         tail_valid = torch.tensor(
#             tail_valid_list,
#             device=device,
#             dtype=torch.bool,
#         )

#         tail_counts = torch.tensor(
#             tail_count_list,
#             device=device,
#             dtype=torch.long,
#         )

#         full_ot_losses = full_ot_losses.scatter(
#             dim=0,
#             index=tail_indices_in_full_batch,
#             src=tail_ot_losses,
#         )

#         foreground_counts[tail_mask] = tail_counts
#         valid_alignment[tail_mask] = tail_valid

#         effective_tail_weights = (
#             tail_weights.to(dtype)
#             * tail_valid.to(dtype)
#         )

#         denominator = effective_tail_weights.sum()

#         if denominator.detach().item() <= self.eps:
#             loss = (
#                 tail_ot_losses.sum()
#                 + student_logits.sum() * 0.0
#             ) * 0.0
#         else:
#             loss = (
#                 effective_tail_weights
#                 * tail_ot_losses
#             ).sum() / denominator.clamp_min(self.eps)

#         return ReliableOTIBPOutput(
#             loss=loss,
#             reliability_weights=full_weights,
#             per_sample_ot_loss=full_ot_losses,
#             tail_mask=tail_mask,
#             teacher_correct=teacher_correct,
#             teacher_has_advantage=teacher_has_advantage,
#             teacher_margin=teacher_margin,
#             student_margin=student_margin,
#             foreground_counts=foreground_counts,
#             valid_alignment=valid_alignment,
#         )

#     def forward_from_models(
#         self,
#         student_model: nn.Module,
#         teacher_model: nn.Module,
#         inputs: torch.Tensor,
#         labels: torch.Tensor,
#         tail_mask: torch.Tensor,
#         student_features: Optional[torch.Tensor] = None,
#         student_logits: Optional[torch.Tensor] = None,
#         teacher_classifier_weight: Optional[torch.Tensor] = None,
#     ) -> ReliableOTIBPOutput:
#         """
#         便捷接口：直接从模型计算输出。

#         为避免学生 backbone 重复前向，可以将训练循环中已经得到的
#         student_features 和 student_logits 传进来。
#         """
#         if student_features is None:
#             student_features = _unwrap_tensor_output(
#                 student_model.forward_features(inputs),
#                 "student_model.forward_features(inputs)",
#             )

#         if student_logits is None:
#             student_logits = _unwrap_tensor_output(
#                 student_model.forward_head(
#                     student_features,
#                     pre_logits=False,
#                 ),
#                 "student_model.forward_head(...)",
#             )

#         teacher_model.eval()
#         teacher_classes = self.teacher_classes.to(labels.device)
#         with torch.no_grad():
#             teacher_features = _unwrap_tensor_output(
#                 teacher_model.forward_features(
#                     inputs[tail_mask]
#                 ),
#                 "teacher_model.forward_features(...)",
#             )

#             teacher_logits = _unwrap_tensor_output(
#                 teacher_model.forward_head(
#                     teacher_features,
#                     pre_logits=False,
#                 ),
#                 "teacher_model.forward_head(...)",
#             )

#         if teacher_classifier_weight is None:
#             teacher_classifier_weight = get_classifier_weight(teacher_model)

#         return self.forward(
#             student_features=student_features,
#             student_logits=student_logits,
#             teacher_features=teacher_features,
#             teacher_logits=teacher_logits,
#             labels=labels,
#             teacher_classifier_weight=teacher_classifier_weight,
#             tail_mask=tail_mask,
#         )


# # -------------------------------------------------------------------------
# # 一个最小可运行示例
# # -------------------------------------------------------------------------

# class _DemoCNN(nn.Module):
#     def __init__(
#         self,
#         feature_dim: int,
#         num_classes: int,
#     ) -> None:
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, feature_dim, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 feature_dim,
#                 feature_dim,
#                 3,
#                 padding=1,
#             ),
#             nn.ReLU(inplace=True),
#         )

#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(
#             feature_dim,
#             num_classes,
#         )
#         self.num_features = feature_dim

#     def forward_features(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         return self.features(x)

#     def forward_head(
#         self,
#         x: torch.Tensor,
#         pre_logits: bool = False,
#     ) -> torch.Tensor:
#         pooled = self.global_pool(x).flatten(1)

#         if pre_logits:
#             return pooled

#         return self.fc(pooled)

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         return self.forward_head(
#             self.forward_features(x),
#             pre_logits=False,
#         )

#     def get_classifier(self) -> nn.Module:
#         return self.fc


# def demo() -> None:
#     torch.manual_seed(11)

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     teacher_classes = [5, 6, 7]

#     student_model = _DemoCNN(
#         feature_dim=32,
#         num_classes=8,
#     ).to(device)

#     teacher_model = _DemoCNN(
#         feature_dim=24,
#         num_classes=3,
#     ).to(device)

#     # 冻结教师
#     for parameter in teacher_model.parameters():
#         parameter.requires_grad_(False)

#     ibp_loss_module = ReliableForegroundOTIBPLoss(
#         teacher_classes=teacher_classes,
#         student_feature_dim=32,
#         teacher_feature_dim=24,
#         tau_weight=0.5,
#         minimum_margin_advantage=0.0,
#         ot_regularization=0.05,
#         sinkhorn_iterations=20,
#         detach_transport_plan=True,
#         normalize_teacher_cam=True,
#         fallback_to_cam_argmax=True,
#         max_foreground_points=64,
#     ).to(device)

#     optimizer = torch.optim.AdamW(
#         list(student_model.parameters())
#         + list(ibp_loss_module.parameters()),
#         lr=1e-3,
#     )

#     inputs = torch.randn(
#         10,
#         3,
#         16,
#         16,
#         device=device,
#     )

#     labels = torch.tensor(
#         [0, 5, 6, 1, 7, 5, 3, 6, 2, 7],
#         device=device,
#         dtype=torch.long,
#     )

#     student_features = student_model.forward_features(
#         inputs
#     )

#     student_logits = student_model.forward_head(
#         student_features,
#         pre_logits=False,
#     )

#     loss_cls = F.cross_entropy(
#         student_logits,
#         labels,
#     )

#     ibp_output = ibp_loss_module.forward_from_models(
#         student_model=student_model,
#         teacher_model=teacher_model,
#         inputs=inputs,
#         labels=labels,
#         student_features=student_features,
#         student_logits=student_logits,
#     )

#     beta_ibp = 0.5
#     total_loss = (
#         loss_cls
#         + beta_ibp * ibp_output.loss
#     )

#     optimizer.zero_grad(set_to_none=True)
#     total_loss.backward()
#     optimizer.step()

#     print("Demo finished successfully.")
#     print("IBP statistics:", ibp_output.statistics())
#     print("Classification loss:", float(loss_cls.detach()))
#     print("IBP loss:", float(ibp_output.loss.detach()))
#     print("Total loss:", float(total_loss.detach()))


# if __name__ == "__main__":
#     demo()




from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReliableOTIBPOutput:
    """
    Reliable foreground OT distillation 的输出。

    loss:
        最终可靠性加权 IBP 损失。

    reliability_weights:
        完整 batch 的可靠性权重，[B]。非尾类样本为 0。

    per_sample_ot_loss:
        完整 batch 的逐样本 OT 损失，[B]。未参与蒸馏的样本为 0。

    tail_mask:
        完整 batch 中的尾类样本掩码，[B]。

    teacher_correct:
        尾类子 batch 中教师是否预测正确，[B_tail]。

    teacher_has_advantage:
        尾类子 batch 中教师是否相对学生具有 margin 优势，[B_tail]。

    teacher_margin / student_margin:
        尾类标签空间中的分类 margin，[B_tail]。

    foreground_counts:
        完整 batch 中每个样本被 CAM 选中的位置数，[B]。

    valid_alignment:
        完整 batch 中是否成功构建 OT 对齐，[B]。
    """

    loss: torch.Tensor
    reliability_weights: torch.Tensor
    per_sample_ot_loss: torch.Tensor
    tail_mask: torch.Tensor
    teacher_correct: torch.Tensor
    teacher_has_advantage: torch.Tensor
    teacher_margin: torch.Tensor
    student_margin: torch.Tensor
    foreground_counts: torch.Tensor
    valid_alignment: torch.Tensor

    def statistics(self) -> Dict[str, float]:
        num_tail = int(self.tail_mask.sum().item())
        num_reliable = int(
            (self.reliability_weights > 0).sum().item()
        )
        num_valid = int(self.valid_alignment.sum().item())

        tail_weights = self.reliability_weights[self.tail_mask]
        valid_losses = self.per_sample_ot_loss[self.valid_alignment]

        return {
            "num_tail": float(num_tail),
            "num_reliable": float(num_reliable),
            "num_valid_alignment": float(num_valid),
            "teacher_correct_ratio": (
                float(self.teacher_correct.float().mean().item())
                if self.teacher_correct.numel() > 0
                else 0.0
            ),
            "teacher_advantage_ratio": (
                float(
                    self.teacher_has_advantage.float().mean().item()
                )
                if self.teacher_has_advantage.numel() > 0
                else 0.0
            ),
            "mean_reliability_weight": (
                float(tail_weights.mean().item())
                if tail_weights.numel() > 0
                else 0.0
            ),
            "mean_foreground_count": (
                float(
                    self.foreground_counts[self.tail_mask]
                    .float()
                    .mean()
                    .item()
                )
                if num_tail > 0
                else 0.0
            ),
            "mean_ot_loss": (
                float(valid_losses.mean().item())
                if valid_losses.numel() > 0
                else 0.0
            ),
            "loss_ibp": float(self.loss.detach().item()),
        }


def _unwrap_tensor_output(output: object, name: str) -> torch.Tensor:
    """
    兼容部分模型返回 Tensor 或 (Tensor, ...)。
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)) and len(output) > 0:
        if torch.is_tensor(output[0]):
            return output[0]

    raise TypeError(
        f"{name} must be a Tensor or a tuple/list whose first item "
        "is a Tensor."
    )



def _unwrap_parallel_model(model: nn.Module) -> nn.Module:
    """兼容 DataParallel / DistributedDataParallel 包装。"""
    current = model
    visited = set()

    while hasattr(current, "module"):
        module = getattr(current, "module")
        if not isinstance(module, nn.Module):
            break
        if id(module) in visited or module is current:
            break
        visited.add(id(current))
        current = module

    return current


def _validate_resnet_layer(layer: int) -> int:
    layer = int(layer)
    if layer not in (1, 2, 3, 4):
        raise ValueError(
            f"layer must be one of 1, 2, 3, or 4, got {layer}."
        )
    return layer


def _forward_timm_resnet_stage(
    model: nn.Module,
    inputs: torch.Tensor,
    layer: int,
    return_final: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    从 timm/torchvision 风格 ResNet 内部提取 layer1~layer4 特征。

    返回：
        selected_features: 指定层输出。
        final_features: layer4 输出；return_final=False 时可为 None。
    """
    layer = _validate_resnet_layer(layer)
    backbone = _unwrap_parallel_model(model)

    required_names = (
        "conv1",
        "bn1",
        "act1",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    )
    missing = [name for name in required_names if not hasattr(backbone, name)]
    if missing:
        raise AttributeError(
            "forward_from_models(layer=...) currently supports timm/"
            "torchvision-style ResNet backbones. Missing attributes: "
            f"{missing}."
        )

    x = backbone.conv1(inputs)
    x = backbone.bn1(x)
    x = backbone.act1(x)
    x = backbone.maxpool(x)

    selected_features: Optional[torch.Tensor] = None
    for stage_index in range(1, 5):
        x = getattr(backbone, f"layer{stage_index}")(x)

        if stage_index == layer:
            selected_features = x
            if not return_final:
                return selected_features, None

    if selected_features is None:
        raise RuntimeError("Failed to extract the selected ResNet layer.")

    return selected_features, x


def _forward_model_head(
    model: nn.Module,
    final_features: torch.Tensor,
) -> torch.Tensor:
    """使用模型原有 forward_head 从 layer4 特征计算 logits。"""
    backbone = _unwrap_parallel_model(model)
    if not hasattr(backbone, "forward_head"):
        raise AttributeError(
            "The model must provide forward_head(final_features, "
            "pre_logits=False)."
        )

    return _unwrap_tensor_output(
        backbone.forward_head(
            final_features,
            pre_logits=False,
        ),
        "model.forward_head(...)"
    )


def _resize_flat_cam(
    cam_flat: torch.Tensor,
    source_spatial_shape: Tuple[int, ...],
    target_spatial_shape: Tuple[int, ...],
) -> torch.Tensor:
    """将由最终层生成的 CAM 插值到所选中间层的空间尺寸。"""
    if source_spatial_shape == target_spatial_shape:
        return cam_flat

    spatial_dims = len(target_spatial_shape)
    if spatial_dims == 1:
        mode = "linear"
    elif spatial_dims == 2:
        mode = "bilinear"
    elif spatial_dims == 3:
        mode = "trilinear"
    else:
        raise ValueError(
            "Only 1D, 2D, and 3D spatial feature maps are supported."
        )

    cam = cam_flat.reshape(
        cam_flat.shape[0],
        1,
        *source_spatial_shape,
    )
    cam = F.interpolate(
        cam,
        size=target_spatial_shape,
        mode=mode,
        align_corners=False,
    )
    return cam.flatten(start_dim=1)


def get_classifier_weight(model: nn.Module) -> torch.Tensor:
    """
    尝试从常见 timm / torchvision / 自定义分类模型中读取最终线性分类器权重。

    支持：
        model.get_classifier()
        model.fc
        model.head
        model.classifier

    返回：
        [num_classes, feature_dim]
    """

    model = _unwrap_parallel_model(model)

    candidates: List[object] = []

    if hasattr(model, "get_classifier"):
        try:
            candidates.append(model.get_classifier())
        except Exception:
            pass

    for attr_name in ("fc", "head", "classifier"):
        if hasattr(model, attr_name):
            candidates.append(getattr(model, attr_name))

    for candidate in candidates:
        if isinstance(candidate, nn.Linear):
            return candidate.weight

        if hasattr(candidate, "weight"):
            weight = getattr(candidate, "weight")
            if torch.is_tensor(weight) and weight.ndim == 2:
                return weight

        if isinstance(candidate, nn.Sequential):
            for layer in reversed(candidate):
                if isinstance(layer, nn.Linear):
                    return layer.weight

    raise AttributeError(
        "Unable to locate the final linear classifier weight. "
        "Pass teacher_classifier_weight explicitly to forward()."
    )


def _map_global_to_teacher_local(
    global_labels: torch.Tensor,
    teacher_classes: torch.Tensor,
) -> torch.Tensor:
    """
    全局尾类标签映射到教师局部标签。

    示例：
        teacher_classes = [5, 6, 7]
        global 5 -> local 0
        global 6 -> local 1
        global 7 -> local 2
    """
    match = (
        global_labels.unsqueeze(1)
        == teacher_classes.unsqueeze(0)
    )

    valid = match.any(dim=1)
    if not valid.all():
        invalid_labels = (
            global_labels[~valid].detach().cpu().tolist()
        )
        raise ValueError(
            "Some labels are not included in teacher_classes: "
            f"{invalid_labels}"
        )

    return match.long().argmax(dim=1)


def _classification_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    真实类相对于最大非真实类的 logit margin：

        margin_i = logit_{i,y_i} - max_{c != y_i} logit_{i,c}
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be [B, C], got {tuple(logits.shape)}."
        )

    if logits.shape[1] < 2:
        raise ValueError(
            "At least two teacher classes are required."
        )

    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            "labels must be [B] and match logits batch size."
        )

    true_logit = logits.gather(
        dim=1,
        index=labels.unsqueeze(1),
    ).squeeze(1)

    true_mask = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).bool()

    max_other = logits.masked_fill(
        true_mask,
        float("-inf"),
    ).max(dim=1).values

    return true_logit - max_other


def _normalize_cam(cam: torch.Tensor, eps: float) -> torch.Tensor:
    """
    对每个样本的 CAM 做 min-max 归一化。

    cam:
        [B, S]，S 为展平后的空间位置数。
    """
    cam_min = cam.amin(dim=1, keepdim=True)
    cam_max = cam.amax(dim=1, keepdim=True)

    return (cam - cam_min) / (
        cam_max - cam_min + eps
    )


def _resize_student_features(
    student_features: torch.Tensor,
    teacher_spatial_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    将学生空间特征插值到教师空间尺寸。

    支持：
        [B, C, L]
        [B, C, H, W]
        [B, C, D, H, W]
    """
    student_spatial_shape = tuple(student_features.shape[2:])

    if student_spatial_shape == teacher_spatial_shape:
        return student_features

    spatial_dims = len(teacher_spatial_shape)

    if spatial_dims == 1:
        mode = "linear"
    elif spatial_dims == 2:
        mode = "bilinear"
    elif spatial_dims == 3:
        mode = "trilinear"
    else:
        raise ValueError(
            "Only 1D, 2D, and 3D spatial feature maps are supported."
        )

    return F.interpolate(
        student_features,
        size=teacher_spatial_shape,
        mode=mode,
        align_corners=False,
    )


def _log_sinkhorn_transport(
    cost: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    regularization: float,
    num_iterations: int,
    eps: float,
) -> torch.Tensor:
    """
    在 log-domain 中求解熵正则化最优传输计划。

    cost:
        [Ns, Nt]

    source_mass:
        [Ns]，和为 1。

    target_mass:
        [Nt]，和为 1。
    """
    if regularization <= 0:
        raise ValueError(
            "OT regularization must be positive."
        )

    if num_iterations <= 0:
        raise ValueError(
            "num_iterations must be positive."
        )

    source_mass = source_mass.clamp_min(eps)
    target_mass = target_mass.clamp_min(eps)

    source_mass = source_mass / source_mass.sum()
    target_mass = target_mass / target_mass.sum()

    log_a = torch.log(source_mass)
    log_b = torch.log(target_mass)
    log_kernel = -cost / regularization

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(num_iterations):
        log_u = log_a - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0),
            dim=1,
        )

        log_v = log_b - torch.logsumexp(
            log_kernel.transpose(0, 1)
            + log_u.unsqueeze(0),
            dim=1,
        )

    log_plan = (
        log_u.unsqueeze(1)
        + log_kernel
        + log_v.unsqueeze(0)
    )

    plan = torch.exp(log_plan)

    # 数值误差下重新归一化总质量。
    return plan / plan.sum().clamp_min(eps)


class ReliableForegroundOTIBPLoss(nn.Module):
    """
    三阶段 IBP：

        Step 1:
            质量门控。教师预测正确且 margin 优于学生时，
            返回连续可靠性权重。

        Step 2:
            使用教师真实类别 CAM，选择 CAM > 样本均值的位置。

        Step 3:
            在教师 CAM 选中的位置，对教师与学生局部特征
            进行熵正则化最优传输对齐。

    最终：
        L_IBP = sum_i weight_i * OT_i / sum_i weight_i

    重要约定
    --------
    1. teacher_classes 顺序必须与教师局部分类器输出顺序一致。
       例如教师输出 3 类，对应全局类别 [5, 6, 7]。

    2. teacher_model 必须冻结，不应放入 optimizer。

    3. student_features 和 teacher_features 应为同一层输出：
           [B, C, H, W]
       或：
           [B, C, D, H, W]

    4. 学生与教师采用相同的 ResNet 架构，因此同一 stage 的
       特征通道数和空间尺寸必须完全一致，不使用任何可学习映射。
    """

    def __init__(
        self,
        teacher_classes: Sequence[int],
        tau_weight: float = 0.5,
        minimum_margin_advantage: float = 0.0,
        ot_regularization: float = 0.05,
        sinkhorn_iterations: int = 30,
        detach_transport_plan: bool = True,
        normalize_teacher_cam: bool = True,
        fallback_to_cam_argmax: bool = True,
        max_foreground_points: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if len(teacher_classes) < 2:
            raise ValueError(
                "teacher_classes must contain at least two classes."
            )

        if tau_weight <= 0:
            raise ValueError(
                "tau_weight must be positive."
            )

        if max_foreground_points is not None:
            if max_foreground_points <= 0:
                raise ValueError(
                    "max_foreground_points must be positive or None."
                )

        self.register_buffer(
            "teacher_classes",
            torch.as_tensor(
                teacher_classes,
                dtype=torch.long,
            ),
            persistent=True,
        )

        self.tau_weight = float(tau_weight)
        self.minimum_margin_advantage = float(
            minimum_margin_advantage
        )
        self.ot_regularization = float(ot_regularization)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.detach_transport_plan = bool(
            detach_transport_plan
        )
        self.normalize_teacher_cam = bool(
            normalize_teacher_cam
        )
        self.fallback_to_cam_argmax = bool(
            fallback_to_cam_argmax
        )
        self.max_foreground_points = max_foreground_points
        self.eps = float(eps)


    @torch.no_grad()
    def _compute_reliability_weights(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        tail_labels_global: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        返回：
            weights
            local_labels
            teacher_correct
            teacher_has_advantage
            teacher_margin
            student_margin
        """
        teacher_classes = self.teacher_classes.to(
            student_logits.device
        )

        local_labels = _map_global_to_teacher_local(
            global_labels=tail_labels_global,
            teacher_classes=teacher_classes,
        )

        num_teacher_classes = int(
            teacher_classes.numel()
        )

        # 教师可以输出局部尾类空间，也可以输出完整类别空间。
        if teacher_logits.shape[1] == num_teacher_classes:
            teacher_tail_logits = teacher_logits
        elif (
            teacher_logits.shape[1]
            > int(teacher_classes.max().item())
        ):
            teacher_tail_logits = teacher_logits.index_select(
                dim=1,
                index=teacher_classes,
            )
        else:
            raise ValueError(
                "teacher_logits cannot be aligned with teacher_classes."
            )

        if (
            student_logits.shape[1]
            <= int(teacher_classes.max().item())
        ):
            raise ValueError(
                "student_logits does not contain every teacher class."
            )

        student_tail_logits = student_logits.index_select(
            dim=1,
            index=teacher_classes,
        )

        teacher_prediction = teacher_tail_logits.argmax(dim=1)
        teacher_correct = teacher_prediction.eq(local_labels)

        teacher_margin = _classification_margin(
            logits=teacher_tail_logits,
            labels=local_labels,
        )

        student_margin = _classification_margin(
            logits=student_tail_logits,
            labels=local_labels,
        )

        margin_advantage = (
            teacher_margin
            - student_margin
            - self.minimum_margin_advantage
        )

        teacher_has_advantage = margin_advantage.gt(0)

        soft_advantage = torch.sigmoid(
            margin_advantage / self.tau_weight
        )

        weights = (
            teacher_correct.to(student_logits.dtype)
            * teacher_has_advantage.to(student_logits.dtype)
            * soft_advantage
        ).detach()

        return (
            weights,
            local_labels,
            teacher_correct,
            teacher_has_advantage,
            teacher_margin,
            student_margin,
        )

    @torch.no_grad()
    def _compute_teacher_cam(
        self,
        teacher_features: torch.Tensor,
        local_labels: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回展平 CAM：
            [B_tail, S]
        """
        if teacher_features.ndim < 3:
            raise ValueError(
                "teacher_features must be [B, C, *spatial]."
            )

        batch_size = teacher_features.shape[0]
        channel_dim = teacher_features.shape[1]
        spatial_size = int(
            teacher_features[0, 0].numel()
        )

        if classifier_weight.ndim != 2:
            raise ValueError(
                "classifier_weight must be [num_classes, C]."
            )

        if classifier_weight.shape[1] != channel_dim:
            raise ValueError(
                "Teacher classifier input dimension does not match "
                f"teacher feature channels: {classifier_weight.shape[1]} "
                f"vs {channel_dim}."
            )

        if classifier_weight.shape[0] <= int(
            local_labels.max().item()
        ):
            raise ValueError(
                "Teacher classifier does not contain every local label."
            )

        selected_weight = classifier_weight.index_select(
            dim=0,
            index=local_labels,
        )

        flat_features = teacher_features.reshape(
            batch_size,
            channel_dim,
            spatial_size,
        )

        cam = torch.einsum(
            "bc,bcs->bs",
            selected_weight,
            flat_features,
        )

        cam = F.relu(cam)

        if self.normalize_teacher_cam:
            cam = _normalize_cam(
                cam=cam,
                eps=self.eps,
            )

        return cam

    def _sample_ot_loss(
        self,
        student_flat: torch.Tensor,
        teacher_flat: torch.Tensor,
        cam_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, bool]:
        """
        单样本 OT 对齐。

        student_flat:
            [S, C_s]

        teacher_flat:
            [S, C_t]

        cam_flat:
            [S]
        """
        mean_activation = cam_flat.mean()
        selected_mask = cam_flat.gt(mean_activation)
        selected_indices = selected_mask.nonzero(
            as_tuple=False
        ).squeeze(1)

        # 极端情况下 CAM 可能为常数，严格 > mean 会选不到位置。
        # 为避免训练崩溃，可退化为 CAM 最大位置。
        if selected_indices.numel() == 0:
            if not self.fallback_to_cam_argmax:
                zero_loss = student_flat.sum() * 0.0
                return zero_loss, 0, False

            selected_indices = cam_flat.argmax().reshape(1)

        if (
            self.max_foreground_points is not None
            and selected_indices.numel()
            > self.max_foreground_points
        ):
            selected_values = cam_flat.index_select(
                dim=0,
                index=selected_indices,
            )

            top_relative = torch.topk(
                selected_values,
                k=self.max_foreground_points,
                largest=True,
                sorted=False,
            ).indices

            selected_indices = selected_indices.index_select(
                dim=0,
                index=top_relative,
            )

        student_selected = student_flat.index_select(
            dim=0,
            index=selected_indices,
        )

        teacher_selected = teacher_flat.index_select(
            dim=0,
            index=selected_indices,
        ).detach()

        student_normalized = F.normalize(
            student_selected,
            p=2,
            dim=1,
            eps=self.eps,
        )

        teacher_normalized = F.normalize(
            teacher_selected,
            p=2,
            dim=1,
            eps=self.eps,
        )

        # 余弦距离，范围通常位于 [0, 2]。
        cost = 1.0 - (
            student_normalized
            @ teacher_normalized.transpose(0, 1)
        )

        selected_cam = cam_flat.index_select(
            dim=0,
            index=selected_indices,
        ).detach()

        if selected_cam.sum().item() <= self.eps:
            mass = torch.full_like(
                selected_cam,
                fill_value=1.0 / selected_cam.numel(),
            )
        else:
            mass = selected_cam.clamp_min(self.eps)
            mass = mass / mass.sum()

        plan_cost = (
            cost.detach()
            if self.detach_transport_plan
            else cost
        )

        transport_plan = _log_sinkhorn_transport(
            cost=plan_cost,
            source_mass=mass,
            target_mass=mass,
            regularization=self.ot_regularization,
            num_iterations=self.sinkhorn_iterations,
            eps=self.eps,
        )

        if self.detach_transport_plan:
            transport_plan = transport_plan.detach()

        ot_loss = torch.sum(
            transport_plan * cost
        )

        return (
            ot_loss,
            int(selected_indices.numel()),
            True,
        )

    def forward(
        self,
        student_features: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_classifier_weight: torch.Tensor,
        teacher_cam_features: Optional[torch.Tensor] = None,
        tail_mask: Optional[torch.Tensor] = None,
        layer: int = 4,
    ) -> ReliableOTIBPOutput:
        """
        基于指定 ResNet stage 的学生/教师特征计算 IBP。

        teacher_features:
            用于 OT 对齐的指定层特征。

        teacher_cam_features:
            用于计算教师 CAM 的最终层特征。对 layer<4，CAM 会被插值
            到 teacher_features 的空间尺寸。若不传，则默认使用
            teacher_features，保持旧接口兼容。
        """
        layer = _validate_resnet_layer(layer)

        if student_features.ndim < 3:
            raise ValueError(
                "student_features must be [B, C, *spatial]."
            )

        if student_logits.ndim != 2:
            raise ValueError(
                "student_logits must be [B, C_all]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must be [B]."
            )

        batch_size = labels.shape[0]

        if student_features.shape[0] != batch_size:
            raise ValueError(
                "student_features and labels batch sizes do not match."
            )

        if student_logits.shape[0] != batch_size:
            raise ValueError(
                "student_logits and labels batch sizes do not match."
            )

        device = student_features.device
        dtype = student_features.dtype

        teacher_classes = self.teacher_classes.to(device)

        if tail_mask is None:
            tail_mask = torch.isin(
                labels,
                teacher_classes,
            )
        else:
            tail_mask = tail_mask.to(
                device=device,
                dtype=torch.bool,
            )

        num_tail = int(tail_mask.sum().item())

        full_weights = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )

        full_ot_losses = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )

        foreground_counts = torch.zeros(
            batch_size,
            device=device,
            dtype=torch.long,
        )

        valid_alignment = torch.zeros(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

        if num_tail == 0:
            empty_float = torch.empty(
                0,
                device=device,
                dtype=dtype,
            )
            empty_bool = torch.empty(
                0,
                device=device,
                dtype=torch.bool,
            )

            zero_loss = (
                student_features.sum()
                + student_logits.sum()
            ) * 0.0

            return ReliableOTIBPOutput(
                loss=zero_loss,
                reliability_weights=full_weights,
                per_sample_ot_loss=full_ot_losses,
                tail_mask=tail_mask,
                teacher_correct=empty_bool,
                teacher_has_advantage=empty_bool,
                teacher_margin=empty_float,
                student_margin=empty_float,
                foreground_counts=foreground_counts,
                valid_alignment=valid_alignment,
            )

        if teacher_features.ndim < 3:
            raise ValueError(
                "teacher_features must be [B_tail, C, *spatial]."
            )

        if teacher_features.shape[0] != num_tail:
            raise ValueError(
                "teacher_features batch size must equal the number "
                f"of tail samples: {teacher_features.shape[0]} vs {num_tail}."
            )

        if teacher_logits.shape[0] != num_tail:
            raise ValueError(
                "teacher_logits batch size must equal the number "
                f"of tail samples: {teacher_logits.shape[0]} vs {num_tail}."
            )

        student_dim = int(student_features.shape[1])
        teacher_dim = int(teacher_features.shape[1])

        if student_dim != teacher_dim:
            raise ValueError(
                f"Student and teacher must use the same architecture at "
                f"layer={layer}, but their channel dimensions differ: "
                f"{student_dim} vs {teacher_dim}."
            )

        student_spatial_shape = tuple(student_features.shape[2:])
        teacher_spatial_shape = tuple(teacher_features.shape[2:])
        if student_spatial_shape != teacher_spatial_shape:
            raise ValueError(
                f"Student and teacher must have the same spatial shape at "
                f"layer={layer}, but got {student_spatial_shape} vs "
                f"{teacher_spatial_shape}."
            )

        tail_labels = labels[tail_mask]
        student_tail_logits = student_logits[tail_mask]

        (
            tail_weights,
            local_labels,
            teacher_correct,
            teacher_has_advantage,
            teacher_margin,
            student_margin,
        ) = self._compute_reliability_weights(
            student_logits=student_tail_logits,
            teacher_logits=teacher_logits.detach(),
            tail_labels_global=tail_labels,
        )

        full_weights[tail_mask] = tail_weights.to(dtype)

        teacher_features = teacher_features.detach().to(
            device=device,
            dtype=dtype,
        )

        if teacher_cam_features is None:
            teacher_cam_features = teacher_features
        else:
            if teacher_cam_features.shape[0] != num_tail:
                raise ValueError(
                    "teacher_cam_features batch size must equal the number "
                    "of tail samples."
                )
            teacher_cam_features = teacher_cam_features.detach().to(
                device=device,
                dtype=dtype,
            )

        teacher_classifier_weight = (
            teacher_classifier_weight.detach().to(
                device=device,
                dtype=dtype,
            )
        )

        teacher_cam = self._compute_teacher_cam(
            teacher_features=teacher_cam_features,
            local_labels=local_labels,
            classifier_weight=teacher_classifier_weight,
        )

        cam_source_shape = tuple(
            teacher_cam_features.shape[2:]
        )
        teacher_spatial_shape = tuple(
            teacher_features.shape[2:]
        )

        if cam_source_shape != teacher_spatial_shape:
            teacher_cam = _resize_flat_cam(
                cam_flat=teacher_cam,
                source_spatial_shape=cam_source_shape,
                target_spatial_shape=teacher_spatial_shape,
            )
            if self.normalize_teacher_cam:
                teacher_cam = _normalize_cam(
                    cam=teacher_cam,
                    eps=self.eps,
                )

        student_tail_features = student_features[tail_mask]

        spatial_size = int(
            teacher_features[0, 0].numel()
        )

        student_flat = student_tail_features.reshape(
            num_tail,
            student_dim,
            spatial_size,
        ).transpose(1, 2)

        teacher_flat = teacher_features.reshape(
            num_tail,
            teacher_dim,
            spatial_size,
        ).transpose(1, 2)

        tail_indices_in_full_batch = tail_mask.nonzero(
            as_tuple=False
        ).squeeze(1)

        tail_ot_loss_list: List[torch.Tensor] = []
        tail_valid_list: List[bool] = []
        tail_count_list: List[int] = []

        for tail_index in range(num_tail):
            sample_ot_loss, count, is_valid = (
                self._sample_ot_loss(
                    student_flat=student_flat[tail_index],
                    teacher_flat=teacher_flat[tail_index],
                    cam_flat=teacher_cam[tail_index],
                )
            )

            tail_ot_loss_list.append(sample_ot_loss)
            tail_count_list.append(count)
            tail_valid_list.append(is_valid)

        tail_ot_losses = torch.stack(
            tail_ot_loss_list,
            dim=0,
        )

        tail_valid = torch.tensor(
            tail_valid_list,
            device=device,
            dtype=torch.bool,
        )

        tail_counts = torch.tensor(
            tail_count_list,
            device=device,
            dtype=torch.long,
        )

        full_ot_losses = full_ot_losses.scatter(
            dim=0,
            index=tail_indices_in_full_batch,
            src=tail_ot_losses,
        )

        foreground_counts[tail_mask] = tail_counts
        valid_alignment[tail_mask] = tail_valid

        effective_tail_weights = (
            tail_weights.to(dtype)
            * tail_valid.to(dtype)
        )

        denominator = effective_tail_weights.sum()

        if denominator.detach().item() <= self.eps:
            loss = (
                tail_ot_losses.sum()
                + student_logits.sum() * 0.0
            ) * 0.0
        else:
            loss = (
                effective_tail_weights
                * tail_ot_losses
            ).sum() / denominator.clamp_min(self.eps)

        return ReliableOTIBPOutput(
            loss=loss,
            reliability_weights=full_weights,
            per_sample_ot_loss=full_ot_losses,
            tail_mask=tail_mask,
            teacher_correct=teacher_correct,
            teacher_has_advantage=teacher_has_advantage,
            teacher_margin=teacher_margin,
            student_margin=student_margin,
            foreground_counts=foreground_counts,
            valid_alignment=valid_alignment,
        )

    def forward_from_models(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        tail_mask: Optional[torch.Tensor] = None,
        layer: int = 4,
        student_logits: Optional[torch.Tensor] = None,
        teacher_classifier_weight: Optional[torch.Tensor] = None,
    ) -> ReliableOTIBPOutput:
        """
        直接从 timm/torchvision 风格 ResNet 内部提取中间层特征。

        layer:
            1, 2, 3, 4 分别对应 model.layer1~model.layer4。

        说明：
            - 不再需要从训练循环传入 student_features。
            - 若已计算 student_logits，可直接传入以复用分类输出。
            - 教师 CAM 始终由 layer4 特征和最终分类器权重生成，随后
              插值到所选 layer 的空间分辨率，再指导该层 OT 对齐。
        """
        layer = _validate_resnet_layer(layer)

        if labels.ndim != 1:
            raise ValueError("labels must have shape [B].")

        teacher_classes = self.teacher_classes.to(labels.device)
        if tail_mask is None:
            tail_mask = torch.isin(labels, teacher_classes)
        else:
            tail_mask = tail_mask.to(
                device=labels.device,
                dtype=torch.bool,
            )

        # 如果已经传入学生 logits，只前向到所选层即可；否则继续到
        # layer4，并使用模型原有 forward_head 计算 logits。
        need_student_final = student_logits is None
        student_features, student_final_features = (
            _forward_timm_resnet_stage(
                model=student_model,
                inputs=inputs,
                layer=layer,
                return_final=need_student_final,
            )
        )

        if student_logits is None:
            if student_final_features is None:
                raise RuntimeError(
                    "student_final_features is unexpectedly None."
                )
            student_logits = _forward_model_head(
                model=student_model,
                final_features=student_final_features,
            )
        else:
            student_logits = _unwrap_tensor_output(
                student_logits,
                "student_logits",
            )

        num_tail = int(tail_mask.sum().item())
        if num_tail == 0:
            empty_teacher_features = student_features.new_empty(
                (0, 1, *student_features.shape[2:])
            )
            empty_teacher_logits = student_logits.new_empty(
                (0, int(self.teacher_classes.numel()))
            )
            empty_classifier_weight = student_logits.new_empty(
                (int(self.teacher_classes.numel()), 1)
            )
            return self.forward(
                student_features=student_features,
                student_logits=student_logits,
                teacher_features=empty_teacher_features,
                teacher_logits=empty_teacher_logits,
                labels=labels,
                teacher_classifier_weight=empty_classifier_weight,
                teacher_cam_features=empty_teacher_features,
                tail_mask=tail_mask,
                layer=layer,
            )

        teacher_model.eval()
        with torch.no_grad():
            teacher_features, teacher_final_features = (
                _forward_timm_resnet_stage(
                    model=teacher_model,
                    inputs=inputs[tail_mask],
                    layer=layer,
                    return_final=True,
                )
            )

            if teacher_final_features is None:
                raise RuntimeError(
                    "teacher_final_features is unexpectedly None."
                )

            teacher_logits = _forward_model_head(
                model=teacher_model,
                final_features=teacher_final_features,
            )

        if teacher_classifier_weight is None:
            teacher_classifier_weight = get_classifier_weight(
                teacher_model
            )

        return self.forward(
            student_features=student_features,
            student_logits=student_logits,
            teacher_features=teacher_features,
            teacher_logits=teacher_logits,
            labels=labels,
            teacher_classifier_weight=teacher_classifier_weight,
            teacher_cam_features=teacher_final_features,
            tail_mask=tail_mask,
            layer=layer,
        )


# -------------------------------------------------------------------------
# 一个最小可运行示例
# -------------------------------------------------------------------------


class _DemoResNet(nn.Module):
    """仅用于测试 layer=1/2/3/4 接口的 ResNet 风格网络。"""

    def __init__(
        self,
        stage_dims: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        if len(stage_dims) != 4:
            raise ValueError("stage_dims must contain four values.")

        c1, c2, c3, c4 = [int(v) for v in stage_dims]

        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_logits: bool = False,
    ) -> torch.Tensor:
        pooled = self.global_pool(x).flatten(1)
        return pooled if pre_logits else self.fc(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(
            self.forward_features(x),
            pre_logits=False,
        )

    def get_classifier(self) -> nn.Module:
        return self.fc


def demo() -> None:
    torch.manual_seed(11)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    teacher_classes = [5, 6, 7]
    stage_dims = [16, 24, 32, 48]

    student_model = _DemoResNet(
        stage_dims=stage_dims,
        num_classes=8,
    ).to(device)

    teacher_model = _DemoResNet(
        stage_dims=stage_dims,
        num_classes=3,
    ).to(device)

    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    ibp_loss_module = ReliableForegroundOTIBPLoss(
        teacher_classes=teacher_classes,
        tau_weight=0.5,
        minimum_margin_advantage=-100.0,
        ot_regularization=0.05,
        sinkhorn_iterations=10,
        detach_transport_plan=True,
        normalize_teacher_cam=True,
        fallback_to_cam_argmax=True,
        max_foreground_points=32,
    ).to(device)

    assert sum(p.numel() for p in ibp_loss_module.parameters()) == 0

    # IBP 模块没有可学习参数，优化器只更新学生模型。
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=1e-3,
    )

    inputs = torch.randn(10, 3, 32, 32, device=device)
    labels = torch.tensor(
        [0, 5, 6, 1, 7, 5, 3, 6, 2, 7],
        device=device,
        dtype=torch.long,
    )
    tail_mask = torch.isin(
        labels,
        torch.tensor(teacher_classes, device=device),
    )

    # 训练循环中已有的学生 logits 可以直接复用。
    student_logits = student_model(inputs)
    loss_cls = F.cross_entropy(student_logits, labels)

    ibp_output = ibp_loss_module.forward_from_models(
        student_model=student_model,
        teacher_model=teacher_model,
        inputs=inputs,
        labels=labels,
        tail_mask=tail_mask,
        layer=3,
        student_logits=student_logits,
    )

    total_loss = loss_cls + 0.5 * ibp_output.loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    print("Demo finished successfully.")
    print("Selected layer: 3")
    print("IBP statistics:", ibp_output.statistics())
    print("Classification loss:", float(loss_cls.detach()))
    print("IBP loss:", float(ibp_output.loss.detach()))
    print("Total loss:", float(total_loss.detach()))


if __name__ == "__main__":
    demo()


