from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


TensorOrNone = Optional[torch.Tensor]


def _project_ap_gradient(
    task_grads: List[TensorOrNone],
    ap_grads: List[TensorOrNone],
    eps: float = 1e-12,
) -> tuple[List[TensorOrNone], Dict[str, float]]:
    """
    当 AP 梯度与任务梯度冲突时，投影 AP 梯度。

    task_grads:
        主任务损失对 model 参数的梯度。

    ap_grads:
        AP 损失对 model 参数的梯度。
        当 AP 中使用 GRL 时，这里获得的已经是反转后的实际梯度。

    返回:
        projected_ap_grads:
            经过任务梯度保护的 AP 梯度。

        statistics:
            梯度冲突、余弦相似度等统计信息。
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

    dot_product = torch.zeros(
        (),
        device=device,
        dtype=dtype,
    )
    task_norm_sq = torch.zeros_like(dot_product)
    ap_norm_sq = torch.zeros_like(dot_product)

    has_shared_gradient = False

    # 只在 task 和 AP 都作用到的参数子空间内判断冲突
    for g_task, g_ap in zip(task_grads, ap_grads):
        if g_task is None or g_ap is None:
            continue

        has_shared_gradient = True

        dot_product = dot_product + torch.sum(
            g_task * g_ap
        )
        task_norm_sq = task_norm_sq + torch.sum(
            g_task * g_task
        )
        ap_norm_sq = ap_norm_sq + torch.sum(
            g_ap * g_ap
        )

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

    # dot >= 0 表示 AP 梯度没有反对主任务梯度
    if dot_product.detach().item() >= 0:
        return ap_grads, {
            "gradient_conflict": 0.0,
            "gradient_cosine": cosine.detach().item(),
            "task_grad_norm": torch.sqrt(
                task_norm_sq
            ).detach().item(),
            "ap_grad_norm": torch.sqrt(
                ap_norm_sq
            ).detach().item(),
        }

    # 移除 AP 梯度中与任务梯度方向相反的分量
    projection_coefficient = dot_product / (
        task_norm_sq + eps
    )

    projected_ap_grads: List[TensorOrNone] = []

    for g_task, g_ap in zip(task_grads, ap_grads):
        if g_ap is None:
            projected_ap_grads.append(None)

        elif g_task is None:
            # AP 独有参数方向，不涉及任务梯度冲突
            projected_ap_grads.append(g_ap)

        else:
            projected_ap_grads.append(
                g_ap
                - projection_coefficient * g_task
            )

    return projected_ap_grads, {
        "gradient_conflict": 1.0,
        "gradient_cosine": cosine.detach().item(),
        "task_grad_norm": torch.sqrt(
            task_norm_sq
        ).detach().item(),
        "ap_grad_norm": torch.sqrt(
            ap_norm_sq
        ).detach().item(),
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

    该函数内部完成：
        1. optimizer.zero_grad()
        2. 计算主任务梯度
        3. 计算 AP 对 model 和 ap_model 的梯度
        4. 对 model 的 AP 梯度进行冲突投影
        5. 将梯度写回 param.grad
        6. 可选梯度裁剪
        7. optimizer.step()

    参数
    ----
    model:
        学生分类模型。梯度保护仅作用于该模型。

    ap_model:
        AP 二分类判别器。其梯度正常更新，不参与投影。

    optimizer:
        可以同时包含 model 和 ap_model 参数的同一个优化器。

    loss_task:
        用于保护的任务损失。推荐：
            loss_cls
        或：
            loss_cls + beta * loss_ibp

    loss_ap:
        AP 对抗损失。没有跨组误分类样本时传 None。

    lambda_ap:
        AP 梯度权重。

    max_grad_norm:
        可选梯度裁剪阈值。

    注意
    ----
    建议 GRL 的 alpha 设置为 1.0，再通过 lambda_ap
    统一控制 AP 强度，避免重复缩放。
    """
    if not loss_task.requires_grad:
        raise ValueError(
            "loss_task must require gradients."
        )

    if lambda_ap < 0:
        raise ValueError(
            "lambda_ap must be non-negative."
        )

    model_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    ap_params = [
        p for p in ap_model.parameters()
        if p.requires_grad
    ]

    # 防止同一个参数同时出现在两个模块中
    model_param_ids = {
        id(p) for p in model_params
    }
    overlapping_params = [
        p for p in ap_params
        if id(p) in model_param_ids
    ]

    if overlapping_params:
        raise ValueError(
            "model and ap_model contain overlapping parameters."
        )

    # 同一个 optimizer，在这里统一清空梯度
    optimizer.zero_grad(set_to_none=True)

    use_ap = (
        loss_ap is not None
        and loss_ap.requires_grad
    )

    # 主任务梯度只作用于学生模型
    task_grads = torch.autograd.grad(
        outputs=loss_task,
        inputs=model_params,
        retain_graph=use_ap,
        create_graph=False,
        allow_unused=True,
    )

    if use_ap:
        # 一次性得到：
        # 1. AP 对学生模型的反转梯度
        # 2. AP 对判别器的正常梯度
        all_ap_params = model_params + ap_params

        all_ap_grads = torch.autograd.grad(
            outputs=loss_ap,
            inputs=all_ap_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        num_model_params = len(model_params)

        ap_model_grads = list(
            all_ap_grads[:num_model_params]
        )
        ap_discriminator_grads = list(
            all_ap_grads[num_model_params:]
        )

        # 只保护学生模型
        protected_ap_grads, statistics = (
            _project_ap_gradient(
                task_grads=list(task_grads),
                ap_grads=ap_model_grads,
                eps=eps,
            )
        )
    else:
        protected_ap_grads = [
            None for _ in model_params
        ]
        ap_discriminator_grads = [
            None for _ in ap_params
        ]

        statistics = {
            "gradient_conflict": 0.0,
            "gradient_cosine": 0.0,
            "task_grad_norm": 0.0,
            "ap_grad_norm": 0.0,
        }

    # --------------------------------------------------
    # 将主任务梯度 + 受保护 AP 梯度写入学生模型
    # --------------------------------------------------
    for param, g_task, g_ap in zip(
        model_params,
        task_grads,
        protected_ap_grads,
    ):
        total_grad = None

        if g_task is not None:
            total_grad = g_task.detach().clone()

        if g_ap is not None:
            weighted_ap_grad = (
                lambda_ap
                * g_ap.detach()
            )

            if total_grad is None:
                total_grad = weighted_ap_grad.clone()
            else:
                total_grad.add_(weighted_ap_grad)

        param.grad = total_grad

    # --------------------------------------------------
    # AP 判别器不参与保护，正常最小化 AP 损失
    # --------------------------------------------------
    for param, g_ap_disc in zip(
        ap_params,
        ap_discriminator_grads,
    ):
        if g_ap_disc is None:
            param.grad = None
        else:
            param.grad = (
                lambda_ap
                * g_ap_disc.detach().clone()
            )

    # 可选梯度裁剪
    if max_grad_norm is not None:
        all_trainable_params = (
            model_params + ap_params
        )

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            all_trainable_params,
            max_norm=max_grad_norm,
        )

        statistics["total_grad_norm"] = (
            float(total_grad_norm)
        )

    # 同一个优化器只调用一次
    optimizer.step()

    statistics["ap_enabled"] = float(use_ap)
    statistics["loss_task"] = (
        loss_task.detach().item()
    )
    statistics["loss_ap"] = (
        loss_ap.detach().item()
        if use_ap
        else 0.0
    )

    return statistics


'''
Usaging:
    stats = gradient_project(
        model=model,
        ap_model=ap_model,
        optimizer=optimizer,
        loss_task=loss_task,
        loss_ap=loss_ap,
        lambda_ap=lambda_ap,
    )
    
并且不需要下面这三个，因为在里面更新了.
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
'''