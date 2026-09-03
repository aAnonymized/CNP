from typing import Sequence, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# class DynamicDecodableInfoMaxLoss(nn.Module):
#     """
#     频率权重 + 动态可解码权重。

#     对类别 c：
#         frequency_weight_c = n_max / n_c

#         r_c^(t) = momentum * r_c^(t-1)
#                   + (1 - momentum) * E_{y=c}[q(c | z)]

#         dynamic_weight_c = 1 - r_c^(t)

#         final_weight_c = frequency_weight_c * dynamic_weight_c

#     最终损失：
#         loss = sum_i w_{y_i} * CE_i / sum_i w_{y_i}

#     除 EMA momentum 外，不引入额外调节参数。
#     """

#     def __init__(
#         self,
#         cls_num_list: Union[Sequence[int], torch.Tensor],
#         power=0.5, 
#         momentum: float = 0.90,
#         eps: float = 1e-12,
#     ) -> None:
#         super().__init__()

#         if not 0.0 <= momentum < 1.0:
#             raise ValueError("momentum must satisfy 0 <= momentum < 1.")

#         class_counts = torch.as_tensor(
#             cls_num_list,
#             dtype=torch.float32,
#         )

#         if class_counts.ndim != 1:
#             raise ValueError("cls_num_list must be one-dimensional.")
#         if torch.any(class_counts <= 0):
#             raise ValueError("Every class count must be positive.")

#         self.num_classes = int(class_counts.numel())
#         self.power = power
#         self.momentum = float(momentum)
#         self.eps = float(eps)
#         class_prior = (class_counts / class_counts.sum()).clamp_min(self.eps)
#         prior_log = class_prior.log()
#         print(f'@@@@@@@@@@@ prior_log: {prior_log}')
#         self.register_buffer("prior_log", prior_log)
#         self.register_buffer("class_confidence", torch.zeros(self.num_classes, dtype=torch.float32))
#         self.register_buffer("class_seen", torch.zeros(self.num_classes, dtype=torch.bool))

#     @torch.no_grad()
#     def update_class_confidence(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> None:
#         probabilities = F.softmax(logits.detach(), dim=1)
#         true_probability = probabilities.gather(
#             dim=1,
#             index=labels.unsqueeze(1),
#         ).squeeze(1)

#         for class_tensor in labels.unique():
#             class_id = int(class_tensor.item())
#             class_mask = labels == class_id
#             batch_confidence = true_probability[class_mask].mean()

#             if not bool(self.class_seen[class_id]):
#                 self.class_confidence[class_id] = batch_confidence
#                 self.class_seen[class_id] = True
#             else:
#                 self.class_confidence[class_id] = (
#                     self.momentum * self.class_confidence[class_id]
#                     + (1.0 - self.momentum) * batch_confidence
#                 )

#     @torch.no_grad()
#     def get_class_weights(self):
#         dynamic_gap = (1.0 - self.class_confidence).clamp_min(self.eps).pow(self.power)
#         normalized_gap = (
#             dynamic_gap
#             / dynamic_gap.max().clamp_min(
#                 self.eps
#             )
#         )
#         dynamic_weights = (0.5 + 1.5 * normalized_gap)
#         logit_adjustment = dynamic_weights * self.prior_log
#         return logit_adjustment

#     def forward(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         loss_function=None, 
#         update_ema: bool = True,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if logits.ndim != 2:
#             raise ValueError("logits must have shape [B, C].")
#         if labels.ndim != 1:
#             raise ValueError("labels must have shape [B].")
#         if logits.shape[0] != labels.shape[0]:
#             raise ValueError("Batch sizes of logits and labels differ.")
#         if logits.shape[1] != self.num_classes:
#             raise ValueError(
#                 "The class dimension of logits must equal len(cls_num_list)."
#             )

#         if update_ema:
#             self.update_class_confidence(logits, labels)

#         logit_adjustment = self.get_class_weights()
#         adjusted_logits = (logits + logit_adjustment.unsqueeze(0))
#         loss = loss_function(adjusted_logits, labels)
#         return loss, logit_adjustment.detach().clone(), self.class_confidence.detach().clone()

#     @torch.no_grad()
#     def reset_state(self) -> None:
#         self.class_confidence.zero_()
#         self.class_seen.zero_()


from typing import Sequence, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDecodableInfoMaxLoss(nn.Module):
    """
    Dynamic Decodable Information Maximization Loss

    ------------------------------------------------------------
    核心变化：
    ------------------------------------------------------------
    1. class_confidence 不再使用 training batch 更新。

    2. 每个 epoch 训练结束以后：
           model.eval()
           with torch.no_grad():
               遍历 reference / anchor pool

       统计每个类别在整个 reference set 上的：

           mean confidence:
               E[p(y=c | x) | y=c]

           class accuracy:
               Acc_c

    3. 对每个类别的 reference confidence 每个 epoch 只做一次 EMA：

           r_c^(t)
               = momentum * r_c^(t-1)
               + (1 - momentum) * mean_confidence_c^(t)

    4. 训练阶段 forward() 只读取当前 class_confidence，
       默认不会修改它。

    ------------------------------------------------------------
    Dynamic logit adjustment
    ------------------------------------------------------------

        prior_c = n_c / sum_j n_j

        dynamic_gap_c = (1 - r_c)^power

        dynamic_weight_c
            = 0.5 + 1.5 * normalized_gap_c

        logit_adjustment_c
            = dynamic_weight_c * log(prior_c)

        adjusted_logits
            = logits + logit_adjustment

        loss
            = CE(adjusted_logits, labels)

    ------------------------------------------------------------
    注意：
    ------------------------------------------------------------
    datasets['val'] 如果用于更新 class_confidence / Anchor Bank，
    它在方法上应称为：

        Anchor Pool / Reference Set

    而不再是传统意义上完全独立的 validation set。
    """

    def __init__(
        self,
        cls_num_list: Union[Sequence[int], torch.Tensor],
        power: float = 0.5,
        momentum: float = 0.90,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1."
            )

        if power < 0:
            raise ValueError(
                "power must be non-negative."
            )

        class_counts = torch.as_tensor(
            cls_num_list,
            dtype=torch.float32,
        )

        if class_counts.ndim != 1:
            raise ValueError(
                "cls_num_list must be one-dimensional."
            )

        if torch.any(class_counts <= 0):
            raise ValueError(
                "Every class count must be positive."
            )

        self.num_classes = int(
            class_counts.numel()
        )

        self.power = float(power)
        self.momentum = float(momentum)
        self.eps = float(eps)

        # ============================================================
        # Long-tailed class prior
        # ============================================================
        class_prior = (
            class_counts
            /
            class_counts.sum()
        ).clamp_min(
            self.eps
        )
        prior_log = class_prior.log()
        print(
            f'@@@@@@@@@@@ prior_log: {prior_log}'
        )
        self.register_buffer(
            "prior_log",
            prior_log,
        )

        # ============================================================
        # Dynamic class confidence
        #
        # 注意：
        # 现在只由 Anchor Pool / Reference Set 更新
        # ============================================================
        self.register_buffer(
            "class_confidence",
            torch.zeros(
                self.num_classes,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "class_seen",
            torch.zeros(
                self.num_classes,
                dtype=torch.bool,
            ),
        )

        # ============================================================
        # 仅用于日志 / 分析
        # 不参与 gradient
        # ============================================================
        self.register_buffer(
            "class_accuracy",
            torch.zeros(
                self.num_classes,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "class_reference_count",
            torch.zeros(
                self.num_classes,
                dtype=torch.long,
            ),
        )

    # ================================================================
    # Helper
    # ================================================================
    @staticmethod
    def _unwrap_logits(
        output,
    ) -> torch.Tensor:
        """
        兼容：
            model(x) -> logits

        以及：
            model(x) -> (logits, ...)
        """

        if torch.is_tensor(output):
            return output

        if isinstance(
            output,
            (tuple, list),
        ):
            if len(output) == 0:
                raise ValueError(
                    "Model returned an empty tuple/list."
                )

            if not torch.is_tensor(
                output[0]
            ):
                raise TypeError(
                    "The first model output must be logits Tensor."
                )

            return output[0]

        raise TypeError(
            "Model output must be Tensor or tuple/list."
        )

    # ================================================================
    # Update class confidence
    # ================================================================
    @torch.no_grad()
    def update_class_confidence(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        保留原函数名。

        这个函数现在的推荐用途是：

            把整个 Anchor Pool / Reference Set
            的 logits 和 labels 汇总以后，
            一次性调用。

        不建议在 training batch 中调用。
        """

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [N, C]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [N]."
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "Batch sizes of logits and labels differ."
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "The class dimension of logits must "
                "equal len(cls_num_list)."
            )

        device = self.class_confidence.device

        logits = logits.to(device)
        labels = labels.to(device)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        true_probability = probabilities.gather(
            dim=1,
            index=labels.unsqueeze(1),
        ).squeeze(1)

        prediction = logits.argmax(
            dim=1
        )

        # ============================================================
        # 每个类别只更新一次
        # ============================================================
        for class_id in range(
            self.num_classes
        ):

            class_mask = (
                labels == class_id
            )

            num_samples = int(
                class_mask.sum().item()
            )

            # Reference Set 中这个类别不存在
            # 保留旧 confidence，不更新
            if num_samples == 0:
                continue

            # --------------------------------------------------------
            # 当前 epoch Reference Set 上的类别可靠程度
            # --------------------------------------------------------
            epoch_confidence = (
                true_probability[
                    class_mask
                ].mean()
            )

            # --------------------------------------------------------
            # 类别准确率，仅用于观察
            # --------------------------------------------------------
            epoch_accuracy = (
                prediction[
                    class_mask
                ]
                .eq(
                    labels[
                        class_mask
                    ]
                )
                .float()
                .mean()
            )

            self.class_accuracy[
                class_id
            ] = epoch_accuracy

            self.class_reference_count[
                class_id
            ] = num_samples

            # --------------------------------------------------------
            # 第一次见到该类别：
            # 不做 EMA，直接初始化
            # --------------------------------------------------------
            if not bool(
                self.class_seen[
                    class_id
                ].item()
            ):

                self.class_confidence[
                    class_id
                ] = epoch_confidence

                self.class_seen[
                    class_id
                ] = True

            # --------------------------------------------------------
            # 后续 epoch：
            # 每轮只做一次 EMA
            # --------------------------------------------------------
            else:

                self.class_confidence[
                    class_id
                ] = (
                    self.momentum
                    *
                    self.class_confidence[
                        class_id
                    ]
                    +
                    (
                        1.0
                        -
                        self.momentum
                    )
                    *
                    epoch_confidence
                )

    # ================================================================
    # NEW:
    # Directly update confidence from Anchor Pool / Reference Set
    # ================================================================
    @torch.no_grad()
    def update_class_confidence_from_loader(
        self,
        model: nn.Module,
        reference_loader,
        device,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        每个 epoch 训练结束后调用。

        ------------------------------------------------------------
        功能：
        ------------------------------------------------------------
        1. model.eval()
        2. torch.no_grad()
        3. 遍历整个 Anchor Pool / Reference Set
        4. 汇总全部 logits / labels
        5. 每个类别计算：
               mean true-class probability
               class accuracy
        6. 每类别只做一次 EMA
        7. 恢复模型原来的 train/eval 状态

        reference_loader 支持：
            (inputs, labels)

        或：
            (inputs, labels, index)

        返回：
            statistics dict
        """

        was_training = model.training

        model.eval()

        # ============================================================
        # 不需要把所有 logits 保存在 GPU
        # 每类直接累计 sum / count
        # ============================================================
        confidence_sum = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.float64,
        )

        correct_sum = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.float64,
        )

        class_count = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.long,
        )

        total_correct = 0
        total_samples = 0

        with torch.no_grad():

            for batch in reference_loader:

                # ----------------------------------------------------
                # 兼容：
                # (image, label)
                # (image, label, index)
                # ----------------------------------------------------
                if not isinstance(
                    batch,
                    (tuple, list),
                ):
                    raise TypeError(
                        "reference_loader must return "
                        "(inputs, labels) or "
                        "(inputs, labels, index)."
                    )

                if len(batch) < 2:
                    raise ValueError(
                        "reference_loader batch must "
                        "contain at least inputs and labels."
                    )

                inputs = batch[0]
                labels = batch[1]

                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                ).long()

                output = model(
                    inputs
                )

                logits = self._unwrap_logits(
                    output
                )

                probabilities = F.softmax(
                    logits,
                    dim=1,
                )

                true_probability = probabilities.gather(
                    dim=1,
                    index=labels.unsqueeze(1),
                ).squeeze(1)

                prediction = logits.argmax(
                    dim=1
                )

                correct = prediction.eq(
                    labels
                )

                # ====================================================
                # 按类别累计
                # ====================================================
                for class_id in labels.unique():

                    c = int(
                        class_id.item()
                    )

                    class_mask = (
                        labels == c
                    )

                    confidence_sum[
                        c
                    ] += (
                        true_probability[
                            class_mask
                        ]
                        .double()
                        .sum()
                    )

                    correct_sum[
                        c
                    ] += (
                        correct[
                            class_mask
                        ]
                        .double()
                        .sum()
                    )

                    class_count[
                        c
                    ] += class_mask.sum()

                total_correct += int(
                    correct.sum().item()
                )

                total_samples += int(
                    labels.numel()
                )

        # ============================================================
        # 整个 Reference Set 跑完以后
        # 才更新 EMA
        # ============================================================
        epoch_class_confidence = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.float32,
        )

        epoch_class_accuracy = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.float32,
        )

        valid_class_mask = (
            class_count > 0
        )

        epoch_class_confidence[
            valid_class_mask
        ] = (
            confidence_sum[
                valid_class_mask
            ]
            /
            class_count[
                valid_class_mask
            ].double()
        ).float()

        epoch_class_accuracy[
            valid_class_mask
        ] = (
            correct_sum[
                valid_class_mask
            ]
            /
            class_count[
                valid_class_mask
            ].double()
        ).float()

        # ============================================================
        # EMA：每个类别一个 epoch 只更新一次
        # ============================================================
        for class_id in range(
            self.num_classes
        ):

            if not bool(
                valid_class_mask[
                    class_id
                ].item()
            ):
                continue

            current_confidence = (
                epoch_class_confidence[
                    class_id
                ]
            )

            self.class_accuracy[
                class_id
            ] = (
                epoch_class_accuracy[
                    class_id
                ]
            ).to(
                self.class_accuracy.device
            )

            self.class_reference_count[
                class_id
            ] = (
                class_count[
                    class_id
                ]
            ).to(
                self.class_reference_count.device
            )

            current_confidence = (
                current_confidence.to(
                    self.class_confidence.device
                )
            )

            # 第一次初始化
            if not bool(
                self.class_seen[
                    class_id
                ].item()
            ):

                self.class_confidence[
                    class_id
                ] = current_confidence

                self.class_seen[
                    class_id
                ] = True

            else:

                self.class_confidence[
                    class_id
                ] = (
                    self.momentum
                    *
                    self.class_confidence[
                        class_id
                    ]
                    +
                    (
                        1.0
                        -
                        self.momentum
                    )
                    *
                    current_confidence
                )

        # ============================================================
        # 恢复 model 状态
        # ============================================================
        if was_training:
            model.train()

        overall_accuracy = (
            float(
                total_correct
                /
                max(
                    total_samples,
                    1,
                )
            )
        )

        valid_acc = (
            epoch_class_accuracy[
                valid_class_mask
            ]
        )

        if valid_acc.numel() > 0:
            balanced_accuracy = float(
                valid_acc.mean().item()
            )
        else:
            balanced_accuracy = 0.0

        return {
            # 当前 epoch reference set 原始统计
            "epoch_class_confidence":
                epoch_class_confidence
                .detach()
                .cpu(),

            "epoch_class_accuracy":
                epoch_class_accuracy
                .detach()
                .cpu(),

            "class_count":
                class_count
                .detach()
                .cpu(),

            # EMA 后真正用于下一轮训练的 confidence
            "ema_class_confidence":
                self.class_confidence
                .detach()
                .cpu()
                .clone(),

            "overall_accuracy":
                overall_accuracy,

            "balanced_accuracy":
                balanced_accuracy,
        }

    # ================================================================
    # Dynamic class weights
    # ================================================================
    @torch.no_grad()
    def get_class_weights(
        self,
    ) -> torch.Tensor:

        # ============================================================
        # 如果某类别还从未在 Reference Set 中被看到：
        # confidence=0
        #
        # 即视为当前还不可靠 / 难解码
        # ============================================================
        dynamic_gap = (
            1.0
            -
            self.class_confidence
        ).clamp_min(
            self.eps
        ).pow(
            self.power
        )

        # normalized_gap = (
        #     dynamic_gap
        #     /
        #     dynamic_gap.max().clamp_min(
        #         self.eps
        #     )
        # )
        
        dynamic_weights = (0.1 + 2.0*dynamic_gap)
        logit_adjustment = (
            dynamic_weights
            *
            self.prior_log
        )

        return logit_adjustment

    # ================================================================
    # Forward
    # ================================================================
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_function: None, 
        update_ema: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        训练阶段直接：

            train_loss, class_weights, class_confidence = \
                bdec_loss_fn(
                    logits=outputs,
                    labels=labels,
                )

        默认：
            update_ema=False

        因此 training batch 不会再修改 class_confidence。

        ------------------------------------------------------------
        为兼容旧代码：
        如果显式写：
            update_ema=True

        仍然可以调用原 update_class_confidence()。

        但现在不推荐在训练阶段这么做。
        """

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [B, C]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [B]."
            )

        if (
            logits.shape[0]
            !=
            labels.shape[0]
        ):
            raise ValueError(
                "Batch sizes of logits and labels differ."
            )

        if (
            logits.shape[1]
            !=
            self.num_classes
        ):
            raise ValueError(
                "The class dimension of logits must "
                "equal len(cls_num_list)."
            )

        # ============================================================
        # 默认关闭！
        #
        # 现在 confidence 应由 Reference Set 更新
        # ============================================================
        if update_ema:
            self.update_class_confidence(
                logits,
                labels,
            )

        logit_adjustment = (
            self.get_class_weights()
        )

        adjusted_logits = (
            logits
            +
            logit_adjustment.unsqueeze(
                0
            )
        )

        loss = loss_function(adjusted_logits,labels)

        return (
            loss,
            logit_adjustment
            .detach()
            .clone(),

            self.class_confidence
            .detach()
            .clone(),
        )

    # ================================================================
    # Reset
    # ================================================================
    @torch.no_grad()
    def reset_state(
        self,
    ) -> None:

        self.class_confidence.zero_()
        self.class_seen.zero_()
        self.class_accuracy.zero_()
        self.class_reference_count.zero_()