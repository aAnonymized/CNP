import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_to_standard_normal(mu, logvar):
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) )
    return: [B_tail]
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1)


class SubTaskTeacherDistillLoss(nn.Module):
    """
    子任务教师蒸馏损失。

    输入:
        img_feat:  全任务学生特征, [B, D]
        t_feat:    tail-only teacher 特征, [B_tail, D] 或 [B, D]
        labels:    全局标签, [B]
        tail_mask: tail 样本 mask, [B]

    损失:
        L = lambda_rate  * KL(q(t_hat|t_feat) || N(0,I))
          + lambda_cls   * CE(teacher_fc(t_hat), tail_label)
          + lambda_align * d(img_feat_tail, sg(t_hat))
    """

    def __init__(
        self,
        feat_dim,
        tail_classes,
        teacher_fc,
        lambda_rate=1e-4,
        lambda_cls=1.0,
        lambda_align=1.0,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.lambda_rate = lambda_rate
        self.lambda_cls = lambda_cls
        self.lambda_align = lambda_align

        # tail_classes: e.g. [5, 6, 7]
        tail_classes = torch.tensor(tail_classes, dtype=torch.long)
        self.register_buffer("tail_classes", tail_classes, persistent=False)

        # 教师分类头，建议冻结
        self.teacher_fc = teacher_fc
        for p in self.teacher_fc.parameters():
            p.requires_grad = False

        # 极简压缩器：teacher feature -> mu, logvar
        self.to_mu = nn.Linear(feat_dim, feat_dim)
        self.to_logvar = nn.Linear(feat_dim, feat_dim)

        # 残差尺度，避免一开始破坏 teacher 特征空间
        self.scale = nn.Parameter(torch.tensor(0.1))

    def map_to_tail_label(self, labels_tail):
        """
        全局标签 -> tail 子任务标签

        例如:
            tail_classes = [5, 6, 7]
            labels_tail  = [5, 7, 6]
            return       = [0, 2, 1]
        """
        tail_classes = self.tail_classes.to(labels_tail.device)
        match = labels_tail[:, None].eq(tail_classes[None, :])

        if not match.any(dim=1).all():
            bad = labels_tail[~match.any(dim=1)].detach().cpu().tolist()
            raise ValueError(f"labels not in tail_classes: {bad}")

        return match.float().argmax(dim=1).long()

    def compress_teacher_feat(self, t_feat):
        """
        t_feat: [B_tail, D]

        使用残差压缩:
            mu = t_feat + alpha * Linear(t_feat)

        这样比完全重新生成 t_hat 更稳定。
        """
        delta = self.to_mu(t_feat)
        mu = t_feat + self.scale * delta

        logvar = self.to_logvar(t_feat)
        logvar = logvar.clamp(min=-10.0, max=10.0)

        # 为了稳定，直接用 mu 作为 t_hat，不采样
        t_hat = mu

        return t_hat, mu, logvar

    def cosine_align_loss(self, z, t_hat):
        z = F.normalize(z, dim=-1)
        t_hat = F.normalize(t_hat, dim=-1)
        return (1.0 - (z * t_hat).sum(dim=-1)).mean()

    def forward(self, img_feat, t_feat, labels, tail_mask):
        device = img_feat.device

        labels = labels.to(device).long()
        tail_mask = tail_mask.to(device).bool()
        t_feat = t_feat.to(device)

        # 没有 tail 样本，直接返回 0
        if tail_mask.sum() == 0:
            zero = img_feat.sum() * 0.0
            logs = {
                "loss_teacher": 0.0,
                "loss_rate": 0.0,
                "loss_tcls": 0.0,
                "loss_align": 0.0,
            }
            return zero, logs

        # -------------------------------------------------
        # 1. 取学生 tail 特征
        # -------------------------------------------------
        z_tail = img_feat[tail_mask]          # [B_tail, D]
        labels_tail = labels[tail_mask]       # [B_tail]
        t_tail = t_feat

        # -------------------------------------------------
        # 3. 全局 tail 标签 -> 子任务标签
        # -------------------------------------------------
        tail_labels = self.map_to_tail_label(labels_tail)  # [B_tail]

        # -------------------------------------------------
        # 4. teacher feature 压缩
        # -------------------------------------------------
        t_hat, mu, logvar = self.compress_teacher_feat(t_tail)

        # -------------------------------------------------
        # loss 1: rate
        # -------------------------------------------------
        loss_rate = kl_to_standard_normal(mu, logvar).mean()

        # -------------------------------------------------
        # loss 2: t_hat -> tail label
        # 使用传入的 teacher_fc
        # -------------------------------------------------
        teacher_logits = self.teacher_fc(t_hat)
        loss_tcls = F.cross_entropy(teacher_logits, tail_labels)

        # -------------------------------------------------
        # loss 3: student tail feature <-> compressed teacher feature
        # stop-gradient on t_hat
        # -------------------------------------------------
        loss_align = self.cosine_align_loss(z_tail, t_hat.detach())

        # -------------------------------------------------
        # total
        # -------------------------------------------------
        loss = (
            self.lambda_rate * loss_rate
            + self.lambda_cls * loss_tcls
            + self.lambda_align * loss_align
        )

        logs = {
            "loss_teacher": float(loss.detach().item()),
            "loss_rate": float(loss_rate.detach().item()),
            "loss_tcls": float(loss_tcls.detach().item()),
            "loss_align": float(loss_align.detach().item()),
        }

        return loss, logs