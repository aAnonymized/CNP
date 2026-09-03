import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TailForegroundBackgroundDecoupleLoss(nn.Module):
    """
    Tail-only foreground-background decoupling loss.

    Teacher CAM is only used as a class-agnostic foreground prior.
    No feature-level alignment between student and teacher is enforced.
    """

    def __init__(
        self,
        teacher_fc,
        student_fc,
        num_classes=8,
        tau=0.3,
        lambda_fg=1.0,
        lambda_bg=0.5,
        lambda_dec=0.1,
        bg_temp=1.0,
    ):
        super().__init__()
        self.teacher_fc = teacher_fc
        self.student_fc = student_fc
        self.num_classes = num_classes
        self.tau = tau

        self.lambda_fg = lambda_fg
        self.lambda_bg = lambda_bg
        self.lambda_dec = lambda_dec
        self.bg_temp = bg_temp

    @torch.no_grad()
    def class_cam(self, feat, weight, labels):
        """
        feat: [B, C, H, W]
        weight: [num_teacher_classes, C]
        labels: [B], teacher-side labels
        """
        w = weight[labels].detach()  # [B, C]

        cam = torch.einsum("bc,bchw->bhw", w, feat)
        cam = F.relu(cam)

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

        return cam.unsqueeze(1)  # [B, 1, H, W]

    def forward(self, s_feat, t_feat, labels, tail_classes):
        """
        s_feat: [B, Cs, Hs, Ws], student feature
        t_feat: [B, Ct, Ht, Wt], teacher feature
        labels: [B], global labels, e.g. 0~7
        tail_classes: list, e.g. [5, 6, 7]
        """
        device = s_feat.device
        labels = labels.to(device).long()

        if not torch.is_tensor(tail_classes):
            tail_classes = torch.tensor(tail_classes, device=device).long()
        else:
            tail_classes = tail_classes.to(device).long()

        # tail sample mask
        mask = (labels[:, None] == tail_classes[None, :]).any(dim=1)

        if mask.sum() == 0:
            return s_feat.sum() * 0.0

        # Important: mask both student and teacher features
        s_feat = s_feat[mask]
        t_feat = t_feat
        labels = labels[mask]

        map_table = torch.full(
            (self.num_classes,),
            -1,
            device=device,
            dtype=torch.long,
        )
        map_table[tail_classes] = torch.arange(len(tail_classes), device=device)
        t_labels = map_table[labels]

        assert (t_labels >= 0).all(), "Some labels are not mapped to teacher labels."

        with torch.no_grad():
            A_t = self.class_cam(t_feat, self.teacher_fc.weight, t_labels)
            if A_t.shape[-2:] != s_feat.shape[-2:]:
                A_t = F.interpolate(
                    A_t,
                    size=s_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        B, C, H, W = s_feat.shape
        N = H * W

        A_flat = A_t.flatten(1)  # [B, N]

        # top-k foreground region
        k = max(1, int(N * self.tau))
        k = min(k, N - 1) if N > 1 else 1

        _, idx_fg = torch.topk(A_flat, k, dim=1)

        mask_fg = torch.zeros_like(A_flat)
        mask_fg.scatter_(1, idx_fg, 1.0)

        mask_bg = 1.0 - mask_fg

        # normalize masks
        mask_fg = mask_fg / (mask_fg.sum(dim=1, keepdim=True) + 1e-8)
        mask_bg = mask_bg / (mask_bg.sum(dim=1, keepdim=True) + 1e-8)

        # 2. aggregate student foreground / background features
        s_tokens = s_feat.flatten(2).transpose(1, 2)  # [B, N, C]

        z_fg = torch.bmm(mask_fg.unsqueeze(1), s_tokens).squeeze(1)  # [B, C]
        z_bg = torch.bmm(mask_bg.unsqueeze(1), s_tokens).squeeze(1)  # [B, C]

        # 3. use student classifier as a fixed probe for auxiliary losses
        # This prevents auxiliary loss from directly damaging classifier weights.
        fc_w = self.student_fc.weight.detach()
        fc_b = self.student_fc.bias.detach() if self.student_fc.bias is not None else None

        logits_fg = F.linear(z_fg, fc_w, fc_b)
        logits_bg = F.linear(z_bg, fc_w, fc_b) / self.bg_temp

        # foreground sufficiency: foreground should classify correctly
        loss_fg = F.cross_entropy(logits_fg, labels)

        # background suppression: background prediction should be uniform
        logp_bg = F.log_softmax(logits_bg, dim=1)
        p_bg = logp_bg.exp()

        loss_bg = (
            p_bg * (logp_bg + math.log(self.num_classes))
        ).sum(dim=1).mean()
        # This is KL(p_bg || Uniform)

        # foreground-background decoupling
        z_fg_n = F.normalize(z_fg, dim=1)
        z_bg_n = F.normalize(z_bg, dim=1)

        loss_dec = (z_fg_n * z_bg_n).sum(dim=1).pow(2).mean()

        loss = (
            self.lambda_fg * loss_fg
            + self.lambda_bg * loss_bg
            + self.lambda_dec * loss_dec
        )

        return loss