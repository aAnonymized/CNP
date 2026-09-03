import torch
import torch.nn.functional as F


def sinkhorn_ot_loss(
    z_tail,
    t_clean,
    labels_tail=None,
    eps=0.05,
    iters=50,
    label_penalty=5.0,
    normalize=False,
):
    """
    Label-aware Sinkhorn OT loss.

    Args:
        z_tail:      student tail features, [B_tail, D]
        t_clean:     purified teacher features, [B_tail, D]
        labels_tail: tail labels, [B_tail]
        eps:         entropy regularization coefficient
        iters:       number of Sinkhorn iterations
        label_penalty: cost penalty for matching different labels
        normalize:   whether to L2-normalize features before computing cost

    Returns:
        loss_ot: scalar
    """

    B = z_tail.size(0)

    if B == 0:
        return z_tail.sum() * 0.0

    if B == 1:
        if normalize:
            z = F.normalize(z_tail, dim=1)
            t = F.normalize(t_clean.detach(), dim=1)
            return F.mse_loss(z, t)
        else:
            return F.mse_loss(z_tail, t_clean.detach())

    # stop-gradient on teacher side
    t_clean = t_clean.detach()

    if normalize:
        z = F.normalize(z_tail, dim=1)
        t = F.normalize(t_clean, dim=1)
    else:
        z = z_tail
        t = t_clean
        
    cost = (z[:, None, :] - t[None, :, :]).pow(2).mean(dim=-1)  # [B, B]
    if labels_tail is not None:
        labels_tail = labels_tail.to(z_tail.device).long()
        same_label = labels_tail[:, None].eq(labels_tail[None, :])
        cost = cost + (~same_label).float() * label_penalty

    # -------------------------------------------------
    # 3. Uniform marginals
    # -------------------------------------------------
    a = torch.full((B,), 1.0 / B, device=z_tail.device)
    b = torch.full((B,), 1.0 / B, device=z_tail.device)

    # -------------------------------------------------
    # 4. Sinkhorn iterations
    # -------------------------------------------------
    K = torch.exp(-cost / eps).clamp_min(1e-8)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)
    P = u[:, None] * K * v[None, :]
    loss_ot = (P * cost).sum()

    return loss_ot