import numpy as np
import torch
from torch.nn import functional as F


def listwise_rank_loss(vector_distances, target_distances, nearest, span):
    logits = -(target_distances - nearest[:, None]) / span[:, None]
    target = torch.softmax(logits, dim=-1)
    return F.kl_div(
        torch.log_softmax(-vector_distances, dim=-1),
        target,
        reduction="batchmean",
    )


def _directional_kl(logits, energy, source_valid, target_valid, confidence):
    target = torch.softmax(
        -energy.detach().masked_fill(~target_valid[:, None, :], torch.inf), dim=-1
    )
    prediction = torch.log_softmax(
        logits.masked_fill(~target_valid[:, None, :], -torch.inf), dim=-1
    ).masked_fill(~target_valid[:, None, :], 0.0)
    log_target = torch.special.xlogy(target, target)
    kl = (log_target - target * prediction).sum(dim=-1)
    if confidence:
        entropy = -log_target.sum(dim=-1)
        size = target_valid.sum(dim=-1).to(logits.dtype)[:, None]
        normalized = torch.where(
            size > 1,
            entropy / size.clamp_min(2).log(),
            torch.zeros_like(entropy),
        )
        row_weight = 0.05 + 0.95 * (1.0 - normalized).clamp(0.0, 1.0)
    else:
        row_weight = torch.ones_like(kl)
    row_weight = row_weight.detach() * source_valid
    return (row_weight * kl).sum(dim=-1) / row_weight.sum(dim=-1)


def soft_matching_loss(
    tokens_a,
    tokens_b,
    energy,
    lengths_a,
    lengths_b,
    temperature=0.05,
    confidence=False,
):
    a = F.normalize(tokens_a, dim=-1)
    b = F.normalize(tokens_b, dim=-1)
    logits = a @ b.transpose(1, 2) / temperature
    valid_a = torch.arange(a.shape[1], device=a.device)[None] < lengths_a[:, None]
    valid_b = torch.arange(b.shape[1], device=b.device)[None] < lengths_b[:, None]
    forward = _directional_kl(logits, energy, valid_a, valid_b, confidence)
    backward = _directional_kl(
        logits.transpose(1, 2),
        energy.transpose(1, 2),
        valid_b,
        valid_a,
        confidence,
    )
    return 0.5 * (forward + backward).mean()


def target_probabilities(values, nearest, span):
    logits = -(values - nearest[:, None]) / span[:, None]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)

