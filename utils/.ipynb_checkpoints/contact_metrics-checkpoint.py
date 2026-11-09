"""Contact-probability metrics shared by training and evaluation notebooks."""

import torch


def precision_topk(score, target, topk_list, pad_value=-100):
    """Return precision at each K, excluding padding and clamping K to valid pairs."""
    if score.shape != target.shape:
        raise ValueError(f'Score and target shapes differ: {score.shape} vs {target.shape}')
    ks = [int(k) for k in topk_list]
    if any(k != original or k < 0 for k, original in zip(ks, topk_list)):
        raise ValueError('K values must be nonnegative integers')
    valid = target != pad_value
    scores = score[valid].reshape(-1)
    labels = (target[valid].reshape(-1) > 0).float()
    max_k = min(max(ks, default=0), scores.numel())
    if max_k == 0:
        return {k: 0.0 for k in ks}
    indices = scores.topk(max_k, sorted=True).indices
    labels_cpu = labels.detach().cpu()
    indices_cpu = indices.detach().cpu()
    
    hits = labels_cpu[indices_cpu].cumsum(0)
    return {
        k: float(hits[min(k, scores.numel()) - 1] / min(k, scores.numel())) if k > 0 else 0.0
        for k in ks
    }
