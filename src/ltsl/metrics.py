import numpy as np


METRIC_NAMES = ("HR@1", "HR@10", "HR@50", "R10@50")


def retrieval_metrics(target_distances, predicted_distances):
    truth = np.argsort(target_distances, axis=1, kind="stable")
    prediction = np.argsort(predicted_distances, axis=1, kind="stable")
    scores = {}
    for k in (1, 10, 50):
        scores[f"HR@{k}"] = float(
            np.mean([
                np.isin(pred[:k], true[:k]).sum() / k
                for true, pred in zip(truth, prediction)
            ])
        )
    scores["R10@50"] = float(
        np.mean([
            np.isin(pred[:50], true[:10]).sum() / 10
            for true, pred in zip(truth, prediction)
        ])
    )
    return scores


def full_pool_metrics(target_distances, predicted_distances):
    if target_distances.shape != predicted_distances.shape:
        raise ValueError("target and prediction matrices must have the same shape")
    target = np.asarray(target_distances).copy()
    prediction = np.asarray(predicted_distances).copy()
    np.fill_diagonal(target, np.inf)
    np.fill_diagonal(prediction, np.inf)
    result = retrieval_metrics(target, prediction)
    truth = np.argsort(target, axis=1, kind="stable")
    predicted = np.argsort(prediction, axis=1, kind="stable")
    for size, limit, name in (
        (1, 1, "HR@1"),
        (10, 10, "HR@10"),
        (50, 50, "HR@50"),
        (10, 50, "R10@50"),
    ):
        values = []
        for row, (true_order, predicted_order) in enumerate(zip(truth, predicted)):
            cutoff = target[row, true_order[size - 1]]
            strict = target[row] < cutoff
            tied = target[row] == cutoff
            selected = predicted_order[:limit]
            strict_count = int(strict[selected].sum())
            tied_count = int(tied[selected].sum())
            required_ties = size - int(strict.sum())
            values.append((strict_count + min(required_ties, tied_count)) / size)
        result[f"tie_{name}"] = float(np.mean(values))
    return result

