import numpy as np


def build_neighbor_plans(train_ids, distance_matrix, epochs=10, seed=43):
    lookup = np.full(int(train_ids.max()) + 1, -1, dtype=np.int64)
    lookup[train_ids] = np.arange(len(train_ids))
    top = np.empty((len(train_ids), 200), dtype=np.int64)
    for row in range(len(train_ids)):
        order = np.argsort(distance_matrix[row], kind="stable")
        order = order[order != row]
        top[row] = train_ids[order[:200]]

    candidates = np.empty((epochs, len(train_ids), 8), dtype=np.int64)
    partners = np.empty((epochs, len(train_ids), 2), dtype=np.int64)
    partner_columns = np.empty_like(partners)
    for epoch in range(1, epochs + 1):
        for row in range(len(train_ids)):
            rng = np.random.default_rng(
                4100009 + row * 1019 + epoch * 100043 + (seed - 42) * 1000000007
            )
            near = rng.choice(top[row, :10], 2, replace=False)
            middle = rng.choice(top[row, 10:50], 2, replace=False)
            hard = rng.choice(top[row, 50:200], 2, replace=False)
            forbidden = set(map(int, np.r_[train_ids[row], top[row]]))
            far = []
            while len(far) < 2:
                value = int(train_ids[rng.integers(len(train_ids))])
                if value not in forbidden:
                    far.append(value)
                    forbidden.add(value)
            candidates[epoch - 1, row] = [
                near[0], far[0], near[1], middle[0], middle[1], hard[0], hard[1], far[1]
            ]
            columns = np.random.default_rng(
                1700003 + row * 1009 + epoch * 100003 + (seed - 42) * 1000000007
            ).choice(50, 2, replace=False)
            partner_columns[epoch - 1, row] = columns
            partners[epoch - 1, row] = top[row, columns]

    top50_values = distance_matrix[
        np.arange(len(train_ids))[:, None], lookup[top[:, :50]]
    ]
    scale = np.c_[top50_values[:, 0], top50_values[:, -1] - top50_values[:, 0] + 1e-8]
    return top[:, :50], candidates, partners, partner_columns, scale

