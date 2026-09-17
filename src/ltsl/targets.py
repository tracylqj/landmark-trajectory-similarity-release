import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix


@njit(cache=True)
def netdtw(a, b, shortest_paths):
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for i in range(len(a)):
        current = np.full(len(b) + 1, np.inf)
        for j in range(len(b)):
            current[j + 1] = shortest_paths[a[i], b[j]] + min(
                previous[j], previous[j + 1], current[j]
            )
        previous = current
    return previous[-1]


@njit(cache=True)
def netdtw_energy(a, b, shortest_paths, temperature=1.0):
    n, m = len(a), len(b)
    cost = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            cost[i, j] = shortest_paths[a[i], b[j]]
    forward = np.full((n + 1, m + 1), np.inf)
    forward[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            forward[i + 1, j + 1] = cost[i, j] + min(
                forward[i, j], forward[i + 1, j], forward[i, j + 1]
            )
    backward = np.full((n + 1, m + 1), np.inf)
    backward[n, m] = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            backward[i, j] = cost[i, j] + min(
                backward[i + 1, j], backward[i, j + 1], backward[i + 1, j + 1]
            )
    energy = np.empty((n, m), dtype=np.float64)
    optimum = forward[n, m]
    for i in range(n):
        for j in range(m):
            energy[i, j] = max(
                0.0,
                forward[i + 1, j + 1] + backward[i, j] - cost[i, j] - optimum,
            ) / temperature
    return energy


def tp_sigma(sequences, lengths, train_ids, shortest_paths, seed=20260915):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(4096):
        left, right = rng.choice(train_ids, 2, replace=False)
        a = sequences[left, : lengths[left]]
        b = sequences[right, : lengths[right]]
        cost = shortest_paths[np.ix_(a, b)]
        values.extend(cost.min(axis=0).tolist())
        values.extend(cost.min(axis=1).tolist())
    values = np.asarray(values)
    return float(np.median(values[values > 0]))


@njit(cache=True)
def tp_distance(a, b, shortest_paths, sigma):
    forward = 0.0
    for i in range(len(a)):
        best = np.inf
        for j in range(len(b)):
            best = min(best, shortest_paths[a[i], b[j]])
        forward += -np.expm1(-best / sigma) / len(a)
    backward = 0.0
    for j in range(len(b)):
        best = np.inf
        for i in range(len(a)):
            best = min(best, shortest_paths[a[i], b[j]])
        backward += -np.expm1(-best / sigma) / len(b)
    return 0.5 * (forward + backward)


@njit(parallel=True, cache=True)
def pairwise_matrix(sequences, lengths, ids, shortest_paths, metric, sigma):
    result = np.empty((len(ids), len(ids)), dtype=np.float64)
    for i in prange(len(ids)):
        a = sequences[ids[i], : lengths[ids[i]]]
        for j in range(len(ids)):
            b = sequences[ids[j], : lengths[ids[j]]]
            if metric == 0:
                result[i, j] = netdtw(a, b, shortest_paths)
            else:
                result[i, j] = tp_distance(a, b, shortest_paths, sigma)
    return result


@njit(parallel=True, cache=True)
def pairwise_block(sequences, lengths, row_ids, column_ids, shortest_paths, metric, sigma):
    result = np.empty((len(row_ids), len(column_ids)), dtype=np.float64)
    for i in prange(len(row_ids)):
        a = sequences[row_ids[i], : lengths[row_ids[i]]]
        for j in range(len(column_ids)):
            b = sequences[column_ids[j], : lengths[column_ids[j]]]
            if metric == 0:
                result[i, j] = netdtw(a, b, shortest_paths)
            else:
                result[i, j] = tp_distance(a, b, shortest_paths, sigma)
    return result
