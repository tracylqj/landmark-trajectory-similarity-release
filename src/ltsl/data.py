import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def read_trajectories(path):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["trajectory_id"], tuple(map(int, row["cpath"].split(",")))))
    return rows


def load_network(path):
    import pyogrio

    frame = pyogrio.read_dataframe(
        path, layer="edges", columns=["id", "source", "target", "length"]
    ).sort_values("id")
    ids = frame.id.to_numpy(np.int64)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("road segment IDs must be consecutive from zero")
    return frame


def prepare_trajectories(rows, frame, min_segments=9, max_segments=99, split_seed=42):
    sources = frame.source.to_numpy(np.int64)
    targets = frame.target.to_numpy(np.int64)
    edge_ids = frame.id.to_numpy(np.int64)
    edge_lookup = {int(edge): i for i, edge in enumerate(edge_ids)}

    paths, ids, node_groups = [], [], {}
    for trajectory_id, path in rows:
        if not min_segments <= len(path) <= max_segments:
            continue
        indices = np.array([edge_lookup[edge] for edge in path], dtype=np.int32)
        if not np.array_equal(targets[indices[:-1]], sources[indices[1:]]):
            raise ValueError(f"disconnected map-matched path: {trajectory_id}")
        node_path = tuple(np.r_[sources[indices[:1]], targets[indices]].tolist())
        row_id = len(paths)
        paths.append(indices)
        ids.append(trajectory_id)
        node_groups.setdefault(node_path, []).append(row_id)

    rng = np.random.default_rng(split_seed)
    keys = list(node_groups)
    order = rng.permutation(len(keys))
    first, second = int(0.6 * len(keys)), int(0.8 * len(keys))
    splits = {}
    for name, selected in zip(
        ("train", "val", "test"),
        (order[:first], order[first:second], order[second:]),
    ):
        splits[name] = np.array(
            [row for key in selected for row in node_groups[keys[key]]], dtype=np.int32
        )

    used = np.unique(np.concatenate(paths))
    remap = np.full(len(frame), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    lengths = np.array([len(path) for path in paths], dtype=np.int32)
    sequences = np.zeros((len(paths), max_segments), dtype=np.int32)
    for row, path in enumerate(paths):
        sequences[row, : len(path)] = remap[path]
    return {
        "sequences": sequences,
        "lengths": lengths,
        "trajectory_ids": np.asarray(ids),
        "used_edge_ids": edge_ids[used],
        "used_full_indices": used,
        **splits,
    }


def build_segment_graph(frame):
    starts = frame.source.to_numpy(np.int64)
    ends = frame.target.to_numpy(np.int64)
    length_km = frame.geometry.length.to_numpy(np.float64) / 1000.0
    outgoing = {}
    for row, node in enumerate(starts):
        outgoing.setdefault(int(node), []).append(row)
    pairs = {}
    for left, node in enumerate(ends):
        for right in outgoing.get(int(node), []):
            if left != right:
                key = (min(left, right), max(left, right))
                pairs[key] = 0.5 * (length_km[left] + length_km[right])
    indices = np.asarray(list(pairs), dtype=np.int64)
    weights = np.asarray(list(pairs.values()), dtype=np.float64)
    graph = coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[indices[:, 0], indices[:, 1]], np.r_[indices[:, 1], indices[:, 0]]),
        ),
        shape=(len(frame), len(frame)),
    ).tocsr()
    return graph


def prepare_landmarks(graph, used_full_indices, train_segments, count=32, seed=20260914):
    _, labels = connected_components(graph, directed=False)
    components = np.unique(labels[used_full_indices])
    if len(components) != 1:
        raise ValueError("trajectory segments span disconnected components")
    component = np.flatnonzero(labels == components[0])
    local = np.full(graph.shape[0], -1, np.int64)
    local[component] = np.arange(len(component))
    subgraph = graph[component][:, component]
    used_local = local[used_full_indices]

    rng = np.random.default_rng(seed)
    selected, columns = [], []
    nearest = np.full(len(component), np.inf)
    landmark = int(rng.integers(len(component)))
    for _ in range(count):
        distance = dijkstra(subgraph, directed=False, indices=landmark)
        selected.append(landmark)
        columns.append(distance)
        nearest = np.minimum(nearest, distance)
        landmark = int(nearest.argmax())
    full_table = np.stack(columns, axis=1)
    table = full_table[used_local] / float(full_table.max())
    mean = table[train_segments].mean(axis=0)
    std = np.maximum(table[train_segments].std(axis=0), 1e-8)
    normalized = ((table - mean) / std).astype(np.float32)
    return subgraph, used_local, normalized, component[np.asarray(selected)], mean, std


def all_used_shortest_paths(subgraph, used_local, output, block_size=64):
    output = Path(output)
    matrix = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float64, shape=(len(used_local), len(used_local))
    )
    for start in range(0, len(used_local), block_size):
        stop = min(start + block_size, len(used_local))
        matrix[start:stop] = dijkstra(
            subgraph, directed=False, indices=used_local[start:stop]
        )[:, used_local]
        matrix.flush()
    return matrix


def isolate_near_duplicates(sequences, lengths, ids, threshold=0.8):
    rows, columns, sizes = [], [], []
    for row, trajectory in enumerate(ids):
        tokens = set(map(int, sequences[trajectory, : lengths[trajectory]]))
        rows.extend([row] * len(tokens))
        columns.extend(tokens)
        sizes.append(len(tokens))
    incidence = csr_matrix(
        (np.ones(len(rows), np.int32), (rows, columns)),
        shape=(len(ids), int(sequences.max()) + 1),
    )
    intersection = (incidence @ incidence.T).toarray()
    sizes = np.asarray(sizes)
    jaccard = intersection / (sizes[:, None] + sizes[None, :] - intersection)
    left, right = np.where(np.triu(jaccard >= threshold - 1e-12, k=1))
    excluded = np.zeros(len(ids), dtype=bool)
    for x, y in zip(left, right):
        a = sequences[ids[x], : lengths[ids[x]]]
        b = sequences[ids[y], : lengths[ids[y]]]
        previous = [0] * (len(b) + 1)
        for token in a:
            current = [0]
            for col, other in enumerate(b):
                current.append(
                    previous[col] + 1
                    if token == other
                    else max(previous[col + 1], current[-1])
                )
            previous = current
        if previous[-1] / max(len(a), len(b)) >= threshold - 1e-12:
            excluded[x] = excluded[y] = True
    return ids[~excluded]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
