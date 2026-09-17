import argparse
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads

from .data import (
    all_used_shortest_paths,
    build_segment_graph,
    isolate_near_duplicates,
    load_network,
    prepare_landmarks,
    prepare_trajectories,
    read_trajectories,
    save_json,
)
from .sampling import build_neighbor_plans
from .targets import netdtw_energy, pairwise_block, tp_sigma


def load_config(path):
    return json.loads(Path(path).read_text())


def prepare_city(city, data_root, cache_root, config):
    source = Path(data_root) / city
    output = Path(cache_root) / city
    output.mkdir(parents=True, exist_ok=True)

    frame = load_network(source / "network.gpkg")
    trajectories = prepare_trajectories(
        read_trajectories(source / "trajectories.csv"),
        frame,
        min_segments=config["min_segments"],
        max_segments=config["max_segments"],
        split_seed=config["split_seed"],
    )
    np.savez_compressed(output / "trajectories.npz", **trajectories)

    segment_graph = build_segment_graph(frame)
    train = trajectories["train"]
    sequences = trajectories["sequences"]
    lengths = trajectories["lengths"]
    train_segments = np.unique(
        sequences[train][
            np.arange(sequences.shape[1])[None] < lengths[train, None]
        ]
    )
    subgraph, used_local, landmarks, landmark_ids, mean, std = prepare_landmarks(
        segment_graph,
        trajectories["used_full_indices"],
        train_segments,
        count=config["num_landmarks"],
        seed=config["landmark_seed"],
    )
    np.save(output / "landmarks.npy", landmarks)
    np.savez(
        output / "landmark_metadata.npz",
        landmark_edge_ids=frame.id.to_numpy(np.int64)[landmark_ids],
        mean=mean,
        std=std,
        train_segments=train_segments,
    )
    shortest_path_file = output / "shortest_paths.npy"
    if not shortest_path_file.exists():
        all_used_shortest_paths(subgraph, used_local, shortest_path_file)

    benchmark = output / "benchmark"
    benchmark.mkdir(exist_ok=True)
    for split in ("val", "test"):
        ids = isolate_near_duplicates(
            sequences,
            lengths,
            trajectories[split],
            threshold=config["route_overlap_threshold"],
        )
        np.save(benchmark / f"{split}_ids.npy", ids)

    save_json(
        output / "metadata.json",
        {
            "city": city,
            "trajectory_count": int(len(lengths)),
            "train_count": int(len(train)),
            "validation_count": int(len(trajectories["val"])),
            "test_count": int(len(trajectories["test"])),
            "retained_validation_count": int(len(np.load(benchmark / "val_ids.npy"))),
            "retained_test_count": int(len(np.load(benchmark / "test_ids.npy"))),
            "used_segments": int(len(trajectories["used_edge_ids"])),
            "split_seed": config["split_seed"],
            "landmark_seed": config["landmark_seed"],
        },
    )


def _write_matrix(path, sequences, lengths, ids, shortest_paths, metric, sigma):
    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(len(ids), len(ids))
    )
    metric_code = 0 if metric == "netdtw" else 1
    for start in range(0, len(ids), 32):
        stop = min(start + 32, len(ids))
        matrix[start:stop] = pairwise_block(
            sequences,
            lengths,
            ids[start:stop],
            ids,
            shortest_paths,
            metric_code,
            sigma,
        )
        matrix.flush()
    return matrix


def _energy_cache(path, metric, sequences, lengths, train, top50, selected, shortest_paths, config):
    path.mkdir(parents=True, exist_ok=True)
    pair_ids = np.unique(
        (np.arange(len(train))[None, :, None] * 50 + selected).ravel()
    )
    sizes = np.zeros(len(train) * 50, dtype=np.int64)
    left = train[pair_ids // 50]
    right = top50[pair_ids // 50, pair_ids % 50]
    sizes[pair_ids] = lengths[left] * lengths[right]
    offsets = np.r_[0, np.cumsum(sizes)]
    np.save(path / "offsets.npy", offsets)
    flat = np.lib.format.open_memmap(
        path / "values.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]),)
    )
    for pair_id, a, b in zip(pair_ids, left, right):
        aa = sequences[a, : lengths[a]]
        bb = sequences[b, : lengths[b]]
        if metric == "netdtw":
            value = netdtw_energy(
                aa, bb, shortest_paths, config["netdtw_temperature_km"]
            )
        else:
            value = shortest_paths[np.ix_(aa, bb)] / config["tp_temperature_km"]
        flat[offsets[pair_id] : offsets[pair_id + 1]] = value.astype(np.float32).ravel()
    flat.flush()


def prepare_metric(city, metric, cache_root, config):
    city_root = Path(cache_root) / city
    output = city_root / metric
    output.mkdir(parents=True, exist_ok=True)
    with np.load(city_root / "trajectories.npz") as data:
        sequences = data["sequences"]
        lengths = data["lengths"]
        train = data["train"]
    shortest_paths = np.load(city_root / "shortest_paths.npy", mmap_mode="r")
    sigma = 1.0
    if metric == "tp":
        sigma = tp_sigma(sequences, lengths, train, shortest_paths)

    split_ids = {
        "train": train,
        "val": np.load(city_root / "benchmark/val_ids.npy"),
        "test": np.load(city_root / "benchmark/test_ids.npy"),
    }
    matrices = {}
    for split, ids in split_ids.items():
        np.save(output / f"{split}_ids.npy", ids)
        path = output / f"{split}_distances.npy"
        matrices[split] = _write_matrix(
            path, sequences, lengths, ids, shortest_paths, metric, sigma
        )

    top50, candidates, partners, selected, scale = build_neighbor_plans(
        train, matrices["train"], epochs=config["epochs"], seed=config["seed"]
    )
    for name, value in (
        ("top50", top50),
        ("candidates", candidates),
        ("partners", partners),
        ("partner_columns", selected),
        ("rank_scale", scale),
    ):
        np.save(output / f"{name}.npy", value)
    _energy_cache(
        output / "matching_energy",
        metric,
        sequences,
        lengths,
        train,
        top50,
        selected,
        shortest_paths,
        config,
    )
    save_json(
        output / "metadata.json",
        {
            "metric": metric,
            "tp_sigma_km": None if metric == "netdtw" else sigma,
            "train_count": int(len(train)),
            "validation_count": int(len(split_ids["val"])),
            "test_count": int(len(split_ids["test"])),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("porto", "chengdu"), required=True)
    parser.add_argument("--metric", choices=("netdtw", "tp", "both"), default="both")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    set_num_threads(args.threads)
    config = load_config(args.config)
    prepare_city(args.city, args.data_root, args.cache_root, config)
    metrics = ("netdtw", "tp") if args.metric == "both" else (args.metric,)
    for metric in metrics:
        prepare_metric(args.city, metric, args.cache_root, config)


if __name__ == "__main__":
    main()

