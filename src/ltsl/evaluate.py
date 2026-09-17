import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .metrics import full_pool_metrics
from .model import TrajectoryEncoder


@torch.no_grad()
def encode_all(model, sequences, lengths, ids, device, batch_size=128):
    order = np.argsort(lengths[ids], kind="stable")
    vectors = []
    model.eval()
    for start in range(0, len(ids), batch_size):
        batch = ids[order[start : start + batch_size]]
        width = int(lengths[batch].max())
        seq = torch.as_tensor(sequences[batch, :width], dtype=torch.long, device=device)
        lens = torch.as_tensor(lengths[batch], dtype=torch.long, device=device)
        vectors.append(model(seq, lens)[0].cpu())
    return torch.cat(vectors)[np.argsort(order)].double()


def evaluate_checkpoint(checkpoint, cache_root, city, metric, split="test", device="cuda"):
    checkpoint = Path(checkpoint)
    city_root = Path(cache_root) / city
    metric_root = city_root / metric
    with np.load(city_root / "trajectories.npz") as data:
        sequences, lengths = data["sequences"], data["lengths"]
    ids = np.load(metric_root / f"{split}_ids.npy")
    target = np.load(metric_root / f"{split}_distances.npy", mmap_mode="r")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TrajectoryEncoder(state["table"]).to(device)
    model.load_state_dict(state)
    vectors = encode_all(model, sequences, lengths, ids, device)
    prediction = torch.cdist(vectors, vectors).numpy()
    result = full_pool_metrics(target, prediction)
    result.update(query_count=len(ids), candidates_per_query=len(ids) - 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--city", choices=("porto", "chengdu"), required=True)
    parser.add_argument("--metric", choices=("netdtw", "tp"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint, args.cache_root, args.city, args.metric, args.split, args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

