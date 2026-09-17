import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import encode_all
from .losses import soft_matching_loss
from .metrics import full_pool_metrics
from .model import TrajectoryEncoder


def set_seed(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class Trainer:
    def __init__(self, city, metric, cache_root, output, config, device="cuda"):
        set_seed(config["seed"])
        self.city = city
        self.metric = metric
        self.config = config
        self.device = torch.device(device)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        city_root = Path(cache_root) / city
        self.metric_root = city_root / metric

        with np.load(city_root / "trajectories.npz") as data:
            self.sequences = data["sequences"]
            self.lengths = data["lengths"]
            self.train_ids = data["train"]
        self.landmarks = np.load(city_root / "landmarks.npy")
        self.model = TrajectoryEncoder(self.landmarks).to(self.device)
        self.train_distances = np.load(
            self.metric_root / "train_distances.npy", mmap_mode="r"
        )
        self.candidate_plans = np.load(
            self.metric_root / "candidates.npy", mmap_mode="r"
        )
        self.partner_plans = np.load(
            self.metric_root / "partners.npy", mmap_mode="r"
        )
        self.partner_columns = np.load(
            self.metric_root / "partner_columns.npy", mmap_mode="r"
        )
        self.rank_scale = np.load(self.metric_root / "rank_scale.npy")
        self.energy = np.load(
            self.metric_root / "matching_energy/values.npy", mmap_mode="r"
        )
        self.offsets = np.load(self.metric_root / "matching_energy/offsets.npy")
        self.lookup = np.full(len(self.lengths), -1, dtype=np.int64)
        self.lookup[self.train_ids] = np.arange(len(self.train_ids))
        self.sequences_gpu = torch.as_tensor(
            self.sequences, dtype=torch.long, device=self.device
        )
        self.lengths_gpu = torch.as_tensor(
            self.lengths, dtype=torch.long, device=self.device
        )
        self.validation_ids = np.load(self.metric_root / "val_ids.npy")
        self.validation_distances = np.load(
            self.metric_root / "val_distances.npy", mmap_mode="r"
        )

    def encode(self, ids):
        ids = np.asarray(ids)
        width = int(self.lengths[ids].max())
        tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        lengths = self.lengths_gpu[tensor_ids]
        return (*self.model(self.sequences_gpu[tensor_ids, :width], lengths), lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.candidates = self.candidate_plans[epoch - 1]
        self.partners = self.partner_plans[epoch - 1]
        target = self.train_distances[
            np.arange(len(self.train_ids))[:, None], self.lookup[self.candidates]
        ]
        nearest, span = self.rank_scale.T
        logits = -(target - nearest[:, None]) / span[:, None]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        self.rank_target = torch.as_tensor(
            probabilities, dtype=torch.float32, device=self.device
        )

    def _matching_energy(self, rows, width):
        columns = self.partner_columns[self.epoch - 1, rows]
        pair_ids = (rows[:, None] * 50 + columns).ravel()
        anchors = np.repeat(self.train_ids[rows], 2)
        partners = self.partners[rows].ravel()
        result = np.zeros((len(pair_ids), width, width), dtype=np.float32)
        for index, pair_id in enumerate(pair_ids):
            left = int(self.lengths[anchors[index]])
            right = int(self.lengths[partners[index]])
            start, stop = self.offsets[pair_id : pair_id + 2]
            result[index, :left, :right] = self.energy[start:stop].reshape(left, right)
        return torch.as_tensor(result, device=self.device)

    def loss(self, rows):
        rows = np.atleast_1d(rows)
        batch = len(rows)
        anchors = self.train_ids[rows]
        candidates = self.candidates[rows]
        partners = self.partners[rows]
        ids, inverse = np.unique(
            np.r_[anchors, candidates.ravel(), partners.ravel()], return_inverse=True
        )
        vectors, tokens, lengths = self.encode(ids)
        anchor_index = torch.as_tensor(inverse[:batch], device=self.device)
        candidate_index = torch.as_tensor(
            inverse[batch : batch + batch * 8].reshape(batch, 8), device=self.device
        )
        vector_distances = torch.linalg.vector_norm(
            vectors[anchor_index, None] - vectors[candidate_index], dim=-1
        )
        main = torch.nn.functional.kl_div(
            torch.log_softmax(-vector_distances, dim=-1),
            self.rank_target[torch.as_tensor(rows, device=self.device)],
            reduction="batchmean",
        )

        anchor_token_index = anchor_index[:, None].expand(-1, 2).reshape(-1)
        partner_index = torch.as_tensor(
            inverse[batch + batch * 8 :], device=self.device
        )
        energy = self._matching_energy(rows, tokens.shape[1])
        auxiliary = soft_matching_loss(
            tokens[anchor_token_index],
            tokens[partner_index],
            energy,
            lengths[anchor_token_index],
            lengths[partner_index],
            temperature=self.config["soft_temperature"],
            confidence=self.metric == "netdtw",
        )
        return main, auxiliary

    @torch.no_grad()
    def validate(self):
        vectors = encode_all(
            self.model,
            self.sequences,
            self.lengths,
            self.validation_ids,
            self.device,
        )
        prediction = torch.cdist(vectors, vectors).numpy()
        return full_pool_metrics(self.validation_distances, prediction)

    def run(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        best_score = (-1.0, -1.0)
        best_epoch = None
        history = self.output / "history.jsonl"
        started = time.time()
        with history.open("w", buffering=1) as log:
            for epoch in range(1, self.config["epochs"] + 1):
                self.set_epoch(epoch)
                self.model.train()
                order = np.random.default_rng(
                    self.config["seed"] + epoch * 1009
                ).permutation(len(self.train_ids))
                total_main = total_aux = 0.0
                for offset in range(0, len(order), self.config["batch_size"]):
                    batch = order[offset : offset + self.config["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(batch), self.config["micro_batch_size"]):
                        rows = batch[start : start + self.config["micro_batch_size"]]
                        main, auxiliary = self.loss(rows)
                        weight = len(rows) / len(batch)
                        (weight * (main + self.config["soft_weight"] * auxiliary)).backward()
                        total_main += float(main.detach()) * len(rows)
                        total_aux += float(auxiliary.detach()) * len(rows)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                validation = self.validate()
                prefix = "tie_" if self.metric == "tp" else ""
                score = (
                    validation[prefix + "HR@10"],
                    validation[prefix + "HR@50"],
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), self.output / "best.pt")
                row = {
                    "epoch": epoch,
                    "main_loss": total_main / len(order),
                    "matching_loss": total_aux / len(order),
                    "validation": validation,
                    "seconds": time.time() - started,
                }
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        result = {
            "city": self.city,
            "metric": self.metric,
            "seed": self.config["seed"],
            "best_epoch": best_epoch,
            "validation": self.validate_checkpoint(self.output / "best.pt"),
        }
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def validate_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self.validate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("porto", "chengdu"), required=True)
    parser.add_argument("--metric", choices=("netdtw", "tp"), required=True)
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    Trainer(
        args.city,
        args.metric,
        args.cache_root,
        args.output,
        config,
        args.device,
    ).run()


if __name__ == "__main__":
    main()

