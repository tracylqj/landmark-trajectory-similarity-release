# Landmark-Based Road-Network Trajectory Similarity Learning

This repository contains the implementation and data used to reproduce the main
trajectory-retrieval results for Porto and Chengdu. The model represents every
road segment by its shortest-path distances to 32 farthest-point landmarks. A
one-layer Transformer produces contextual position representations and a
trajectory vector. Training combines a listwise trajectory-ranking objective
with matching knowledge transfer (MKT) at the position level.

## Data

`data/porto.zip` and `data/chengdu.zip` each contain:

- `trajectories.csv`: all released map-matched trajectories, with an anonymous
  trajectory ID and a comma-separated road-segment sequence (`cpath`);
- `network.gpkg`: the directed road network (`edges` layer) with segment ID,
  source node, target node, length, and geometry.

The archives do not contain predefined train/validation/test files or processed
features. The preprocessing command recreates the paper split and all derived
targets from the complete trajectory collection.

## Model

The released configuration uses 32 landmark-distance features, a 128-dimensional
one-layer four-head Transformer, masked mean pooling, listwise rank KL, and two
MKT trajectory pairs per anchor. Separate models are trained for NetDTW and TP.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

unzip data/porto.zip -d data/
unzip data/chengdu.zip -d data/

python -m ltsl.preprocess --city porto --metric both
python -m ltsl.preprocess --city chengdu --metric both

python -m ltsl.train --city porto --metric netdtw \
  --output outputs/porto_netdtw
python -m ltsl.evaluate --city porto --metric netdtw \
  --checkpoint outputs/porto_netdtw/best.pt
```

Use the same commands with `chengdu` and/or `tp` for the other settings. Exact
all-pair target construction is CPU- and storage-intensive; Porto preprocessing
requires several gigabytes of temporary cache space. The four released
checkpoints under `checkpoints/` can be evaluated after preprocessing.

## Reference performance

Full-pool test retrieval, seed 43:

| Dataset | Target | HR@1 | HR@10 | HR@50 | R10@50 |
|---|---|---:|---:|---:|---:|
| Porto | NetDTW | 0.5994 | 0.7243 | 0.7426 | 0.9888 |
| Chengdu | NetDTW | 0.6249 | 0.7605 | 0.7942 | 0.9943 |
| Porto | TP | 0.5383 | 0.6469 | 0.6846 | 0.9647 |
| Chengdu | TP | 0.6112 | 0.7204 | 0.7619 | 0.9873 |

## Citation

Please cite the accompanying paper when using this code or data. The BibTeX
entry will be added after publication.

