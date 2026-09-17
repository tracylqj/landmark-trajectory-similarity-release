# Data format

Each city archive contains one trajectory table and one road-network file.

`trajectories.csv` has two columns:

- `trajectory_id`: anonymous identifier used only to preserve a stable order;
- `cpath`: comma-separated directed road-segment IDs.

`network.gpkg` contains an `edges` layer with `id`, `source`, `target`, `length`,
and `geometry`. Road-segment IDs in `cpath` refer to the `id` column.

The code license does not supersede the terms of the original trajectory and
road-network data sources. Users are responsible for following those terms.

