# ECHI-Scene-Generation

## Installation

The package is being managed by `uv` and can be installed using the following command:

```bash
uv sync
source .venv/bin/activate
```

## Usage

Hydra is being used for configuration

```bash
python src/libri_demand_scene_gen.py
```

To override the data root directory, use the `--data_root` flag.

```bash
python src/libri_demand_scene_gen.py data_root=/path/to/data
```

To see the full set of configuration options, use the `--help` flag.

```bash
python src/libri_demand_scene_gen.py --help
```
