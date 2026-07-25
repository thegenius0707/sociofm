# SocioFM

SocioFM is an open, compute-efficient family of causal language models for
structured societal event streams. The release accompanies the manuscript
**“Open, compute-efficient foundation models reveal scale–transfer trade-offs
in societal event streams.”**

## Release scope

This repository contains:

- the keyless GDELT 2.0 acquisition and quality-control pipeline;
- chronological split generation;
- compact causal-model pretraining;
- language-model, downstream, calibration, geographic and media audits;
- statistical inference and compute-accounting code; and
- source code that regenerates the manuscript figures.

Rendered figures, large processed datasets, cached GDELT archives, embedding
caches, training logs and model weights are intentionally excluded from Git.
The processed dataset and five reported model checkpoints are distributed as
separate Zenodo records.

## Repository structure

```text
src/sociofm/
├── data/            # collection, filtering, aggregation and split manifests
├── training/        # compact causal-model pretraining
├── evaluation/      # benchmarks, robustness, calibration and statistics
└── visualization/   # manuscript figure generation
configs/             # model, training and temporal-split specifications
manifests/           # instructions for external dataset manifests
results/             # machine-readable result schema; large outputs excluded
tests/               # release-layout and syntax checks
```

## Installation

Python 3.12 and a CUDA-capable PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Minimal pipeline check

```bash
python -m sociofm.data.smoke_test   --gdelt-files 2   --countries ALL   --out data/smoke
```

## Build a dated event shard

```bash
python -m sociofm.data.build_dataset   --start-date 2025-01-01   --end-date 2025-01-07   --out data/example_week   --workers 8   --cache-dir data/cache/gdelt   --source-cap 0
```

## Create chronological manifests

```bash
python -m sociofm.data.prepare_splits   --data-root data/monthly_2022_2025   --out data/manifests
```

The split policy is:

- training: target dates before 1 January 2025;
- validation: 1 January–30 June 2025; and
- forward test: 1 July–31 December 2025.

## Train a compact model

```bash
python -m sociofm.training.train_compact   --manifests data/manifests   --output checkpoints/sociofm_57m   --layers 10   --hidden 512   --heads 8   --max-steps 100000   --batch-size 4   --grad-accum 8   --block-size 256
```

## Reproduce the main analyses

```bash
python -m sociofm.evaluation.run_downstream_suite   --data-root data/monthly_2022_2025   --out results/downstream_suite.json   --seeds 7,17,29

python -m sociofm.evaluation.run_geographic_generalization   --data-root data/monthly_2022_2025   --out results/geographic_generalization.json   --folds 5   --seed 7
```

GPU audits and frozen-representation benchmarks require a released checkpoint;
their complete command-line interfaces are available through `--help`.

## Released checkpoints

Only the five checkpoints reported in the manuscript are part of the model
release:

1. SocioFM 26.5M at 50,000 steps;
2. SocioFM 26.5M at 100,000 steps;
3. SocioFM 57.4M at 100,000 steps;
4. SocioFM 101.3M at 100,000 steps; and
5. SocioFM 101.3M at 200,000 steps.

Intermediate checkpoints saved every 2,000 steps are not distributed.

## Data provenance

The source records are public GDELT 2.0 event exports. Full news-article text is
not collected or redistributed. Dataset manifests retain archive URLs,
timestamps, arguments and quality counts.

## Reproducibility

`SHA256SUMS.txt` lists every release file. `release_manifest.json` records the
source-to-release mapping and source hashes. Run:

```bash
python tests/test_release_layout.py
```

before creating a GitHub release.

## Licence

Source code is released under the MIT License. Dataset and model records include
their own provenance and reuse documentation.
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21576730.svg)](https://doi.org/10.5281/zenodo.21576730)
[![Model DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21577245.svg)](https://doi.org/10.5281/zenodo.21577245)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21577601.svg)](https://doi.org/10.5281/zenodo.21577601)

## Archived research objects

- Software release: https://doi.org/10.5281/zenodo.21576730
- Model checkpoints: https://doi.org/10.5281/zenodo.21577245
- Processed dataset: https://doi.org/10.5281/zenodo.21577601
