# Commands

This project benchmarks a single model — **`dual_kd_gnn`** (Double GCN +
EMA-teacher knowledge distillation + transformer) — across five MoleculeNet
classification datasets: **BACE, BBBP, SIDER, Tox21, ClinTox**.

The number of prediction tasks differs per dataset and is resolved
automatically from the dataset registry ([common/datasets.py](common/datasets.py)),
so the model's classifier head, the training loss, and the evaluation metric all
adapt to whichever dataset you select.

| Dataset | File          | SMILES column | Tasks | Notes |
| ------- | ------------- | ------------- | ----- | ----- |
| bace    | `data/bace.csv`    | `mol`    | 1  | BACE-1 inhibition |
| bbbp    | `data/bbbp.csv`    | `smiles` | 1  | Blood-brain barrier penetration |
| sider   | `data/sider.csv`   | `smiles` | 27 | Adverse-reaction system-organ classes |
| tox21   | `data/tox21.csv`   | `smiles` | 12 | Toxicity assays |
| clintox | `data/clintox.csv` | `smiles` | 2  | FDA approval vs. clinical-trial toxicity |

---

## 0. Environment

```bash
conda activate dualgnn
pip install -r requirements.txt   # only if dependencies are missing
```

---

## 1. Download datasets

Tox21 ships with the repo. Download the rest with **either** option.

### Option A — Python (recommended)

```bash
python scripts/download_data.py            # download all five datasets
python scripts/download_data.py bbbp bace  # download a subset
python scripts/download_data.py --force tox21   # force re-download
```

Files land in `data/` with the canonical names above; gzipped sources are
decompressed automatically.

### Option B — Terminal (wget + gunzip)

```bash
# BBBP and BACE are plain CSV
wget -O data/bbbp.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
wget -O data/bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv

# SIDER, ClinTox, Tox21 are gzipped -> decompress to plain .csv
wget -O data/sider.csv.gz   https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz   && gunzip -f data/sider.csv.gz
wget -O data/clintox.csv.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz && gunzip -f data/clintox.csv.gz
wget -O data/tox21.csv.gz   https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz   && gunzip -f data/tox21.csv.gz
```

(Use `curl -L -o <file> <url>` if `wget` is unavailable.)

---

## 2. Train on a single dataset

[dual_kd_gnn/main.py](dual_kd_gnn/main.py) trains one dataset and saves all
artifacts under `dual_kd_gnn/runs/<dataset>/`.

```bash
python dual_kd_gnn/main.py --dataset bbbp
python dual_kd_gnn/main.py --dataset tox21
python dual_kd_gnn/main.py --dataset sider --gcn-pretrain-epochs 150 --transformer-epochs 150
```

> Device defaults to **cuda**. Override with `--device cpu`, `--device cuda:1`, etc.

Common overrides (all optional — sensible defaults exist):

- Training schedule: `--gcn-pretrain-epochs`, `--transformer-epochs`, `--patience`
- Optimization: `--batch-size`, `--lr`, `--pretrain-lr`, `--transformer-lr`, `--weight-decay`
- Distillation: `--distill-weight`, `--cross-distill-weight`, `--ema-decay`, `--ema-decay-init`
- Reuse tuned hyperparameters: `--best-config dual_kd_gnn/optuna/<study>/best_config.json`
- Manual data overrides: `--data-path`, `--smiles-column`, `--target-columns`

---

## 3. Batch benchmark across all datasets

[main.py](main.py) trains `dual_kd_gnn` on every dataset sequentially and writes
a comparison table.

```bash
python main.py                              # all five datasets (cuda by default)
python main.py --datasets bace bbbp tox21   # a subset
python main.py --best-config dual_kd_gnn/optuna/tox21_xkd/best_config.json  # same tuned HPs for all
```

Missing dataset files are skipped with a download hint. A summary table is saved
to `results/artifacts/benchmark_summary.csv`.

---

## 4. Hyperparameter tuning (Optuna)

[dual_kd_gnn/tune_optuna.py](dual_kd_gnn/tune_optuna.py) tunes on one dataset.
Use a dataset-specific `--study-name` so studies do not collide.

```bash
python dual_kd_gnn/tune_optuna.py --dataset bbbp  --study-name bbbp_xkd  --n-trials 30
python dual_kd_gnn/tune_optuna.py --dataset tox21 --study-name tox21_xkd --n-trials 50
python dual_kd_gnn/tune_optuna.py --dataset sider --study-name sider_xkd --n-trials 40 \
    --gcn-pretrain-epochs 150 --transformer-epochs 150 --patience 10
```

Study artifacts (SQLite DB, `best_config.json`, `trials.csv`) are saved under
`dual_kd_gnn/optuna/<study-name>/`.

Replay the best config to train + evaluate on the test split (saves weights and
plots):

```bash
python dual_kd_gnn/tune_optuna.py --replay-best dual_kd_gnn/optuna/bbbp_xkd/best_config.json
```

---

## 5. Aggregate results

[results/main.py](results/main.py) scans `dual_kd_gnn/runs/*` and writes the
cross-dataset comparison table and plots to `results/artifacts/`.

```bash
python results/main.py
```

Outputs: `all_metrics.csv`, `results_table.csv`, `results_table.md`,
`val_auc_curves.png`, `test_auc_by_dataset.png`.

---

---

## 6. Multi-seed runs (mean ± std AUROC)

JCIM reviewers typically reject single-seed results. Use `--seeds` to run N seeds
and print **mean ± std** automatically. Per-seed artifacts land in
`dual_kd_gnn/runs/<dataset>_seed<N>/`.

```bash
# Single dataset, 5 seeds
python dual_kd_gnn/main.py --dataset bbbp --seeds 42 0 1 2 3

# Batch benchmark, 5 seeds
python main.py --seeds 42 0 1 2 3

# Subset of datasets, 3 seeds
python main.py --datasets bace bbbp tox21 --seeds 42 0 1
```

`--seeds` overrides `--seed`. When a single value is provided the behaviour is
identical to `--seed`.

---

## 7. Ablation experiments (JCIM)

All five ablations are single-flag changes on top of the normal training command.
**Ablation results are saved separately** under `ablation/runs/<name>/` and never
overwrite the main `dual_kd_gnn/runs/` artifacts.

`--ablation-name` is **required** to route results to the ablation directory.
Without it the command behaves identically to a normal training run.

| ID | Claim verified | Extra flag(s) | Suggested `--ablation-name` |
|----|---------------|---------------|-----------------------------|
| A1 | 3D physical features contribute | `--no-phys-branch` | `a1_no_phys` |
| A2 | Cross-modal InfoNCE (λ₂) contributes | `--cross-distill-weight 0` | `a2_no_infonce` |
| A3 | Intra-modal MSE KD (λ₁) contributes | `--distill-weight 0` | `a3_no_mse_kd` |
| A4 | Shared codebook (U_k) contributes | `--ih-num-prototypes 0` | `a4_no_codebook` |
| A5 | Interaction tensor head (ITH) contributes | `--ih-rank 0` | `a5_linear_head` |

Each run directory (`ablation/runs/<name>/<dataset>/`) contains:
- `metrics.json` — `test_roc_auc`, `best_val_auc`, `seed`, `ablation_name`,
  `ablation_settings` (all key flags and their values)
- `run_metadata.json` — full hparams and model_kwargs
- `training_log.csv`, `training_curves.png`, `model_weights.pt`

### A1 — No physical branch (x_phys = zeros)

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --no-phys-branch --ablation-name a1_no_phys --seeds 42 0 1 2 3

python main.py \
    --no-phys-branch --ablation-name a1_no_phys --seeds 42 0 1 2 3
```

### A2 — No cross-modal InfoNCE (λ₂ = 0)

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --cross-distill-weight 0 --ablation-name a2_no_infonce --seeds 42 0 1 2 3

python main.py \
    --cross-distill-weight 0 --ablation-name a2_no_infonce --seeds 42 0 1 2 3
```

### A3 — No intra-modal MSE KD (λ₁ = 0)

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --distill-weight 0 --ablation-name a3_no_mse_kd --seeds 42 0 1 2 3

python main.py \
    --distill-weight 0 --ablation-name a3_no_mse_kd --seeds 42 0 1 2 3
```

### A4 — No codebook / per-class U_k (M = 0)

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --ih-num-prototypes 0 --ablation-name a4_no_codebook --seeds 42 0 1 2 3

python main.py \
    --ih-num-prototypes 0 --ablation-name a4_no_codebook --seeds 42 0 1 2 3
```

### A5 — No interaction tensor head (linear head only)

`--ih-rank 0` replaces the interaction tensor with a pure linear classifier.

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --ih-rank 0 --ablation-name a5_linear_head --seeds 42 0 1 2 3

python main.py \
    --ih-rank 0 --ablation-name a5_linear_head --seeds 42 0 1 2 3
```

### Full ablation sweep (all datasets, all 5 ablations)

```bash
python main.py --no-phys-branch      --ablation-name a1_no_phys    --seeds 42 0 1 2 3
python main.py --cross-distill-weight 0 --ablation-name a2_no_infonce --seeds 42 0 1 2 3
python main.py --distill-weight 0    --ablation-name a3_no_mse_kd  --seeds 42 0 1 2 3
python main.py --ih-num-prototypes 0 --ablation-name a4_no_codebook --seeds 42 0 1 2 3
python main.py --ih-rank 0           --ablation-name a5_linear_head --seeds 42 0 1 2 3
```

---

## 8. Aggregate ablation results

After all ablation runs finish, generate the summary CSV
(`ablation/ablation_summary.csv`):

```bash
python ablation/main.py
```

The script scans every `ablation/runs/<ablation_name>/<dataset>/metrics.json`,
groups results by **(ablation × dataset)**, and writes one row per combination
with the following columns:

| Column | Description |
|--------|-------------|
| `ablation` | Ablation label passed via `--ablation-name` |
| `dataset` | Dataset name (seed suffix stripped) |
| `n_seeds` | Number of seed runs found |
| `seeds` | List of seeds used |
| `mean_test_roc_auc` | Mean test AUROC across seeds |
| `std_test_roc_auc` | Sample std (ddof=1) of test AUROC |
| `mean_best_val_auc` | Mean best validation AUROC |
| `per_seed_aucs` | Per-seed AUROC values |
| `zero_phys_branch` | A1 setting |
| `distill_weight` | A3 setting |
| `cross_distill_weight` | A2 setting |
| `ih_rank` | A5 setting |
| `ih_num_prototypes` | A4 setting |

---

## Output layout

```
dual_kd_gnn/runs/<dataset>/
├── metrics.json          # best val AUC, test ROC-AUC, #params, runtime, target columns
├── run_metadata.json     # hparams, model_kwargs, data path, device
├── training_log.csv      # per-epoch train/val loss and ROC-AUC
├── model_weights.pt      # best-epoch model state_dict
└── training_curves.png   # loss + ROC-AUC curves

dual_kd_gnn/optuna/<study-name>/   # Optuna study DB, best_config.json, trials.csv
results/artifacts/                 # cross-dataset benchmark tables + plots
```

Each dataset gets its own run directory, so weights, logs, plots, and result
tables never overwrite one another.
