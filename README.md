# DiKAT: Dichotomous Knowledge Alignment via Tensors for Molecular Property Prediction

Reference implementation for the paper *"DiKAT: Dichotomous Knowledge Alignment via Tensors for Molecular Property Prediction"* (Lee and Kim, submitted to JCIM).

DiKAT is a dual-branch graph neural network that decomposes molecular representation into a topology-aware chemical branch and a geometry-aware physical branch (MMFF94-derived 3D coordinates + van der Waals radii). The two branches are co-trained under a two-stage protocol combining EMA self-distillation and a cross-modal InfoNCE term, followed by Transformer fusion over frozen GCN encoders. Predictions come from a codebook-shared interaction-tensor classifier head with Gumbel-softmax prototype routing.

## Environment

- Python 3.10 - 3.12
- PyTorch (CUDA recommended)
- PyTorch Geometric
- RDKit (install via `conda` if pip fails)
- Optuna, NumPy, pandas, scikit-learn, matplotlib

```bash
pip install -r requirements.txt
```

## Data

MoleculeNet classification datasets (BACE, BBBP, ClinTox, SIDER, Tox21) are obtained from the DeepChem distribution and cached under `data/`:

```bash
python scripts/download_data.py
```

## Reproducing the reported numbers

All main-benchmark and ablation numbers reported in the paper use 15 random seeds `{0, 1, 2, 3, 42, 100-109}`. Per-dataset best hyperparameters are stored under `dual_kd_gnn/optuna/<dataset>_xkd/best_config.json`.

### Table 1 — Main benchmark (15 seeds, label-aware scaffold split)

```bash
python scripts/seed_expansion.py --datasets bace bbbp clintox sider tox21 \
    --split scaffold --seeds 0 1 2 3 42 100 101 102 103 104 105 106 107 108 109
```

Per-run outputs land in `ablation/runs/full_model/<dataset>_seed<N>/metrics.json` and are aggregated by:

```bash
python scripts/compute_ci.py \
    --input_dir ablation/runs/full_model \
    --output results/artifacts/revision/ablation_summary_with_ci.csv
```

### Table 2 — Matched-protocol comparison (random split for SIDER/Tox21/ClinTox)

```bash
python scripts/seed_expansion.py --datasets clintox sider tox21 \
    --split random --seeds 0 1 2 3 42 100 101 102 103 104 105 106 107 108 109
```

Aggregate into `results/artifacts/revision/random_split_summary_with_ci.csv` with `compute_ci.py`.

### Ablation study (25 cells × 15 seeds)

```bash
python scripts/seed_expansion.py --datasets bace bbbp clintox sider tox21 \
    --split scaffold \
    --ablations a1_no_phys a2_no_infonce a3_no_mse_kd a4_no_codebook a5_linear_head \
    --seeds 0 1 2 3 42 100 101 102 103 104 105 106 107 108 109
```

Paired Wilcoxon signed-rank tests over the 15-seed per-cell AUCs:

```bash
python scripts/revision_experiments.py --mode wilcoxon
# writes results/artifacts/revision/wilcoxon_tests.csv
```

### Interpretability artifacts (Figure 5)

```bash
python scripts/prototype_analysis.py --dataset tox21
python scripts/prototype_analysis.py --dataset sider
```

Codebook assignment matrices and per-molecule top-K activation tables are written to `results/artifacts/prototypes/`.

### Cross-modal alignment trajectory (Figure 6)

```bash
python scripts/cross_modal_alignment_analysis.py --dataset tox21 --seed 42
```

## Repository layout

```
common/                   Data loading, scaffold splitting, featurisation
dual_kd_gnn/              Model, training, and Optuna tuning
  model.py                DualDistillationModel definition
  main.py                 Standalone training entry point
  tune_optuna.py          Optuna TPE + Hyperband search
  optuna/<dataset>_xkd/   Per-dataset best_config.json
ablation/                 Ablation runner + per-run outputs
scripts/                  Reproduction utilities
  seed_expansion.py       15-seed batch runner
  compute_ci.py           95% Student-t CI aggregator
  revision_experiments.py Wilcoxon + patience diagnostics
  prototype_analysis.py   Codebook interpretability
  cross_modal_alignment_analysis.py  Alignment trajectory
results/artifacts/        Aggregated CSVs and figures
```

## Citation

To be added on acceptance.

## License

MIT. See `LICENSE`.
