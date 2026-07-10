"""Seed expansion to n=15 for statistical power (Reviewer Priority 1).

Extends the existing 5-seed ablation study (150 runs) with 10 additional seeds
across all 6 conditions and 5 datasets, bringing the total to 15 paired seeds
per (condition, dataset) cell. This raises the theoretical Wilcoxon two-sided
minimum p-value from 0.0625 (at n=5) to below 0.001 (at n=15), enabling
conventional significance conclusions on the ablation deltas.

Also runs the same 10 extra seeds on the full_model_random (matched-protocol)
runs to enable Student-t 95% CIs at n=15 on SIDER/Tox21/ClinTox.

Configuration
  New seeds        : {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
  Conditions       : full_model, a1_no_phys, a2_no_infonce, a3_no_mse_kd,
                     a4_no_codebook, a5_linear_head
  Datasets         : bace, bbbp, clintox, sider, tox21
  Total new runs   : 6 conditions x 5 datasets x 10 seeds  = 300 (scaffold)
                   + 1 condition  x 3 datasets x 10 seeds  =  30 (random split)
                   = 330 runs
  Estimated GPU    : ~5-10 days on a single A100 (Tox21 is the bottleneck)

Idempotency: existing metrics.json files are detected and skipped, so this
script is safe to restart after interruption. Each new run is written under
ablation/runs/<condition>/<dataset>_seed<N>/metrics.json in the same layout
the aggregation script already understands.

Usage on server
  conda activate dualgnn
  nohup python -u scripts/seed_expansion.py > seed_expansion.log 2>&1 &
  echo $! > seed_expansion.pid
  tail -f seed_expansion.log

  # Restart after interruption (idempotent):
  nohup python -u scripts/seed_expansion.py > seed_expansion.log 2>&1 &

  # Progress: count of newly completed runs (target: 330)
  find ablation/runs -name "metrics.json" -newer scripts/seed_expansion.py | wc -l

After completion
  1. Re-run ablation/main.py to regenerate ablation_summary.csv with n=15.
  2. Re-run scripts/revision_experiments.py --task c1  for Wilcoxon at n=15.
  3. Re-run the CI computation inline (see paper/draft/manuscript.tex §4.1.2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import set_seed, get_device  # noqa: E402
from common.data import MoleculeDualDataset, create_datasets, scaffold_split  # noqa: E402
from common.datasets import DATASETS, resolve_target_columns  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from dual_kd_gnn.model import DualDistillationModel as Model  # noqa: E402

# Configuration
NEW_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
DATASETS_ALL = ["bace", "bbbp", "clintox", "sider", "tox21"]
DATASETS_RANDOM = ["sider", "tox21", "clintox"]

# Per-ablation model_kwargs / hparams override lambdas.
# Applied on top of the tuned best_config for each dataset.
ABLATION_OVERRIDES = {
    "full_model":       lambda m, h: (m, h),
    "a1_no_phys":       lambda m, h: ({**m, "zero_phys_branch": True}, h),
    "a2_no_infonce":    lambda m, h: (m, {**h, "cross_distill_weight": 0.0}),
    "a3_no_mse_kd":     lambda m, h: (m, {**h, "distill_weight": 0.0}),
    "a4_no_codebook":   lambda m, h: ({**m, "ih_num_prototypes": 0}, h),
    "a5_linear_head":   lambda m, h: ({**m, "ih_rank": 0}, h),
}

RUNS_ROOT = PROJECT_ROOT / "ablation" / "runs"
RANDOM_RUNS_DIR = PROJECT_ROOT / "ablation" / "runs" / "full_model_random"


def load_best_config(dataset: str) -> dict:
    p = PROJECT_ROOT / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def random_split(n: int, train_size: float, val_size: float, seed: int):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    return (
        indices[:n_train].tolist(),
        indices[n_train:n_train + n_val].tolist(),
        indices[n_train + n_val:].tolist(),
    )


def train_one(condition: str, dataset: str, seed: int, split_mode: str, device: torch.device) -> dict:
    """Train and save one (condition, dataset, seed, split) cell. Idempotent."""
    if split_mode == "scaffold":
        run_dir = RUNS_ROOT / condition / f"{dataset}_seed{seed}"
    elif split_mode == "random":
        run_dir = RANDOM_RUNS_DIR / f"{dataset}_seed{seed}"
    else:
        raise ValueError(split_mode)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return {"status": "skipped", "path": str(metrics_path)}

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    num_classes = len(target_columns)

    if split_mode == "scaffold":
        train_ds, val_ds, test_ds = create_datasets(
            data_path=data_path,
            target_columns=target_columns,
            seed=seed,
            dual=True,
            smiles_column=spec.smiles_column,
        )
    else:
        df = pd.read_csv(data_path)
        train_idx, val_idx, test_idx = random_split(len(df), 0.8, 0.1, seed)
        train_ds = MoleculeDualDataset(data_path, target_columns, train_idx, smiles_column=spec.smiles_column)
        val_ds = MoleculeDualDataset(data_path, target_columns, val_idx, smiles_column=spec.smiles_column)
        test_ds = MoleculeDualDataset(data_path, target_columns, test_idx, smiles_column=spec.smiles_column)

    best_config = load_best_config(dataset)
    model_kwargs = dict(best_config["model_kwargs"])
    hparams = dict(best_config["hparams"])
    model_kwargs, hparams = ABLATION_OVERRIDES[condition](model_kwargs, hparams)

    model = Model(num_classes=num_classes, **model_kwargs)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        num_classes=num_classes,
        **hparams,
    )
    t0 = time.time()
    trainer.train()
    test_auc = float(trainer.evaluate(test_ds, batch_size=int(hparams["batch_size"])))
    elapsed = time.time() - t0

    metrics = {
        "ablation_name": condition,
        "dataset_name": f"{dataset}_{condition}_seed{seed}" if split_mode == "scaffold"
                        else f"{dataset}_full_random_seed{seed}",
        "split_protocol": "label_aware_scaffold_80_10_10" if split_mode == "scaffold" else "random_80_10_10",
        "target_columns": target_columns,
        "num_targets": num_classes,
        "best_val_auc": float(trainer.best_val_auc),
        "best_epoch": int(trainer.best_epoch),
        "test_roc_auc": test_auc,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return {"status": "done", "elapsed": elapsed, "test_auc": test_auc, "path": str(metrics_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=NEW_SEEDS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL, choices=DATASETS_ALL)
    parser.add_argument("--conditions", nargs="+",
                        default=list(ABLATION_OVERRIDES.keys()),
                        choices=list(ABLATION_OVERRIDES.keys()))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Skip scaffold-split runs (only do random-split).")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random-split full_model runs.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Seeds:  {args.seeds}")
    print(f"Conditions: {args.conditions}")
    print(f"Datasets:   {args.datasets}")

    total_target = 0
    if not args.skip_scaffold:
        total_target += len(args.conditions) * len(args.datasets) * len(args.seeds)
    if not args.skip_random:
        rand_ds = [d for d in args.datasets if d in DATASETS_RANDOM]
        total_target += len(rand_ds) * len(args.seeds)  # only full_model condition
    print(f"Target new runs (upper bound; idempotent skips reduce actual work): {total_target}\n")

    done = skipped = failed = 0
    if not args.skip_scaffold:
        for condition in args.conditions:
            for dataset in args.datasets:
                for seed in args.seeds:
                    tag = f"[{condition:16s}][{dataset:8s}][seed={seed:3d}]"
                    try:
                        result = train_one(condition, dataset, seed, "scaffold", device)
                    except Exception as e:
                        print(f"{tag} FAILED: {e}", flush=True)
                        failed += 1
                        continue
                    if result["status"] == "skipped":
                        skipped += 1
                        print(f"{tag} skip (exists)", flush=True)
                    else:
                        done += 1
                        print(f"{tag} done  auc={result['test_auc']:.4f}  ({result['elapsed']/60:.1f} min)  [{done+skipped}/{total_target}]", flush=True)

    if not args.skip_random:
        for dataset in [d for d in args.datasets if d in DATASETS_RANDOM]:
            for seed in args.seeds:
                tag = f"[full_model_random][{dataset:8s}][seed={seed:3d}]"
                try:
                    result = train_one("full_model", dataset, seed, "random", device)
                except Exception as e:
                    print(f"{tag} FAILED: {e}", flush=True)
                    failed += 1
                    continue
                if result["status"] == "skipped":
                    skipped += 1
                    print(f"{tag} skip (exists)", flush=True)
                else:
                    done += 1
                    print(f"{tag} done  auc={result['test_auc']:.4f}  ({result['elapsed']/60:.1f} min)  [{done+skipped}/{total_target}]", flush=True)

    print(f"\nSummary: done={done}, skipped={skipped}, failed={failed}, target={total_target}")
    if done > 0:
        print("\nNext steps:")
        print("  1. Regenerate ablation_summary.csv:")
        print("       python ablation/main.py")
        print("  2. Re-run Wilcoxon tests with n=15:")
        print("       python scripts/revision_experiments.py --task c1")
        print("  3. Copy results back to local for paper update.")


if __name__ == "__main__":
    main()
