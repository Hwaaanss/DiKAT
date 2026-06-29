"""Aggregate ablation results from ablation/runs/ into a summary CSV.

Directory layout expected:
  ablation/runs/<ablation_name>/<dataset_run>/metrics.json

<dataset_run> is either <dataset> (single seed) or <dataset>_seed<N> (multi-seed).
Each metrics.json must contain at minimum: test_roc_auc, dataset_name.
When produced by run_experiment with --ablation-name, it also contains:
  ablation_name, seed, ablation_settings (zero_phys_branch, distill_weight, ...).

Output: ablation/ablation_summary.csv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_RUNS_DIR = PROJECT_ROOT / "ablation" / "runs"
OUTPUT_PATH = PROJECT_ROOT / "ablation" / "ablation_summary.csv"

ABLATION_SETTING_KEYS = [
    "zero_phys_branch",
    "distill_weight",
    "cross_distill_weight",
    "ih_rank",
    "ih_num_prototypes",
]


def _strip_seed_suffix(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def _infer_seed(run_dir_name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", run_dir_name)
    return int(m.group(1)) if m else None


def collect_records() -> list[dict]:
    if not ABLATION_RUNS_DIR.exists():
        return []

    records: list[dict] = []
    for ablation_dir in sorted(ABLATION_RUNS_DIR.iterdir()):
        if not ablation_dir.is_dir():
            continue
        ablation_name = ablation_dir.name
        for run_dir in sorted(ablation_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            metadata_path = run_dir / "run_metadata.json"
            if not metrics_path.exists():
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else {}
            )

            base_dataset = _strip_seed_suffix(run_dir.name)
            auc = metrics.get("test_roc_auc")
            val_auc = metrics.get("best_val_auc")
            seed = metrics.get("seed") or _infer_seed(run_dir.name)

            # Ablation settings: prefer explicit field in metrics, fall back to metadata
            abl_settings = metrics.get("ablation_settings", {})
            if not abl_settings:
                hparams = metadata.get("hparams", {})
                mkw = metadata.get("model_kwargs", {})
                abl_settings = {
                    "zero_phys_branch": mkw.get("zero_phys_branch", False),
                    "distill_weight": hparams.get("distill_weight"),
                    "cross_distill_weight": hparams.get("cross_distill_weight"),
                    "ih_rank": mkw.get("ih_rank"),
                    "ih_num_prototypes": mkw.get("ih_num_prototypes"),
                }

            records.append(
                {
                    "ablation": metrics.get("ablation_name", ablation_name),
                    "dataset": base_dataset,
                    "seed": seed,
                    "test_roc_auc": auc,
                    "best_val_auc": val_auc,
                    **{k: abl_settings.get(k) for k in ABLATION_SETTING_KEYS},
                }
            )
    return records


def aggregate(records: list[dict]) -> pd.DataFrame:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["ablation"], r["dataset"])
        groups.setdefault(key, []).append(r)

    rows: list[dict] = []
    for (ablation, dataset), group in sorted(groups.items()):
        aucs = [g["test_roc_auc"] for g in group if g["test_roc_auc"] is not None]
        val_aucs = [g["best_val_auc"] for g in group if g["best_val_auc"] is not None]
        seeds = sorted(g["seed"] for g in group if g["seed"] is not None)

        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        mean_val = float(np.mean(val_aucs)) if val_aucs else float("nan")

        rep = group[0]
        rows.append(
            {
                "ablation": ablation,
                "dataset": dataset,
                "n_seeds": len(aucs),
                "seeds": str(seeds),
                "mean_test_roc_auc": round(mean_auc, 4),
                "std_test_roc_auc": round(std_auc, 4),
                "mean_best_val_auc": round(mean_val, 4),
                "per_seed_aucs": str([round(a, 4) for a in aucs]),
                **{k: rep.get(k) for k in ABLATION_SETTING_KEYS},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    records = collect_records()
    if not records:
        raise SystemExit(
            f"No ablation results found under {ABLATION_RUNS_DIR}.\n"
            "Run ablation experiments with --ablation-name first (see commands.md)."
        )

    df = aggregate(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Ablation summary  ({len(df)} rows across "
          f"{df['ablation'].nunique()} ablations, "
          f"{df['dataset'].nunique()} datasets)")
    print()
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
