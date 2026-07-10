"""Compute Student-t 95% CIs from per-seed metrics.json files.

Automatically discovers all available seeds in each (condition, dataset) cell,
so it works transparently with n=5, n=15, or any other seed count. Aggregates
into three CSVs under results/artifacts/revision/:

  ablation_summary_with_ci.csv   — 30 rows (6 conditions × 5 datasets)
  random_split_summary_with_ci.csv — 3 rows (SIDER/Tox21/ClinTox random split)
  full_model_only_summary.csv    — 5 rows (full_model condition, scaffold split)

Usage on server (after seed_expansion.py completes):
  conda activate dualgnn
  python scripts/compute_ci.py

The script prints a headline table showing the new n and CI half-widths so you
can immediately verify the expansion increased statistical power.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "revision"
ABLATIONS = ["full_model", "a1_no_phys", "a2_no_infonce", "a3_no_mse_kd",
             "a4_no_codebook", "a5_linear_head"]
DATASETS = ["bace", "bbbp", "clintox", "sider", "tox21"]
DATASETS_RANDOM = ["sider", "tox21", "clintox"]


def collect_seeds(condition: str, dataset: str, base: Path) -> dict[int, float]:
    """Discover all seed folders for one (condition, dataset) cell."""
    cond_dir = base / condition
    if not cond_dir.exists():
        return {}
    pattern = re.compile(rf"^{re.escape(dataset)}_seed(\d+)$")
    out: dict[int, float] = {}
    for run_dir in cond_dir.iterdir():
        m = pattern.match(run_dir.name)
        if not m:
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = data.get("test_roc_auc")
        if auc is not None:
            out[int(m.group(1))] = float(auc)
    return out


def student_t_ci(aucs: list[float], alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Return (mean, std_ddof1, ci_lower, ci_upper, margin)."""
    arr = np.array(aucs, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0, mean, mean, 0.0
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = float(t_crit * se)
    return mean, sd, mean - margin, mean + margin, margin


def build_summary(scope: str) -> pd.DataFrame:
    """scope in {'ablation', 'random'}."""
    rows = []
    base = PROJECT_ROOT / "ablation" / "runs"
    if scope == "ablation":
        for condition in ABLATIONS:
            for dataset in DATASETS:
                aucs = list(collect_seeds(condition, dataset, base).values())
                if not aucs:
                    print(f"  [warn] {condition} x {dataset}: no seeds found")
                    continue
                mean, sd, lo, hi, margin = student_t_ci(aucs)
                rows.append({
                    "ablation": condition,
                    "dataset": dataset,
                    "n_seeds": len(aucs),
                    "mean_test_roc_auc": round(mean, 4),
                    "std_test_roc_auc": round(sd, 4),
                    "ci95_lower": round(lo, 4),
                    "ci95_upper": round(hi, 4),
                    "ci95_margin": round(margin, 4),
                    "per_seed_aucs": str([round(a, 4) for a in sorted(aucs, reverse=True)]),
                })
    elif scope == "random":
        for dataset in DATASETS_RANDOM:
            aucs = list(collect_seeds("full_model_random", dataset, base).values())
            if not aucs:
                print(f"  [warn] full_model_random x {dataset}: no seeds found")
                continue
            mean, sd, lo, hi, margin = student_t_ci(aucs)
            rows.append({
                "dataset": dataset,
                "split_protocol": "random_80_10_10",
                "n_seeds": len(aucs),
                "mean_test_roc_auc": round(mean, 4),
                "std_test_roc_auc": round(sd, 4),
                "ci95_lower": round(lo, 4),
                "ci95_upper": round(hi, 4),
                "ci95_margin": round(margin, 4),
                "per_seed_aucs": str([round(a, 4) for a in sorted(aucs, reverse=True)]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ablation_df = build_summary("ablation")
    random_df = build_summary("random")

    ablation_df.to_csv(OUT_DIR / "ablation_summary_with_ci.csv", index=False)
    random_df.to_csv(OUT_DIR / "random_split_summary_with_ci.csv", index=False)

    print("\n=== ablation_summary_with_ci.csv (full_model rows only) ===")
    full_rows = ablation_df[ablation_df["ablation"] == "full_model"][
        ["dataset", "n_seeds", "mean_test_roc_auc", "std_test_roc_auc",
         "ci95_lower", "ci95_upper", "ci95_margin"]
    ]
    print(full_rows.to_string(index=False))

    print("\n=== random_split_summary_with_ci.csv ===")
    print(random_df[["dataset", "n_seeds", "mean_test_roc_auc", "std_test_roc_auc",
                     "ci95_lower", "ci95_upper", "ci95_margin"]].to_string(index=False))

    print(f"\nSaved:")
    print(f"  {OUT_DIR/'ablation_summary_with_ci.csv'}")
    print(f"  {OUT_DIR/'random_split_summary_with_ci.csv'}")


if __name__ == "__main__":
    main()
