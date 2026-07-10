"""Revision experiments triggered by Stage 3 peer review.

Three sub-tasks consolidated into one script for server transfer convenience:

  --task b1   MMFF94 conformer embedding failure rate per dataset (Reviewer R3).
              Counts molecules whose physical-branch features degenerate to zero.
              Fast (~5-10 minutes total).

  --task b2   Re-run full_model under random split (MoleculeNet original
              protocol) for SIDER, Tox21, ClinTox to enable apples-to-apples
              comparison with MLFGNN (Reviewer R2).
              Slow (15 runs at ~10-30 min each; ~3-6 GPU hours total).

  --task c1   Paired Wilcoxon signed-rank tests for every (ablation, dataset)
              cell using per-seed AUCs from ablation/runs/ (Reviewer R5).
              Fast (~30 seconds).

  --task all  Run B1, then B2, then C1, in sequence.

Usage on server:

  conda activate dualgnn
  python scripts/revision_experiments.py --task b1
  python scripts/revision_experiments.py --task b2
  python scripts/revision_experiments.py --task b2 --datasets sider          # subset
  python scripts/revision_experiments.py --task b2 --seeds 42                # one seed only
  python scripts/revision_experiments.py --task c1

  python scripts/revision_experiments.py --task all                          # everything

Outputs land in results/artifacts/revision/ and ablation/runs/full_model_random/.
The script is idempotent: B2 skips runs whose metrics.json already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import set_seed, get_device  # noqa: E402
from common.data import MoleculeDualDataset  # noqa: E402
from common.datasets import DATASETS, resolve_target_columns  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from dual_kd_gnn.model import DualDistillationModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATASETS_ALL = ["bace", "bbbp", "clintox", "sider", "tox21"]
DATASETS_RANDOM = ["sider", "tox21", "clintox"]  # MLFGNN uses random split for these
SEEDS = [0, 1, 2, 3, 42]
ABLATIONS = ["a1_no_phys", "a2_no_infonce", "a3_no_mse_kd", "a4_no_codebook", "a5_linear_head"]

OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "revision"
RANDOM_RUNS_DIR = PROJECT_ROOT / "ablation" / "runs" / "full_model_random"


# ============================================================================
# Task B1: MMFF94 conformer embedding failure rate per dataset (parallelized)
# ============================================================================

def _embed_one(smi: str) -> str:
    """Classify one SMILES as one of: 'success', 'invalid_smiles',
    'embed_fail', 'mmff_fail'. Module-level for multiprocessing pickling.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid_smiles"
    mol_h = Chem.AddHs(mol)
    try:
        embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    except Exception:
        embed_status = -1
    if embed_status < 0:
        return "embed_fail"
    try:
        mmff_status = AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        mmff_status = -1
    if mmff_status not in (0, 1):
        return "mmff_fail"
    try:
        _ = mol_h.GetConformer()
        return "success"
    except Exception:
        return "embed_fail"


def task_b1(args) -> None:
    from multiprocessing import Pool, cpu_count
    print("\n[Task B1] MMFF94 conformer embedding failure rate per dataset")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    n_workers = max(1, min(args.b1_workers, cpu_count()))
    print(f"  Using {n_workers} parallel workers")

    for ds_name in args.datasets:
        spec = DATASETS[ds_name]
        df = pd.read_csv(spec.data_path())
        smiles_list = df[spec.smiles_column].tolist()
        n_total = len(smiles_list)

        counts = {"success": 0, "invalid_smiles": 0, "embed_fail": 0, "mmff_fail": 0}
        t0 = time.time()
        progress_every = max(50, n_total // 20)  # 5% increments
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_embed_one, smiles_list, chunksize=8), start=1):
                counts[result] += 1
                if i % progress_every == 0 or i == n_total:
                    pct = 100.0 * i / n_total
                    print(f"    {ds_name}: {i}/{n_total} ({pct:5.1f}%)  elapsed={time.time()-t0:.0f}s", flush=True)

        elapsed = time.time() - t0
        n_total_fail = counts["invalid_smiles"] + counts["embed_fail"] + counts["mmff_fail"]
        failure_rate = n_total_fail / max(n_total, 1)
        row = {
            "dataset": ds_name,
            "n_total": n_total,
            "n_success": counts["success"],
            "n_invalid_smiles": counts["invalid_smiles"],
            "n_embed_fail": counts["embed_fail"],
            "n_mmff_fail": counts["mmff_fail"],
            "n_total_fail": n_total_fail,
            "failure_rate": round(failure_rate, 4),
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(
            f"  [{ds_name:10s}] n={n_total:5d}  success={counts['success']:5d}  "
            f"smiles_invalid={counts['invalid_smiles']:4d}  embed_fail={counts['embed_fail']:4d}  "
            f"mmff_fail={counts['mmff_fail']:4d}  fail_rate={failure_rate:.3f}  ({elapsed:.1f}s)"
        )

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "mmff_failure_rates.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[B1] Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================================
# Task B2: Re-run full_model under random split (MLFGNN-compatible protocol)
# ============================================================================

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


def load_best_config(dataset: str) -> dict:
    path = PROJECT_ROOT / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"best_config.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _train_single_random(dataset: str, seed: int, device: torch.device) -> dict:
    """Train one (dataset, seed) configuration under random split."""
    run_dir = RANDOM_RUNS_DIR / f"{dataset}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"  [skip] {dataset}_seed{seed} already exists")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    num_classes = len(target_columns)
    df = pd.read_csv(data_path)
    n = len(df)
    train_idx, val_idx, test_idx = random_split(n, 0.8, 0.1, seed)
    print(f"  random split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = MoleculeDualDataset(data_path, target_columns, train_idx, smiles_column=spec.smiles_column)
    val_ds = MoleculeDualDataset(data_path, target_columns, val_idx, smiles_column=spec.smiles_column)
    test_ds = MoleculeDualDataset(data_path, target_columns, test_idx, smiles_column=spec.smiles_column)

    best_config = load_best_config(dataset)
    model_kwargs = dict(best_config["model_kwargs"])
    hparams = dict(best_config["hparams"])

    model = DualDistillationModel(num_classes=num_classes, **model_kwargs)
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

    n_params = int(sum(p.numel() for p in model.parameters()))
    metrics = {
        "model_name": "Dual_KD_GNN_random_split",
        "model_slug": "dual_kd_gnn",
        "dataset_name": f"{dataset}_full_random_seed{seed}",
        "split_protocol": "random_80_10_10",
        "target_columns": target_columns,
        "num_targets": num_classes,
        "best_val_auc": float(trainer.best_val_auc),
        "best_epoch": int(trainer.best_epoch),
        "test_roc_auc": test_auc,
        "num_parameters": n_params,
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "ablation_name": "full_model_random",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"  done: val={metrics['best_val_auc']:.4f}  test={metrics['test_roc_auc']:.4f}  "
        f"({elapsed/60:.1f} min)"
    )
    return metrics


def task_b2(args) -> None:
    print("\n[Task B2] Re-run full_model under MoleculeNet random split (SIDER, Tox21, ClinTox)")
    device = get_device(args.device)
    print(f"Device: {device}")

    datasets = [d for d in args.datasets if d in DATASETS_RANDOM]
    if not datasets:
        print(f"  [warn] no random-split-applicable datasets in {args.datasets}; "
              f"valid choices: {DATASETS_RANDOM}")
        return

    all_metrics = []
    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        for seed in args.seeds:
            print(f"\n[{ds_name} seed={seed}]")
            try:
                m = _train_single_random(ds_name, seed, device)
                all_metrics.append(m)
            except Exception as e:
                print(f"  [error] {ds_name}_seed{seed}: {e}")
                continue

    if all_metrics:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        per_dataset = defaultdict(list)
        for m in all_metrics:
            per_dataset[m["dataset_name"].split("_full_random_seed")[0]].append(m["test_roc_auc"])
        for ds, aucs in per_dataset.items():
            arr = np.array(aucs, dtype=float)
            summary_rows.append({
                "dataset": ds,
                "split_protocol": "random_80_10_10",
                "n_seeds": len(arr),
                "mean_test_roc_auc": round(float(arr.mean()), 4),
                "std_test_roc_auc": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
                "per_seed_aucs": str([round(a, 4) for a in aucs]),
            })
        out_path = OUT_DIR / "random_split_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n[B2] Saved summary: {out_path.relative_to(PROJECT_ROOT)}")
        print(f"[B2] Per-run metrics under: {RANDOM_RUNS_DIR.relative_to(PROJECT_ROOT)}/")


# ============================================================================
# Task C1: Paired Wilcoxon signed-rank tests (ablation vs full_model per cell)
# ============================================================================

def _load_per_seed_aucs(condition: str, dataset: str) -> dict[int, float]:
    """Read test_roc_auc per seed from ablation/runs/<condition>/<dataset>_seed<N>/metrics.json.

    Automatically discovers ALL seed folders (not just the original 5) so n=15
    seed expansion runs are picked up without modifying this function.
    """
    import re
    base = PROJECT_ROOT / "ablation" / "runs" / condition
    out: dict[int, float] = {}
    if not base.exists():
        return out
    pattern = re.compile(rf"^{re.escape(dataset)}_seed(\d+)$")
    for run_dir in base.iterdir():
        m = pattern.match(run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = data.get("test_roc_auc")
        if auc is not None:
            out[seed] = float(auc)
    return out


def task_c1(args) -> None:
    print("\n[Task C1] Paired Wilcoxon signed-rank tests (ablation vs full_model)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ablation in ABLATIONS:
        for dataset in DATASETS_ALL:
            full = _load_per_seed_aucs("full_model", dataset)
            abl = _load_per_seed_aucs(ablation, dataset)
            shared_seeds = sorted(set(full.keys()) & set(abl.keys()))
            if len(shared_seeds) < 3:
                print(f"  [warn] {ablation} x {dataset}: only {len(shared_seeds)} paired seeds, skipping test")
                continue
            full_aucs = np.array([full[s] for s in shared_seeds], dtype=float)
            abl_aucs = np.array([abl[s] for s in shared_seeds], dtype=float)
            deltas = full_aucs - abl_aucs

            try:
                stat, pval = wilcoxon(full_aucs, abl_aucs, zero_method="wilcox", alternative="two-sided")
                stat = float(stat)
                pval = float(pval)
            except ValueError as e:
                # All-zero differences trigger ValueError in scipy wilcoxon
                stat = float("nan")
                pval = float("nan")
                print(f"  [warn] {ablation} x {dataset}: wilcoxon error: {e}")

            row = {
                "ablation": ablation,
                "dataset": dataset,
                "n_paired_seeds": len(shared_seeds),
                "seeds": str(shared_seeds),
                "mean_full_auc": round(float(full_aucs.mean()), 4),
                "mean_ablation_auc": round(float(abl_aucs.mean()), 4),
                "mean_delta_full_minus_ablation": round(float(deltas.mean()), 4),
                "median_delta": round(float(np.median(deltas)), 4),
                "wilcoxon_statistic": round(stat, 6) if not np.isnan(stat) else "nan",
                "p_value": round(pval, 6) if not np.isnan(pval) else "nan",
                "significant_at_0.05": (not np.isnan(pval)) and pval < 0.05,
                "significant_at_0.01": (not np.isnan(pval)) and pval < 0.01,
            }
            rows.append(row)
            sig05 = "*" if row["significant_at_0.05"] else " "
            sig01 = "**" if row["significant_at_0.01"] else "  "
            pstr = f"{pval:.4f}" if not np.isnan(pval) else "  nan "
            print(
                f"  {ablation:18s} x {dataset:8s}  "
                f"delta={row['mean_delta_full_minus_ablation']:+.4f}  p={pstr} {sig05}{sig01}"
            )

    out_path = OUT_DIR / "wilcoxon_tests.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[C1] Saved: {out_path.relative_to(PROJECT_ROOT)}")
    sig05_count = sum(1 for r in rows if r["significant_at_0.05"] is True)
    sig01_count = sum(1 for r in rows if r["significant_at_0.01"] is True)
    print(f"[C1] {len(rows)} tests total; {sig05_count} significant at p<0.05, {sig01_count} at p<0.01")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=["b1", "b2", "c1", "all"], required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL,
                        help="Datasets to process. Default: all 5. B2 ignores BACE and BBBP "
                             "(MLFGNN uses scaffold split for those).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds for B2. Default: all 5 (0,1,2,3,42).")
    parser.add_argument("--device", default=None,
                        help="Compute device for B2. Default: auto (cuda > mps > cpu).")
    parser.add_argument("--b1-workers", type=int, default=8,
                        help="Parallel workers for B1 MMFF embedding. Default: 8.")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Task: {args.task}")
    print(f"Datasets: {args.datasets}")
    if args.task in ("b2", "all"):
        print(f"Seeds: {args.seeds}")

    if args.task == "b1":
        task_b1(args)
    elif args.task == "b2":
        task_b2(args)
    elif args.task == "c1":
        task_c1(args)
    elif args.task == "all":
        task_b1(args)
        task_b2(args)
        task_c1(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
