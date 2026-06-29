"""Prototype interpretability analysis for the Dual-KD-GNN classifier head.

Produces three artifact families per saved single-seed model:

  results/artifacts/prototypes/<dataset>_assignment_heatmap.png
      K (class) x M (prototype) soft-assignment alpha matrix from
      InteractionTensorHead.get_assignment_probabilities().

  results/artifacts/prototypes/<dataset>_prototype_norms.csv
      Per-prototype activation statistics over the test split:
      mean L2 norm of the projection z -> C_m^T z across molecules.

  results/artifacts/prototypes/<dataset>_top_molecules.csv
      Top-K (default 10) molecules per prototype ranked by activation norm,
      with their SMILES, Murcko scaffold SMILES, MACCS-key fingerprints,
      and ground-truth labels. Substructure analysis downstream consumes this.

Single-seed run weights are read from dual_kd_gnn/runs/<dataset>/. These were
trained from a tuned best_config; their hyperparameters are consistent with the
5-seed ablation full_model runs. Per the paper plan, this analysis is reported
as a "representative seed" study; 5-seed consistency is deferred to a Stage 4
revision response if reviewers ask.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import MoleculeDualDataset, scaffold_split  # noqa: E402
from common.datasets import DATASETS, resolve_target_columns  # noqa: E402
from dual_kd_gnn.model import DualDistillationModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEFAULT_DATASETS = ["bace", "bbbp", "clintox", "sider", "tox21"]
RUNS_DIR = PROJECT_ROOT / "dual_kd_gnn" / "runs"
OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "prototypes"


def load_model_for_dataset(dataset: str, device: torch.device) -> tuple[DualDistillationModel, dict, list[str]]:
    run_dir = RUNS_DIR / dataset
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    target_columns = list(metrics["target_columns"])

    model = DualDistillationModel(
        num_classes=len(target_columns),
        **metadata["model_kwargs"],
    ).to(device)

    state = torch.load(run_dir / "model_weights.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] state_dict mismatch (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model, metadata, target_columns


def get_assignment_matrix(model: DualDistillationModel) -> np.ndarray:
    if not model.classifier.use_codebook:
        return np.zeros((model.classifier.num_classes, 0))
    return model.classifier.get_assignment_probabilities().cpu().numpy()


def save_heatmap(alpha: np.ndarray, target_columns: list[str], out_path: Path, title: str) -> None:
    if alpha.size == 0:
        print(f"  [skip] no codebook for {title}")
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, alpha.shape[1] * 0.8), max(3, alpha.shape[0] * 0.35)))
    im = ax.imshow(alpha, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(alpha.shape[1]))
    ax.set_xticklabels([f"P{m}" for m in range(alpha.shape[1])], fontsize=8)
    ax.set_yticks(range(alpha.shape[0]))
    short_labels = [tc[:30] + ("..." if len(tc) > 30 else "") for tc in target_columns]
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Prototype index m")
    ax.set_ylabel("Class index k")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$\alpha_{k,m}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_test_split(dataset: str) -> tuple[MoleculeDualDataset, list[str]]:
    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    dataframe = pd.read_csv(data_path)
    _, _, test_idx = scaffold_split(
        dataframe,
        target_columns=target_columns,
        smiles_column=spec.smiles_column,
        seed=42,
    )
    ds = MoleculeDualDataset(
        data_path=data_path,
        target_columns=target_columns,
        indices=test_idx,
        smiles_column=spec.smiles_column,
    )
    return ds, target_columns


def compute_prototype_activations(
    model: DualDistillationModel,
    dataset: MoleculeDualDataset,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return per-molecule prototype activation norms; shape [N, M].

    Activation[n, m] = || C_m^T z_n ||_2 in the classifier's effective space.
    Higher = molecule n drives prototype m more strongly.
    """
    from torch_geometric.loader import DataLoader

    head = model.classifier
    if not head.use_codebook:
        return np.zeros((len(dataset), 0))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations: list[np.ndarray] = []

    codebook = head.codebook_u.detach()  # [M, d, r]
    proj_layer = head.proj  # optional Linear+ReLU projection to effective_dim

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            student_chem, student_phys = model._run_frozen_student_gcn(data)
            padded_c, padded_p, pad_mask = model._pad_dual_sequences(
                student_chem, student_phys, data.batch
            )
            padded_c = model.input_norm_c(padded_c)
            padded_p = model.input_norm_p(padded_p)
            attn_pad = pad_mask if device.type != "mps" else None
            phys_seq = model.phys_encoder(padded_p, attn_pad)
            chem_seq = model.chem_encoder(padded_c, attn_pad)
            fused = torch.cat([phys_seq, chem_seq], dim=-1)
            fused = fused.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            fused = model.concat_norm(fused)
            fused = fused.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            valid = (~pad_mask).unsqueeze(-1).type_as(fused)
            denom = valid.sum(dim=1).clamp_min(1.0)
            z = (fused * valid).sum(dim=1) / denom  # [B, 2*hidden]

            if proj_layer is not None:
                z = proj_layer(z)

            # y[b, m, r] = sum_d z[b, d] * codebook[m, d, r]
            y = torch.einsum("bd,mdr->bmr", z, codebook)
            norms = y.norm(dim=-1)  # [B, M]
            activations.append(norms.cpu().numpy())

    return np.concatenate(activations, axis=0)


def topk_molecules_per_prototype(
    activations: np.ndarray,
    smiles_list: list[str],
    labels: np.ndarray,
    target_columns: list[str],
    k: int = 10,
) -> pd.DataFrame:
    rows = []
    for m in range(activations.shape[1]):
        order = np.argsort(-activations[:, m])[:k]
        for rank, idx in enumerate(order, start=1):
            smi = smiles_list[idx]
            mol = Chem.MolFromSmiles(smi)
            scaffold = ""
            maccs_bits = ""
            if mol is not None:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                except Exception:
                    scaffold = ""
                try:
                    bits = MACCSkeys.GenMACCSKeys(mol)
                    on_bits = sorted(bit for bit in range(bits.GetNumBits()) if bits.GetBit(bit))
                    maccs_bits = ";".join(map(str, on_bits))
                except Exception:
                    maccs_bits = ""
            label_str = ";".join(
                f"{tc}={int(v) if not np.isnan(v) and v >= 0 else '?'}"
                for tc, v in zip(target_columns, labels[idx])
            )
            rows.append(
                {
                    "prototype": m,
                    "rank": rank,
                    "activation_norm": float(activations[idx, m]),
                    "molecule_index": int(idx),
                    "smiles": smi,
                    "murcko_scaffold": scaffold,
                    "maccs_on_bits": maccs_bits,
                    "labels": label_str,
                }
            )
    return pd.DataFrame(rows)


def run_dataset(dataset: str, device: torch.device) -> None:
    print(f"\n[{dataset}]")
    try:
        model, _, target_columns = load_model_for_dataset(dataset, device)
    except FileNotFoundError as e:
        print(f"  [skip] missing artifact: {e}")
        return

    alpha = get_assignment_matrix(model)
    print(f"  alpha shape: {alpha.shape}")

    heatmap_path = OUT_DIR / f"{dataset}_assignment_heatmap.png"
    save_heatmap(alpha, target_columns, heatmap_path, title=f"{dataset} prototype assignment")
    print(f"  saved heatmap: {heatmap_path.relative_to(PROJECT_ROOT)}")

    if alpha.size == 0:
        return  # no codebook; nothing more to do

    test_ds, _ = build_test_split(dataset)
    activations = compute_prototype_activations(model, test_ds, device)
    print(f"  activations shape: {activations.shape}")

    norm_stats = pd.DataFrame(
        {
            "prototype": np.arange(activations.shape[1]),
            "mean_norm": activations.mean(axis=0),
            "std_norm": activations.std(axis=0, ddof=1) if activations.shape[0] > 1 else 0.0,
            "max_norm": activations.max(axis=0),
        }
    )
    norm_stats.to_csv(OUT_DIR / f"{dataset}_prototype_norms.csv", index=False)
    print(f"  saved norms: {dataset}_prototype_norms.csv")

    smiles_list = test_ds.smiles
    labels = test_ds.labels.cpu().numpy()
    topk = topk_molecules_per_prototype(activations, smiles_list, labels, target_columns, k=10)
    topk.to_csv(OUT_DIR / f"{dataset}_top_molecules.csv", index=False)
    print(f"  saved top molecules: {dataset}_top_molecules.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--device", default=None, help="torch device override (cpu / cuda / mps)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Output dir: {OUT_DIR.relative_to(PROJECT_ROOT)}")

    for dataset in args.datasets:
        run_dataset(dataset, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
