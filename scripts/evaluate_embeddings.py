"""Evaluate embedding quality and batch effects for GEMDiff datasets.

This script computes a set of sanity metrics on top of a low-dimensional
embedding (UMAP by default) derived from the gene expression matrices. It reports
label separability (kNN accuracy and AUROC/AUPRC) and quantifies batch imprint
with a silhouette score. Optionally, simple batch-correction strategies can be
benchmarked side by side.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
import umap


try:
    from neuroCombat import neuroCombat

    _COMBAT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _COMBAT_AVAILABLE = False

try:  # Harmony integration (optional)
    import harmonypy as hm

    _HARMONY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _HARMONY_AVAILABLE = False

try:  # Mutual Nearest Neighbours (optional)
    from mnnpy import mnn_correct

    _MNN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _MNN_AVAILABLE = False


@dataclass
class DatasetBundle:
    train_matrix: pd.DataFrame
    test_matrix: pd.DataFrame
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_batches: np.ndarray
    test_batches: np.ndarray
    label_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding sanity checks for GEMDiff datasets.")
    parser.add_argument("--train", required=True, help="Path to the training GEM (samples rows x genes columns).")
    parser.add_argument("--train-labels", required=True, help="Path to training labels file (sample\tlabel).")
    parser.add_argument("--test", required=False, help="Path to the test GEM. If omitted, a train split is used.")
    parser.add_argument("--test-labels", required=False, help="Path to the test labels file.")
    parser.add_argument(
        "--metadata",
        required=False,
        help="Optional metadata TSV including sample_id, split, label, and batch columns.",
    )
    parser.add_argument(
        "--batch-column",
        default="batch",
        help="Column in metadata that identifies the batch/center (default: %(default)s).",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column in metadata that contains the class label (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/metrics",
        help="Directory where plots and summary tables will be stored.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for UMAP and model selection.")
    parser.add_argument("--umap-components", type=int, default=2, help="Number of dimensions for the UMAP embedding.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of neighbors for the kNN classifier.")
    parser.add_argument(
        "--corrections",
        nargs="*",
        default=None,
        choices=["none", "batch_center", "combat", "harmony", "mnn"],
        help="Batch-correction strategies to evaluate. Supported: none, batch_center, combat, harmony, mnn.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="If explicit test data is not provided, fraction of the training set used for held-out metrics.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Name of the positive class to use for AUROC/AUPRC when exactly two labels are present.",
    )
    parser.add_argument(
        "--curve-plots",
        action="store_true",
        help="Save ROC and PR curve images for binary tasks.",
    )
    return parser.parse_args()


def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.astype(np.float32)


def load_labels(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", header=None, names=["sample_id", "label"])
    return df.set_index("sample_id")["label"]


def align_by_index(matrix: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    common = matrix.index.intersection(labels.index)
    if common.empty:
        raise ValueError("No overlapping samples between matrix and labels.")
    matrix = matrix.loc[common]
    labels = labels.loc[common]
    label_names = sorted(labels.unique())
    label_encoder = {name: idx for idx, name in enumerate(label_names)}
    encoded = labels.map(label_encoder).astype(int).to_numpy()
    return matrix, encoded, label_names


def load_metadata(path: Optional[str], index: Iterable[str], label_column: str, batch_column: str) -> Tuple[np.ndarray, np.ndarray]:
    index_list = list(index)
    if path is None:
        labels = np.array(["unknown"] * len(index_list))
        batches = np.array(["unknown"] * len(index_list))
        return labels, batches

    df = pd.read_csv(path, sep="\t")
    df = df.set_index("sample_id")
    missing = set(index_list) - set(df.index)
    if missing:
        raise ValueError(f"Metadata missing {len(missing)} samples: {sorted(list(missing))[:5]}...")
    meta = df.loc[index_list]
    labels = meta[label_column].astype(str).to_numpy()
    if batch_column not in meta.columns:
        batches = np.array(["unknown"] * len(labels))
    else:
        batches = meta[batch_column].fillna("unknown").astype(str).to_numpy()
    return labels, batches


def prepare_dataset(args: argparse.Namespace) -> DatasetBundle:
    train_matrix = load_matrix(args.train)
    train_labels_raw = load_labels(args.train_labels)
    train_matrix, train_labels, label_names = align_by_index(train_matrix, train_labels_raw)

    if args.test and args.test_labels:
        test_matrix = load_matrix(args.test)
        test_labels_raw = load_labels(args.test_labels)
        test_matrix, test_labels, _ = align_by_index(test_matrix, test_labels_raw)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            train_matrix,
            train_labels,
            test_size=args.test_size,
            stratify=train_labels,
            random_state=args.seed,
        )
        train_matrix, test_matrix = x_train, x_test
        train_labels, test_labels = y_train, y_test

    train_meta_labels, train_batches = load_metadata(
        args.metadata,
        train_matrix.index,
        args.label_column,
        args.batch_column,
    )
    test_meta_labels, test_batches = load_metadata(
        args.metadata,
        test_matrix.index,
        args.label_column,
        args.batch_column,
    )

    train_batches = np.array(train_batches)
    test_batches = np.array(test_batches)

    return DatasetBundle(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        train_labels=train_labels,
        test_labels=test_labels,
        train_batches=train_batches,
        test_batches=test_batches,
        label_names=label_names,
    )


def apply_batch_correction(
    bundle: DatasetBundle,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if method == "none":
        return bundle.train_matrix, bundle.test_matrix

    combined = pd.concat([bundle.train_matrix, bundle.test_matrix], axis=0)
    combined_batches = np.concatenate([bundle.train_batches, bundle.test_batches])
    batch_series = pd.Series(combined_batches, index=combined.index, dtype="object")

    if method == "batch_center":
        train_batches = pd.Series(
            bundle.train_batches, index=bundle.train_matrix.index, dtype="object"
        )
        train_means = bundle.train_matrix.groupby(train_batches).transform("mean")
        corrected_train = bundle.train_matrix - train_means

        global_mean = bundle.train_matrix.mean(axis=0)
        means_by_batch = {
            batch: bundle.train_matrix.loc[train_batches == batch].mean(axis=0)
            for batch in np.unique(bundle.train_batches)
        }

        corrected_test = bundle.test_matrix.copy()
        for idx, batch in zip(corrected_test.index, bundle.test_batches):
            mu = means_by_batch.get(batch, global_mean)
            corrected_test.loc[idx] = corrected_test.loc[idx] - mu

        return corrected_train, corrected_test
    elif method == "combat":
        if not _COMBAT_AVAILABLE:
            raise RuntimeError("neuroCombat is not installed; cannot run ComBat correction.")
        covars = pd.DataFrame({"batch": combined_batches}, index=combined.index)
        combat_res = neuroCombat(
            dat=combined.to_numpy().T,
            covars=covars,
            batch_col="batch",
            categorical_cols=None,
            continuous_cols=None,
            parametric=True,
        )
        corrected = pd.DataFrame(
            combat_res["data"].T,
            index=combined.index,
            columns=combined.columns,
        )
        corrected = corrected.replace([np.inf, -np.inf], np.nan)
        corrected = corrected.where(~corrected.isna(), combined)
    elif method == "harmony":
        if not _HARMONY_AVAILABLE:
            raise RuntimeError(
                "harmonypy is not installed; install it via `pip install harmonypy` to enable Harmony correction."
            )
        n_components = min(50, combined.shape[1], max(2, combined.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=0)
        pcs = pca.fit_transform(combined)
        meta = pd.DataFrame({"batch": combined_batches}, index=combined.index)
        harmony_out = hm.run_harmony(pcs.T, meta, "batch")
        corrected_pcs = harmony_out.Z_corr.T
        corrected = pd.DataFrame(
            corrected_pcs,
            index=combined.index,
            columns=[f"harmony_{i}" for i in range(corrected_pcs.shape[1])],
        )
    elif method == "mnn":
        if not _MNN_AVAILABLE:
            raise RuntimeError(
                "mnnpy is not installed; install it via `pip install mnnpy` (and dependencies) before using MNN correction."
            )
        raise RuntimeError(
            "MNN correction hook not yet implemented. After installing mnnpy, add your preferred implementation to `apply_batch_correction` or remove 'mnn' from --corrections."
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported correction method: {method}")

    corrected_train = corrected.loc[bundle.train_matrix.index]
    corrected_test = corrected.loc[bundle.test_matrix.index]
    return corrected_train, corrected_test



def fit_umap(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    reducer = umap.UMAP(
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
        metric="euclidean",
    )
    reducer.fit(train)
    return reducer.transform(train), reducer.transform(test)


def evaluate_label_metrics(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
    label_names: List[str],
) -> Tuple[Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray, str]]]:
    # kNN accuracy
    knn = KNeighborsClassifier(n_neighbors=args.knn_k)
    knn.fit(train_emb, train_labels)
    knn_preds = knn.predict(test_emb)
    knn_accuracy = accuracy_score(test_labels, knn_preds)

    # Logistic regression for AUROC/AUPRC
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(train_emb, train_labels)
    probs = clf.predict_proba(test_emb)

    curve_payload: Optional[Tuple[np.ndarray, np.ndarray, str]] = None

    if len(label_names) == 2:
        positive_label = args.positive_label or label_names[1]
        if positive_label not in label_names:
            print(
                f"[WARN] Requested positive label '{positive_label}' not in labels; defaulting to '{label_names[1]}'"
            )
            positive_label = label_names[1]
        pos_index = label_names.index(positive_label)
        binary_targets = (test_labels == pos_index).astype(int)
        positive_scores = probs[:, pos_index]
        roc_auc = roc_auc_score(binary_targets, positive_scores)
        auprc = average_precision_score(binary_targets, positive_scores)
        curve_payload = (binary_targets, positive_scores, positive_label)
    else:
        lb = LabelBinarizer().fit(range(len(label_names)))
        test_bin = lb.transform(test_labels)
        roc_auc = roc_auc_score(test_bin, probs, multi_class="ovr")
        auprc = average_precision_score(test_bin, probs, average="macro")

    metrics = {
        "knn_accuracy": knn_accuracy,
        "roc_auc": roc_auc,
        "auprc": auprc,
    }
    return metrics, curve_payload


def compute_silhouette(embeddings: np.ndarray, batches: np.ndarray) -> float:
    unique_batches = np.unique(batches)
    if unique_batches.size < 2:
        return float("nan")
    if embeddings.shape[0] < unique_batches.size:
        return float("nan")
    try:
        return float(silhouette_score(embeddings, batches, metric="euclidean"))
    except ValueError:
        return float("nan")


def save_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    out_path: Path,
    title: str,
    cmap: Optional[str] = "tab10",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    labels_array = np.asarray(labels)
    if labels_array.dtype.kind not in {"i", "u", "f"}:
        codes, uniques = pd.factorize(labels_array)
    else:
        uniques = np.unique(labels_array)
        codes = labels_array

    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1] if embeddings.shape[1] > 1 else np.zeros_like(embeddings[:, 0]),
        c=codes,
        cmap=cmap,
        alpha=0.8,
        s=14,
        edgecolors="none",
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    if embeddings.shape[1] > 1:
        plt.ylabel("UMAP-2")
    else:
        plt.ylabel("UMAP-2 (zero)")
    if labels_array.dtype.kind in {"i", "u", "f"}:
        label_lookup = {idx: label_names[idx] for idx in range(len(label_names))}
        legend_labels = [label_lookup.get(int(val), str(val)) for val in np.unique(codes)]
    else:
        legend_labels = uniques
    handles, _ = scatter.legend_elements()
    plt.legend(handles, legend_labels, title="Legend", loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_curves(
    curve_payload: Tuple[np.ndarray, np.ndarray, str],
    method: str,
    output_dir: Path,
) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    targets, scores, positive_label = curve_payload
    output_dir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(
        targets,
        scores,
        name=f"{positive_label} vs rest",
    )
    plt.title(f"ROC Curve — {method}")
    plt.savefig(output_dir / f"roc_{method}.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(
        targets,
        scores,
        name=f"{positive_label} vs rest",
    )
    plt.title(f"PR Curve — {method}")
    plt.savefig(output_dir / f"pr_{method}.png")
    plt.close()


def main() -> None:
    args = parse_args()
    corrections = args.corrections
    if corrections is None:
        corrections = ["none", "batch_center"]
        if _COMBAT_AVAILABLE:
            corrections.append("combat")

    bundle = prepare_dataset(args)
    print("Label names (encoded):", dict(enumerate(bundle.label_names)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []

    combined_batches = np.concatenate([bundle.train_batches, bundle.test_batches])

    for method in corrections:
        try:
            corrected_train, corrected_test = apply_batch_correction(bundle, method)
        except RuntimeError as exc:
            print(f"[WARN] Skipping correction '{method}': {exc}")
            continue

        train_emb, test_emb = fit_umap(corrected_train.values, corrected_test.values, args)
        metrics, curve_payload = evaluate_label_metrics(
            train_emb,
            test_emb,
            bundle.train_labels,
            bundle.test_labels,
            args,
            bundle.label_names,
        )

        embedding_all = np.vstack([train_emb, test_emb])
        silhouette = compute_silhouette(embedding_all, combined_batches)
        metrics.update({
            "silhouette_batch": silhouette,
            "correction": method,
        })
        if len(bundle.label_names) == 2 and (metrics["roc_auc"] > 0.999 or metrics["auprc"] > 0.999):
            print(f"[WARN] {method} AUROC/AUPRC ~1.0; verify splits and leakage before trusting these scores.")
        results.append(metrics)

        # Save scatter plots for each correction
        save_scatter(
            embedding_all,
            np.concatenate([bundle.train_labels, bundle.test_labels]),
            bundle.label_names,
            output_dir / f"embedding_{method}_label.png",
            title=f"UMAP ({method}) — label",
        )
        save_scatter(
            embedding_all,
            combined_batches,
            [str(b) for b in np.unique(combined_batches)],
            output_dir / f"embedding_{method}_batch.png",
            title=f"UMAP ({method}) — batch",
        )

        if args.curve_plots and curve_payload is not None:
            save_curves(curve_payload, method, output_dir)

    if not results:
        raise RuntimeError("No metrics computed; all correction methods failed or were skipped.")

    metrics_df = pd.DataFrame(results)
    metrics_csv = output_dir / "sanity_metrics.csv"
    metrics_json = output_dir / "sanity_metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[DONE] Saved metrics to:")
    print(f"  - {metrics_csv}")
    print(f"  - {metrics_json}")
    print("[INFO] Plots: ")
    for method in metrics_df["correction"].unique():
        print(f"  - embedding_{method}_label.png")
        print(f"  - embedding_{method}_batch.png")


if __name__ == "__main__":
    main()
