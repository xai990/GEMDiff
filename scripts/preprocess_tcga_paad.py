"""Command-line utility to convert TCGA-PAAD expression and clinical data
into the gene expression matrices and label files expected by GEMDiff.

The script expects an expression matrix where rows correspond to genes and
columns correspond to samples (as distributed by GDC/TCGA), along with a
clinical table that contains the `sample` identifier and the
`sample_type.samples` column. The output mirrors the structure used by the
breast cancer example (samples as rows, genes as columns) and produces
separate train/test splits together with their label files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_MAP: Dict[str, str] = {
    "Primary Tumor": "tumor",
    "Solid Tissue Normal": "normal",
    "Metastatic": "tumor",
}


@dataclass(frozen=True)
class SplitResult:
    train_ids: List[str]
    test_ids: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GEMDiff-ready pancreatic cancer datasets from TCGA files.",
    )
    parser.add_argument(
        "--expression",
        required=True,
        help="Path to TCGA-PAAD expression matrix (genes x samples, TSV).",
    )
    parser.add_argument(
        "--clinical",
        required=True,
        help="Path to TCGA-PAAD clinical table (TSV).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/pancreatic",
        help="Directory where processed train/test files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="paad",
        help="Filename prefix for generated outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of samples per class placed in the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--strip-version",
        action="store_true",
        help="Remove the version suffix from Ensembl gene identifiers.",
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=1,
        help="Ensure at least this many samples per class end up in the test set when possible.",
    )
    parser.add_argument(
        "--batch-column",
        default=None,
        help=
        "Optional clinical column to use as the batch identifier for stratification and reporting.",
    )
    parser.add_argument(
        "--metadata-suffix",
        default="metadata.tsv",
        help="Filename suffix for the consolidated metadata table (default: %(default)s).",
    )
    return parser.parse_args()


def load_expression(path: Path, strip_version: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if strip_version:
        df.index = df.index.str.split(".").str[0]
    df = df[~df.index.duplicated(keep="first")]
    df = df.transpose()
    df.index.name = "sample"
    return df


def load_labels(path: Path, label_map: Dict[str, str], batch_column: str | None) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", dtype=str)
    if "sample" not in clinical.columns:
        raise ValueError("Expected a 'sample' column in the clinical file.")
    if "sample_type.samples" not in clinical.columns:
        raise ValueError("Expected a 'sample_type.samples' column in the clinical file.")

    columns = {"sample": "sample_id", "sample_type.samples": "sample_type"}
    select_cols = ["sample", "sample_type.samples"]
    if batch_column and batch_column in clinical.columns:
        columns[batch_column] = "batch"
        select_cols.append(batch_column)
    elif batch_column:
        raise ValueError(f"Requested batch column '{batch_column}' not found in clinical file.")

    labels = clinical[select_cols].rename(columns=columns)
    labels = labels.dropna(subset=["sample_id", "sample_type"])
    labels = labels.drop_duplicates(subset="sample_id", keep="first")
    labels["label"] = labels["sample_type"].map(label_map)
    labels = labels.dropna(subset=["label"])
    return labels


def align_expression(
    expression: pd.DataFrame, labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shared_samples = labels[labels["sample_id"].isin(expression.index)].copy()
    missing = set(expression.index) - set(shared_samples["sample_id"])
    if missing:
        print(f"[INFO] Dropping {len(missing)} expression samples without labels.")
    missing_labels = set(shared_samples["sample_id"]) - set(expression.index)
    if missing_labels:
        print(f"[INFO] Dropping {len(missing_labels)} clinical records without expression data.")

    shared_samples = shared_samples[shared_samples["sample_id"].isin(expression.index)]
    shared_samples = shared_samples.sort_values("sample_id")
    filtered_expression = expression.loc[shared_samples["sample_id"].values].copy()
    return filtered_expression, shared_samples


def stratified_split(
    labels: pd.DataFrame,
    train_fraction: float,
    seed: int,
    min_test_per_class: int,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    train_ids: List[str] = []
    test_ids: List[str] = []

    if "batch" in labels.columns:
        batch_series = labels["batch"].fillna("__missing_batch__")
    else:
        batch_series = pd.Series(
            ["__no_batch__"] * len(labels), index=labels.index, dtype="object"
        )

    for _, group in labels.groupby(["label", batch_series]):
        sample_ids = group["sample_id"].to_numpy()
        if sample_ids.size == 0:
            continue
        permuted = rng.permutation(sample_ids)
        min_test = min(min_test_per_class, sample_ids.size - 1)
        tentative_train = int(np.floor(sample_ids.size * train_fraction))
        tentative_train = max(1, min(sample_ids.size - min_test, tentative_train))
        split = tentative_train if sample_ids.size > 1 else sample_ids.size

        train_ids.extend(permuted[:split])
        test_ids.extend(permuted[split:])

    # Preserve original ordering for deterministic downstream usage
    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)
    return SplitResult(train_ids=train_ids, test_ids=test_ids)


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=True, index_label="")


def write_labels(sample_ids: Sequence[str], labels: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"sample_id": sample_ids, "label": labels.loc[sample_ids].values})
    out.to_csv(path, sep="\t", header=False, index=False)


def main() -> None:
    args = parse_args()

    expression_path = Path(args.expression)
    clinical_path = Path(args.clinical)
    out_dir = Path(args.out_dir)

    expression = load_expression(expression_path, strip_version=args.strip_version)
    labels = load_labels(clinical_path, LABEL_MAP, args.batch_column)
    expression, labels = align_expression(expression, labels)

    if expression.empty:
        raise ValueError("No samples remain after aligning expression and clinical files.")

    split = stratified_split(labels, args.train_fraction, args.seed, args.min_test_per_class)
    labels = labels.set_index("sample_id")

    train_matrix = expression.loc[split.train_ids]
    test_matrix = expression.loc[split.test_ids]

    if train_matrix.empty:
        raise ValueError("Training split is empty. Adjust the split parameters.")
    if test_matrix.empty:
        raise ValueError("Test split is empty. Adjust the split parameters.")

    prefix = args.prefix
    train_matrix_path = out_dir / f"{prefix}_train_GEM.txt"
    test_matrix_path = out_dir / f"{prefix}_test_GEM.txt"
    train_label_path = out_dir / f"{prefix}_train_labels.txt"
    test_label_path = out_dir / f"{prefix}_test_labels.txt"

    write_matrix(train_matrix, train_matrix_path)
    write_matrix(test_matrix, test_matrix_path)
    label_series = labels["label"]
    write_labels(split.train_ids, label_series, train_label_path)
    write_labels(split.test_ids, label_series, test_label_path)

    metadata_columns = {"sample_id": [], "split": [], "label": [], "batch": []}
    for sample_id in split.train_ids:
        metadata_columns["sample_id"].append(sample_id)
        metadata_columns["split"].append("train")
        metadata_columns["label"].append(labels.loc[sample_id, "label"])
        metadata_columns["batch"].append(labels.loc[sample_id, "batch"] if "batch" in labels.columns else "NA")
    for sample_id in split.test_ids:
        metadata_columns["sample_id"].append(sample_id)
        metadata_columns["split"].append("test")
        metadata_columns["label"].append(labels.loc[sample_id, "label"])
        metadata_columns["batch"].append(labels.loc[sample_id, "batch"] if "batch" in labels.columns else "NA")

    metadata = pd.DataFrame(metadata_columns)
    metadata_path = out_dir / f"{prefix}_{args.metadata_suffix}"
    metadata.to_csv(metadata_path, sep="\t", index=False)

    print("[DONE] Wrote datasets to:")
    for path in (train_matrix_path, train_label_path, test_matrix_path, test_label_path):
        print(f"  - {path}")
    print(f"  - {metadata_path}")
    print("[SUMMARY] Samples per split:")
    print(f"  Train: {train_matrix.shape[0]} | Test: {test_matrix.shape[0]}")
    print("[SUMMARY] Feature count:")
    print(f"  Genes: {train_matrix.shape[1]}")


if __name__ == "__main__":
    main()
