"""End-to-end preprocessing and QC workflow for TCGA-style expression matrices.

Steps
-----
1. Load raw count and TPM matrices (genes as rows, samples as columns).
2. Harmonise gene IDs (drop version suffix), collapse duplicates, and filter
   low-expression genes (TPM ≥ threshold in ≥ proportion of samples).
3. Emit tidy matrices (samples × genes): counts, TPM, log2(TPM+1).
4. Parse clinical metadata, join derived QC metrics (library size, % zeros,
   MAD-based outlier flags, PCA/UMAP coordinates).
5. Generate PCA/UMAP plots coloured by label and batch.
6. Compute batch silhouette scores and variance explained tables.
7. Write a compact markdown QC report summarising findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TCGA expression data and perform QC.")
    parser.add_argument("--counts", required=True, help="Path to raw STAR counts TSV (genes × samples).")
    parser.add_argument("--tpm", required=True, help="Path to STAR TPM TSV (genes × samples).")
    parser.add_argument("--clinical", required=True, help="Path to clinical metadata TSV.")
    parser.add_argument(
        "--out-dir", default="analysis/qc", help="Directory where processed outputs will be written."
    )
    parser.add_argument(
        "--min-tpm",
        type=float,
        default=1.0,
        help="Minimum TPM to consider a gene expressed (default: 1.0).",
    )
    parser.add_argument(
        "--min-proportion",
        type=float,
        default=0.2,
        help="Minimum proportion of samples that must reach min-tpm (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for UMAP and any stochastic components.",
    )
    return parser.parse_args()


def tidy_expression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gene_column = df.columns[0]
    df[gene_column] = df[gene_column].astype(str).str.split(".").str[0]
    df = df.groupby(gene_column).sum()
    df.index.name = "gene_id"
    return df


def filter_genes(tpm: pd.DataFrame, min_tpm: float, min_prop: float) -> pd.Index:
    expressed = (tpm >= min_tpm).sum(axis=1) >= (min_prop * tpm.shape[1])
    return tpm.index[expressed]


def compute_library_metrics(counts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    library_size = counts.sum(axis=1)
    percent_zero = (counts == 0).mean(axis=1)
    return library_size, percent_zero


def mad_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - med) / mad
    return np.abs(modified_z) > threshold


def run_pca(log_tpm: pd.DataFrame, n_components: int = 10) -> Tuple[pd.DataFrame, PCA]:
    scaler = StandardScaler(with_mean=True, with_std=False)
    centered = scaler.fit_transform(log_tpm)
    pca = PCA(n_components=min(n_components, centered.shape[0], centered.shape[1]))
    pcs = pca.fit_transform(centered)
    columns = [f"PC{i+1}" for i in range(pcs.shape[1])]
    return pd.DataFrame(pcs, index=log_tpm.index, columns=columns), pca


def run_umap(log_tpm: pd.DataFrame, seed: int) -> pd.DataFrame:
    reducer = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(log_tpm)
    return pd.DataFrame(embedding, index=log_tpm.index, columns=["UMAP1", "UMAP2"])


def variance_by_covariate(pcs: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    enc = OneHotEncoder(drop="first", sparse_output=False)
    encoded = enc.fit_transform(covariates.fillna("Unknown"))
    results: Dict[str, Dict[str, float]] = {}
    for pc in pcs.columns[:5]:
        y = pcs[pc].values
        x = encoded
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        y_hat = x @ coef
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        results[pc] = {"R2": r2}
    r2_df = pd.DataFrame(results).T
    r2_df.index.name = "PC"
    r2_df["covariates"] = ", ".join(covariates.columns)
    return r2_df


def plot_embedding(df_embed: pd.DataFrame, labels: pd.Series, title: str, out_path: Path) -> None:
    palette = sns.color_palette("tab10", n_colors=len(labels.unique()))
    cmap = ListedColormap(palette)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=df_embed.iloc[:, 0],
        y=df_embed.iloc[:, 1],
        hue=labels,
        palette=cmap,
        s=40,
        edgecolor="none",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_qc_report(
    out_path: Path,
    metadata: pd.DataFrame,
    gene_count: int,
    library_size: pd.Series,
    silhouette_batch: float,
    outliers_lib: pd.Series,
    outliers_pc1: pd.Series,
    variance_table: pd.DataFrame,
) -> None:
    num_samples = metadata.shape[0]
    label_counts = metadata["label"].value_counts().to_dict()
    outlier_summary = {
        "library_size_outliers": metadata.loc[outliers_lib, "sample_id"].tolist(),
        "pc1_outliers": metadata.loc[outliers_pc1, "sample_id"].tolist(),
    }
    report = f"""# QC Summary\n\n"""
    report += f"**Samples:** {num_samples} (label distribution: {json.dumps(label_counts)})\n\n"
    report += f"**Genes retained:** {gene_count}\n\n"
    report += f"**Library size (median ± IQR):** {library_size.median():.2f} ± {library_size.quantile(0.75) - library_size.quantile(0.25):.2f}\n\n"
    report += f"**Batch silhouette (log TPM, UMAP embedding):** {silhouette_batch:.3f}\n\n"
    report += "**Outliers (MAD > 3.5):**\n\n"
    for key, values in outlier_summary.items():
        report += f"- {key}: {values if values else 'None'}\n"
    report += "\n**Variance explained (covariates on top PCs):**\n\n"
    report += variance_table.to_markdown() + "\n"
    out_path.write_text(report)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    matrices_dir = out_dir / "matrices"
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    counts_raw = tidy_expression(pd.read_csv(args.counts, sep="\t"))
    tpm_raw = tidy_expression(pd.read_csv(args.tpm, sep="\t"))

    counts_common = counts_raw.reindex(tpm_raw.index).fillna(0)
    gene_filter = filter_genes(tpm_raw, args.min_tpm, args.min_proportion)
    counts_filtered = counts_common.loc[gene_filter]
    tpm_filtered = tpm_raw.loc[gene_filter]

    counts_t = counts_filtered.T
    tpm_t = tpm_filtered.T
    log_tpm = np.log2(tpm_t + 1)

    counts_t.to_csv(matrices_dir / "counts.tsv", sep="\t")
    tpm_t.to_csv(matrices_dir / "tpm.tsv", sep="\t")
    log_tpm.to_csv(matrices_dir / "log_tpm.tsv", sep="\t")

    library_size, percent_zero = compute_library_metrics(counts_t)

    pcs, pca_model = run_pca(log_tpm, n_components=10)
    pcs.to_csv(tables_dir / "pca_scores.tsv", sep="\t")
    umap_df = run_umap(log_tpm, args.seed)
    umap_df.to_csv(tables_dir / "umap.tsv", sep="\t")

    outliers_lib = mad_outliers(library_size)
    outliers_pc1 = mad_outliers(pcs["PC1"])

    clinical = pd.read_csv(args.clinical, sep="\t", dtype=str)
    metadata = prepare_metadata(clinical, counts_t.index)
    metadata = metadata.merge(
        pd.DataFrame(
            {
                "sample_id": counts_t.index,
                "library_size": library_size.values,
                "percent_zero": percent_zero.values,
                "pc1": pcs["PC1"].values,
                "pc2": pcs["PC2"].values,
                "umap1": umap_df["UMAP1"].values,
                "umap2": umap_df["UMAP2"].values,
                "mad_outlier_library": outliers_lib.values,
                "mad_outlier_pc1": outliers_pc1.values,
            }
        ),
        on="sample_id",
        how="left",
    )
    metadata.to_csv(out_dir / "sample_metadata.tsv", sep="\t", index=False)

    plot_embedding(pcs[["PC1", "PC2"]], metadata["label"], "PCA (PC1 vs PC2) — label", plots_dir / "pca_label.png")
    plot_embedding(pcs[["PC1", "PC2"]], metadata["batch"], "PCA (PC1 vs PC2) — batch", plots_dir / "pca_batch.png")
    plot_embedding(umap_df, metadata["label"], "UMAP — label", plots_dir / "umap_label.png")
    plot_embedding(umap_df, metadata["batch"], "UMAP — batch", plots_dir / "umap_batch.png")

    silhouette_batch = float("nan")
    if metadata["batch"].nunique() > 1:
        try:
            silhouette_batch = silhouette_score(umap_df.values, metadata["batch"].fillna("Unknown"))
        except ValueError:
            silhouette_batch = float("nan")

    variance_table = variance_by_covariate(pcs, metadata[["label", "batch"]])
    variance_table.to_csv(tables_dir / "variance_by_covariate.tsv", sep="\t")

    write_qc_report(
        out_dir / "qc_summary.md",
        metadata,
        gene_count=len(gene_filter),
        library_size=library_size,
        silhouette_batch=silhouette_batch,
        outliers_lib=outliers_lib,
        outliers_pc1=outliers_pc1,
        variance_table=variance_table,
    )


def prepare_metadata(clinical: pd.DataFrame, samples: Iterable[str]) -> pd.DataFrame:
    clinical = clinical.copy()
    clinical_columns = {
        "sample": "sample_id",
        "sample_type.samples": "sample_type",
        "name.tissue_source_site": "batch",
        "project_id.project": "project_id",
        "age_at_index.demographic": "age",
        "gender.demographic": "gender",
        "ajcc_pathologic_stage.diagnoses": "path_stage",
    }
    for original, new in clinical_columns.items():
        if original in clinical.columns:
            clinical[new] = clinical[original]
        else:
            clinical[new] = np.nan

    label_map = {
        "Primary Tumor": "tumor",
        "Solid Tissue Normal": "normal",
        "Metastatic": "tumor",
    }
    clinical["label"] = clinical["sample_type"].map(label_map).fillna("unknown")
    clinical["KRAS_status"] = "unknown"
    clinical["TP53_status"] = "unknown"
    subset = clinical[[
        "sample_id",
        "label",
        "sample_type",
        "batch",
        "project_id",
        "age",
        "gender",
        "path_stage",
        "KRAS_status",
        "TP53_status",
    ]]
    subset = subset.drop_duplicates("sample_id")
    subset = subset[subset["sample_id"].isin(samples)]
    subset = subset.set_index("sample_id").reindex(samples).reset_index()
    return subset


if __name__ == "__main__":
    main()

