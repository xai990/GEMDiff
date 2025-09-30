"""Baseline classifiers for expression data.

Loads log-transformed expression, builds multiple baseline models, computes
AUROC/AUPRC/Accuracy/F1 plus mean ± 95% CI via bootstrap, and saves ROC/PR
curves from a held-out fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline models on log expression data.")
    parser.add_argument("--log-tpm", required=True, help="Path to log2(TPM+1) matrix (samples × genes).")
    parser.add_argument("--metadata", required=True, help="Sample metadata TSV with target column.")
    parser.add_argument("--label-column", default="label", help="Target variable for classification.")
    parser.add_argument(
        "--out-dir", default="analysis/baselines", help="Directory to store metrics and plots."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--pca-components", type=int, default=50, help="Number of PCA components for PCA+LR baseline."
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of CV folds (default 5)."
    )
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, n_iterations: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_iterations):
        idx = rng.integers(0, len(values), len(values))
        samples.append(np.mean(values[idx]))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_model(model, x_train, y_train, x_test, y_test) -> Dict[str, float]:
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "auprc": average_precision_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "probs": probs,
        "preds": preds,
    }


def run_baselines(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_tpm = pd.read_csv(args.log_tpm, sep="\t", index_col=0)
    metadata = pd.read_csv(args.metadata, sep="\t")
    data = log_tpm.join(metadata.set_index("sample_id"), how="inner")
    y = (data[args.label_column] == "tumor").astype(int).values
    X = data.iloc[:, : log_tpm.shape[1]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(args.pca_components, X.shape[1], X.shape[0]-1))
    X_pca = pca.fit_transform(X_scaled)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    results = {"pca_logreg": [], "logreg": [], "random_forest": [], "mlp": []}

    fold = 0
    roc_payload = {}
    for train_idx, test_idx in skf.split(X_scaled, y):
        fold += 1
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]

        models = {
            "pca_logreg": (LogisticRegression(max_iter=200, random_state=args.seed), X_train_pca, X_test_pca),
            "logreg": (LogisticRegression(max_iter=200, random_state=args.seed), X_train, X_test),
            "random_forest": (
                RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
                X_train,
                X_test,
            ),
            "mlp": (
                MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=args.seed),
                X_train,
                X_test,
            ),
        }

        for name, (model, Xtr, Xte) in models.items():
            metrics = evaluate_model(model, Xtr, y_train, Xte, y_test)
            results[name].append(metrics)
            if fold == 1:
                roc_payload[name] = (y_test, metrics["probs"])

    summary_rows = []
    for model_name, records in results.items():
        for metric in ["roc_auc", "auprc", "accuracy", "f1"]:
            values = np.array([r[metric] for r in records])
            mean = float(values.mean())
            lower, upper = bootstrap_ci(values, seed=args.seed)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "baseline_metrics.tsv", sep="\t", index=False)

    for model_name, (y_true, probs) in roc_payload.items():
        RocCurveDisplay.from_predictions(y_true, probs)
        plt = RocCurveDisplay.from_predictions(y_true, probs)
        plt.figure_.savefig(out_dir / f"roc_{model_name}.png", dpi=150)
        plt.figure_.clf()

        pr = PrecisionRecallDisplay.from_predictions(y_true, probs)
        pr.figure_.savefig(out_dir / f"pr_{model_name}.png", dpi=150)
        pr.figure_.clf()

    (out_dir / "baseline_config.json").write_text(
        json.dumps(
            {
                "log_tpm": args.log_tpm,
                "metadata": args.metadata,
                "label_column": args.label_column,
                "pca_components": args.pca_components,
                "n_folds": args.n_folds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    args = parse_args()
    run_baselines(args)

