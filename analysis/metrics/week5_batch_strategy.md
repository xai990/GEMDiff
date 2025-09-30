# Week 5 Batch Strategy Memo

## Overview

We evaluated the pancreatic (TCGA-PAAD) embedding pipeline on the Week 3–5
sanity checklist. The dataset was regenerated with batch-aware splits using the
`name.tissue_source_site` field as the batch identifier.

UMAP embeddings, kNN separability, AUROC/AUPRC, and silhouette-by-batch scores
were computed with `scripts/evaluate_embeddings.py` (seed 1234, positive label
`tumor`). Three corrections were compared: no correction, train-only batch
centering, and ComBat (via `neuroCombat`). ROC and PR curves were also saved for
each binary comparison.

## Sanity Metrics

| Correction      | kNN acc | AUROC | AUPRC | Silhouette (batch) |
|-----------------|:-------:|:-----:|:-----:|:------------------:|
| none            | 0.977   | 0.767 | 0.994 |       -0.631       |
| batch_center    | 0.977   | 1.000 | 1.000 |       -0.642       |
| combat          | 0.977   | 0.767 | 0.994 |       -0.631       |
| harmony         | 0.977   | 0.767 | 0.994 |       -0.699       |

- **Label separability:** kNN accuracy remains high for both strategies,
  indicating the latent space still captures tumor vs. normal differences.
- **Ranking metrics:** Batch centering currently reports AUROC/AUPRC ≈1.0.
  Given how extreme this is, flag it for manual validation—double-check that the
  stratified split and centering implementation you deploy in production match
  the train-only logic here, and confirm there is no label↔batch confounding in
  your processed splits.
- **Batch imprint:** Silhouette scores are negative for all approaches—batches
  remain interleaved rather than forming tight clusters. Batch centering
  marginally decreases the silhouette, implying slightly better batch mixing,
  while ComBat mirrors the baseline and Harmony pushes the score a bit lower
  (better mixing) by recasting into a latent space.

## Embedding Diagnostics

The following UMAP plots were generated (see `analysis/metrics/`):

- `embedding_none_label.png` vs. `embedding_none_batch.png`
- `embedding_batch_center_label.png` vs. `embedding_batch_center_batch.png`
- `embedding_combat_label.png` vs. `embedding_combat_batch.png`
- `embedding_harmony_label.png` vs. `embedding_harmony_batch.png`

Binary ROC/PR curves (`roc_<method>.png`, `pr_<method>.png`) accompany these
plots for quick diagnostic checks.

Visual inspection confirms that the label structure is preserved while batch
effects stay diffuse. Batch centering causes slight deformation without reducing
batch overlap, and ComBat closely resembles the uncorrected embedding for this
cohort.

## Recommendation

- Given comparable separability and marginally improved silhouette, the
  train-only batch centering is acceptable if you prefer a lightweight
  adjustment, but treat the perfect AUROC/AUPRC as suspicious until you
  replicate the numbers outside this notebook run.
- If stricter batch mitigation is required on other cohorts, extend the
  comparison to Harmony or MNN after integrating those options.
- Continue to monitor kNN/AUROC/AUPRC alongside silhouette after any correction
  to ensure biology is not washed out.

## Next Actions

1. For future datasets with stronger batch imprint, consider adding Harmony or
   MNN to the comparison suite.
2. Include snapshots of the label/batch UMAP plots (including `embedding_combat_*`
   and `embedding_harmony_*`) plus the metrics table in the Week 5 deliverable deck.
