# DES-NIDS: revision experiments

Code accompanying the revised manuscript
"Fine-Grained Dynamic Ensemble Selection for Network Intrusion Detection:
Local Competence and False Alarms" (M. Hajjouz, A. Hajjouz, A. Musayev),
Scientific and Technical Journal of Information Technologies, Mechanics and Optics.

One script, `des_nids_revision.py`, produces all results reported in the revised
paper; `aggregate.py` merges the per-seed files of a run.

## What the pipeline does

1. **Dataset audit** (`dataset_audit.json`): class counts, exact duplicate rows,
   feature vectors that occur with more than one label, share of rows that have an
   exact copy per class, and the ratio of each class count to the official
   CIC-IDS2017 counts.
2. **Leakage control**: exact duplicates and label-conflicting vectors are removed
   *before* splitting (`--dedup exact`). The control run `--dedup none` reproduces
   the original protocol. Every run also reports the share of test vectors that
   are identical to a training or DSEL vector after preprocessing.
3. **Disjoint partitions** per seed (stratified): pool training 55 %,
   DSEL 20 %, validation 5 %, test 20 %. All preprocessing statistics
   (imputation, zero-variance filter, |r| > 0.95 correlation filter, MinMax) are
   fitted on the pool-training partition only.
4. **Pool**: Gaussian NB, Logistic Regression, Decision Tree, Random Forest, XGBoost.
5. **Compared methods on the same pool and data**
   * single models (the five pool members);
   * static fusion: hard voting, soft voting, stacking (logistic meta-learner
     trained on the DSEL partition), SingleBest, Oracle upper bound;
   * dynamic selection (DESlib): KNORA-U (proposed), KNORA-E, META-DES, DES-P,
     OLA, LCA (MCB optional).
6. **k selection** on the validation split over k ∈ {3, 5, 7, 9, 11, 13, 15}
   (`k_sensitivity_validation.csv`, `k_sensitivity_summary.csv`,
   `fig_k_sensitivity.png`). The test split is never used for tuning.
7. **Metrics**: accuracy, balanced accuracy, macro precision/recall/F1/F2, MCC,
   Cohen's kappa (6 decimals), per-class FPR, average FPR, benign false-alarm
   rate, attack miss rate, macro one-vs-rest AUC, log loss.
8. **Statistics over seeds**: mean, std and 95 % t-intervals
   (`summary_mean_std_ci95.csv`); Friedman test, one-sided Wilcoxon signed-rank
   test of KNORA-U against every other method with Holm correction
   (`significance_tests.json`); exact McNemar test on per-flow predictions
   (`mcnemar_seed<first>.json`). Use at least 10 seeds: with 5 seeds the smallest
   attainable one-sided Wilcoxon p-value is 0.031 and Holm correction over 13
   comparisons cannot reach 0.05.
9. **Inference cost** in batch mode for batch sizes 1, 100 and 1000
   (`throughput_knorau.csv`), plus single-thread fit times.
10. **Environment** (`environment.json`): Python, OS, CPU, RAM, library versions,
    all command-line arguments.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

DESlib 0.3.7 calls a private scikit-learn method removed in scikit-learn 1.6; the
script restores it through the public `validate_data` API, so recent
scikit-learn versions work.

## Running

```bash
# 0) smoke test, no data needed (about 1 minute)
python des_nids_revision.py --synthetic --seeds 0 1 2 --out results_smoke

# 1) audit only
python des_nids_revision.py --data "MachineLearningCVE/*.csv" \
  --classes "BENIGN,DoS Hulk,DoS GoldenEye,DoS slowloris,DoS Slowhttptest,FTP-Patator,SSH-Patator" \
  --audit-only --out results_audit

# 2) main experiment, deduplicated, 10 seeds
python des_nids_revision.py --data "MachineLearningCVE/*.csv" \
  --classes "BENIGN,DoS Hulk,DoS GoldenEye,DoS slowloris,DoS Slowhttptest,FTP-Patator,SSH-Patator" \
  --dedup exact --seeds 0 1 2 3 4 5 6 7 8 9 --n-jobs 8 --out results_dedup

# 3) control experiment, original protocol without deduplication
python des_nids_revision.py ... --dedup none --out results_nodedup

# 4) optional external dataset (CIC-DDoS2019 or CSE-CIC-IDS2018)
python des_nids_revision.py --data "CICDDoS2019/*.csv" --label-col Label \
  --dedup exact --out results_ddos2019
```

Runtime control on a laptop: `--pool-max 600000 --dsel-max 200000` draws
stratified caps of the pool-training and DSEL partitions (the test partition is
never subsampled). Report any cap used in the paper. META-DES is the slowest
method; drop it with `--methods KNORAU KNORAE DESP OLA LCA SingleBest` only if
time does not allow it, and state this.

## Results in brief (folder `results/`)

| Setting | KNORA-U macro-F1, % | XGBoost macro-F1, % | KNORA-U BFAR, ×10⁻⁴ | XGBoost BFAR, ×10⁻⁴ |
|---|---|---|---|---|
| CIC-IDS2017, deduplicated, 10 seeds (`main_dedup`) | 99.621 ± 0.049 | 99.600 ± 0.068 | 0.71 ± 0.14 | 0.91 ± 0.18 |
| CIC-IDS2017, duplicates kept, 5 seeds (`control_nodedup`) | 99.68 ± 0.09 | 99.68 ± 0.08 | 1.63 ± 0.10 | 1.41 ± 0.12 |
| First-version setting, 1 seed (`repro_first_version`) | 99.74 | 99.74 | 2.51 | 1.82 |
| CSE-CIC-IDS2018, 5 classes, 5 seeds (`ext_cicids2018`) | 99.99 ± 0.01 | 99.99 ± 0.01 | 0.06 ± 0.04 | 0.09 ± 0.05 |

BFAR: share of benign flows classified as an attack. Statistics over seeds are in
`results/*/summary_mean_std_ci95.csv` and `results/main_dedup/significance_tests.json`.
`RUN_ALL.sh` lists the exact commands.

## Note on the class counts used in the first submission

The first submission reported DoS-Hulk 462,146, DoS-GoldenEye 20,586,
DoS-slowloris 11,592 and DoS-Slowhttptest 10,998 flows: exactly twice the official
CIC-IDS2017 counts (231,073; 10,293; 5,796; 5,499), while FTP (7,938) and SSH
(5,897) matched. The copy of the data used then contained the DoS records twice.
All results in this repository use the official files, loaded once
(`results/audit/dataset_audit.json`: ratio 1.0 for every class).
`--repeat-files wednesday` reproduces the duplicated-file error for the control
study in `results/repro_first_version`.

## Outputs (per run directory)

| File | Content |
|---|---|
| `dataset_audit.json` | duplicate and count audit |
| `results_all_runs.csv` | one row per method and seed, all metrics |
| `summary_mean_std_ci95.csv` | mean, std, 95 % CI per method |
| `significance_tests.json` | Friedman, Wilcoxon + Holm, mean ranks |
| `mcnemar_seed*.json` | exact McNemar vs KNORA-U |
| `per_class_seed*.json` | per-class precision, recall, F1, FPR |
| `k_sensitivity_*.csv`, `fig_k_sensitivity.png` | k analysis on validation |
| `fig_confusion_matrix.png` | readable 7x7 matrix, counts and row shares |
| `throughput_knorau.csv` | batch throughput and latency per flow |
| `results_seed*.csv`, `info_seed*.json`, `k_seed*.csv`, `throughput_seed*.csv` | per-seed outputs merged by `aggregate.py` |
| `run_info.json` | partition sizes, selected k, train/test overlap, oracle bound |
| `environment.json` | software and hardware |

## Licence

MIT
