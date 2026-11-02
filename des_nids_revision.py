#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             fbeta_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def _deslib_compat():
    try:
        from deslib.base import BaseDS
        from deslib.static.base import BaseStaticEnsemble
    except Exception:
        return
    try:
        from sklearn.utils.validation import validate_data
    except ImportError:
        return

    def _validate_data(self, X="no_validation", y="no_validation", reset=True, **kw):
        return validate_data(self, X=X, y=y, reset=reset, **kw)

    for cls in (BaseDS, BaseStaticEnsemble):
        if not hasattr(cls, "_validate_data"):
            cls._validate_data = _validate_data


_deslib_compat()

OFFICIAL_CICIDS2017 = {
    "BENIGN": 2273097, "DoS Hulk": 231073, "DoS GoldenEye": 10293,
    "DoS slowloris": 5796, "DoS Slowhttptest": 5499,
    "FTP-Patator": 7938, "SSH-Patator": 5897,
}

ID_COLUMNS = {"flow id", "source ip", "src ip", "destination ip", "dst ip",
              "source port", "src port", "timestamp", "simillarhttp", "unnamed: 0"}


def load_data(args) -> pd.DataFrame:
    if args.synthetic:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=args.synthetic_n, n_features=40, n_informative=20,
            n_redundant=10, n_classes=7, n_clusters_per_class=2,
            weights=[0.82, 0.13, 0.015, 0.012, 0.011, 0.007, 0.005],
            class_sep=1.2, random_state=0)
        X = np.abs(X)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["Label"] = [f"class_{c}" for c in y]
        dup = df[df["Label"] == "class_1"].sample(frac=0.5, random_state=0)
        return pd.concat([df, dup], ignore_index=True)

    files = sorted(glob.glob(args.data))
    if not files:
        sys.exit(f"No files match {args.data}")
    keep = [c.strip() for c in args.classes.split(",")] if args.classes else None
    repeat = args.repeat_files.lower() if args.repeat_files else None
    parts = []
    for f in files:
        part = pd.read_csv(f, low_memory=False, encoding="latin-1")
        part.columns = [c.strip() for c in part.columns]
        lab = args.label_col
        part[lab] = part[lab].astype(str).str.strip()
        n_raw = len(part)
        if keep:
            part = part[part[lab].isin(keep)]
        if lab != "Label":
            part = part.rename(columns={lab: "Label"})
        feats = [c for c in part.columns
                 if c != "Label" and c.lower() not in ID_COLUMNS]
        part = part[feats + ["Label"]].copy()
        part = part[part["Label"] != "Label"]
        for c in feats:
            v = pd.to_numeric(part[c], errors="coerce").astype(np.float32)
            v[~np.isfinite(v)] = np.nan
            part[c] = v
        part["Label"] = part["Label"].astype("category")
        parts.append(part)
        print(f"  loaded {os.path.basename(f)}: {n_raw:,} rows, kept {len(part):,}")
        if repeat and repeat in os.path.basename(f).lower():
            parts.append(part.copy())
            print(f"  ... appended a second copy of {os.path.basename(f)} (--repeat-files)")
    df = pd.concat(parts, ignore_index=True)
    df["Label"] = df["Label"].astype(str)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "Label" and c.lower() not in ID_COLUMNS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def audit_and_dedup(df: pd.DataFrame, mode: str, out: str) -> pd.DataFrame:
    feats = feature_columns(df)
    key = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
    codes = pd.factorize(df["Label"])[0].astype(np.uint64)
    full_key = key ^ ((codes + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15))
    full_dup_first = pd.Series(full_key).duplicated().to_numpy()
    full_dup_any = pd.Series(full_key).duplicated(keep=False).to_numpy()
    conflict = (pd.DataFrame({"k": key, "y": codes})
                .groupby("k")["y"].transform("nunique") > 1).to_numpy()

    counts = df["Label"].value_counts()
    audit = {
        "rows_loaded": int(len(df)),
        "n_feature_columns": len(feats),
        "class_counts_loaded": {k: int(v) for k, v in counts.items()},
        "exact_duplicate_rows_features_and_label": int(full_dup_first.sum()),
        "rows_whose_feature_vector_has_another_label": int(conflict.sum()),
    }
    per_class = pd.DataFrame({"dup": full_dup_any, "first": full_dup_first,
                              "y": df["Label"].to_numpy()}).groupby("y")
    audit["share_of_rows_with_an_exact_copy_per_class"] = {
        k: round(float(v), 6) for k, v in per_class["dup"].mean().items()}
    audit["redundant_copies_per_class"] = {
        k: int(v) for k, v in per_class["first"].sum().items()}
    ratio = {k: round(counts[k] / v, 4) for k, v in OFFICIAL_CICIDS2017.items()
             if k in counts}
    if ratio:
        audit["ratio_to_official_cicids2017_counts"] = ratio

    if mode == "exact":
        keep = ~full_dup_first & ~conflict
        df = df[keep].reset_index(drop=True)
        audit["rows_after_dedup"] = int(len(df))
        audit["class_counts_after_dedup"] = {
            k: int(v) for k, v in df["Label"].value_counts().items()}
    with open(os.path.join(out, "dataset_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    return df


@dataclass
class Preprocessor:
    corr_threshold: float = 0.95
    corr_sample: int = 200_000

    def fit(self, X: pd.DataFrame, seed: int):
        X = X.replace([np.inf, -np.inf], np.nan)
        neg_share = (X < 0).mean()
        self.nonneg_ = list(neg_share[neg_share < 1e-3].index)
        X = self._neg_to_nan(X)
        self.median_ = X.median()
        X = X.fillna(self.median_)
        var = X.var()
        self.zero_var_ = list(var[var <= 0].index)
        X = X.drop(columns=self.zero_var_)
        S = X.sample(min(len(X), self.corr_sample), random_state=seed)
        corr = S.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self.corr_drop_ = [c for c in upper.columns
                           if (upper[c] > self.corr_threshold).any()]
        X = X.drop(columns=self.corr_drop_)
        self.columns_ = list(X.columns)
        self.min_ = X.min()
        self.range_ = (X.max() - self.min_).replace(0, 1.0)
        return self

    def _neg_to_nan(self, X):
        X = X.copy()
        for c in self.nonneg_:
            X.loc[X[c] < 0, c] = np.nan
        return X

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.replace([np.inf, -np.inf], np.nan)
        X = self._neg_to_nan(X).fillna(self.median_)[self.columns_]
        Z = ((X - self.min_) / self.range_).clip(0.0, 1.0)
        return Z.to_numpy(dtype=np.float32)

    def summary(self):
        return {"zero_variance_removed": len(self.zero_var_),
                "correlated_removed": len(self.corr_drop_),
                "final_features": len(self.columns_),
                "final_feature_names": self.columns_}


def sanitize_proba(P: np.ndarray) -> np.ndarray:
    P = np.nan_to_num(np.asarray(P, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    zero = (s[:, 0] == 0)
    P[zero] = 1.0 / P.shape[1]
    s[zero] = 1.0
    return P / s


def compute_metrics(y, yp, P, n_classes, benign_idx):
    cm = confusion_matrix(y, yp, labels=list(range(n_classes)))
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fpr = fp / np.maximum(fp + tn, 1)
    m = {
        "accuracy": accuracy_score(y, yp),
        "balanced_accuracy": balanced_accuracy_score(y, yp),
        "macro_precision": precision_score(y, yp, average="macro", zero_division=0),
        "macro_recall": recall_score(y, yp, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yp, average="macro", zero_division=0),
        "macro_f2": fbeta_score(y, yp, beta=2, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, yp),
        "kappa": cohen_kappa_score(y, yp),
        "avg_fpr": float(fpr.mean()),
    }
    if benign_idx is not None:
        b = (y == benign_idx)
        m["benign_false_alarm_rate"] = float((yp[b] != benign_idx).mean())
        m["attack_miss_rate"] = float((yp[~b] == benign_idx).mean())
    if P is not None:
        P = sanitize_proba(P)
        try:
            m["macro_auc_ovr"] = roc_auc_score(y, P, multi_class="ovr",
                                               average="macro",
                                               labels=list(range(n_classes)))
        except ValueError:
            m["macro_auc_ovr"] = np.nan
        m["log_loss"] = log_loss(y, np.clip(P, 1e-15, 1),
                                 labels=list(range(n_classes)))
    per_class = {
        "fpr": fpr.tolist(),
        "recall": recall_score(y, yp, average=None, labels=list(range(n_classes)),
                               zero_division=0).tolist(),
        "precision": precision_score(y, yp, average=None,
                                     labels=list(range(n_classes)),
                                     zero_division=0).tolist(),
        "f1": f1_score(y, yp, average=None, labels=list(range(n_classes)),
                       zero_division=0).tolist(),
    }
    return m, per_class, cm


def make_pool(seed, n_jobs):
    from xgboost import XGBClassifier
    return {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=n_jobs,
                                               random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 tree_method="hist", n_jobs=n_jobs,
                                 random_state=seed, eval_metric="mlogloss"),
    }


class StaticFusion:

    def __init__(self, pool, kind, n_classes, seed=0):
        self.pool, self.kind, self.C, self.seed = pool, kind, n_classes, seed

    def fit(self, X_dsel, y_dsel):
        if self.kind == "stacking":
            Z = np.hstack([m.predict_proba(X_dsel) for m in self.pool])
            self.meta_ = LogisticRegression(max_iter=3000, random_state=self.seed)
            self.meta_.fit(Z, y_dsel)
        return self

    def predict_proba(self, X):
        if self.kind == "soft_voting":
            return np.mean([m.predict_proba(X) for m in self.pool], axis=0)
        if self.kind == "hard_voting":
            votes = np.zeros((len(X), self.C))
            for m in self.pool:
                votes[np.arange(len(X)), m.predict(X).astype(int)] += 1
            return votes / len(self.pool)
        Z = np.hstack([m.predict_proba(X) for m in self.pool])
        return self.meta_.predict_proba(Z)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def make_ds(name, pool, k, seed, n_jobs):
    from deslib.dcs import LCA, MCB, OLA
    from deslib.des import DESP, KNORAE, KNORAU, METADES
    from deslib.static import SingleBest
    ctor = {"KNORAU": KNORAU, "KNORAE": KNORAE, "METADES": METADES,
            "DESP": DESP, "OLA": OLA, "LCA": LCA, "MCB": MCB}
    if name == "SingleBest":
        return SingleBest(pool_classifiers=pool, random_state=seed, n_jobs=n_jobs)
    kw = dict(pool_classifiers=pool, k=k, random_state=seed)
    if name not in ("OLA", "LCA", "MCB"):
        kw["n_jobs"] = n_jobs
    return ctor[name](**kw)


def oracle_accuracy(pool, X, y):
    hit = np.zeros(len(y), dtype=bool)
    for m in pool:
        hit |= (m.predict(X).astype(int) == y)
    return float(hit.mean())


def throughput(model, X, batch, runs, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        idx = rng.choice(len(X), size=min(batch, len(X)), replace=False)
        xb = X[idx]
        t0 = time.perf_counter()
        model.predict(xb)
        out.append(len(xb) / (time.perf_counter() - t0))
    return np.array(out)


def stratified_cap(X, y, cap, seed):
    if cap is None or len(y) <= cap:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=cap, stratify=y,
                                    random_state=seed)
    return Xs, ys


T0 = time.perf_counter()
SELECTED_K: list[int] = []


def log(msg):
    print(f'[{time.perf_counter() - T0:7.0f}s] {msg}', flush=True)


def run_seed(df, le, args, seed, out):
    feats = feature_columns(df)
    y_all = le.transform(df["Label"])
    C = len(le.classes_)
    benign = next((i for i, c in enumerate(le.classes_)
                   if c.lower() in ("benign", "class_0")), None)

    idx = np.arange(len(df))
    i_rest, i_test = train_test_split(idx, test_size=args.test_frac,
                                      stratify=y_all, random_state=seed)
    frac_dsel = args.dsel_frac / (1 - args.test_frac)
    i_pool, i_tmp = train_test_split(i_rest, test_size=frac_dsel + args.val_frac / (1 - args.test_frac),
                                     stratify=y_all[i_rest], random_state=seed)
    val_share = args.val_frac / (args.val_frac + args.dsel_frac)
    i_dsel, i_val = train_test_split(i_tmp, test_size=val_share,
                                     stratify=y_all[i_tmp], random_state=seed)

    Xdf = df[feats]
    pre = Preprocessor(corr_threshold=args.corr).fit(Xdf.iloc[i_pool], seed)
    Xp, Xd, Xv, Xt = (pre.transform(Xdf.iloc[i]) for i in (i_pool, i_dsel, i_val, i_test))
    log(f'preprocessed: {pre.summary()["final_features"]} features; pool {len(i_pool):,} dsel {len(i_dsel):,} val {len(i_val):,} test {len(i_test):,}')
    yp_, yd, yv, yt = (y_all[i] for i in (i_pool, i_dsel, i_val, i_test))

    def row_hashes(A):
        return {hashlib.blake2b(r.tobytes(), digest_size=8).digest() for r in A}
    train_h = row_hashes(np.vstack([Xp, Xd]))
    overlap = float(np.mean([hashlib.blake2b(r.tobytes(), digest_size=8).digest()
                             in train_h for r in Xt]))

    Xp_fit, yp_fit = stratified_cap(Xp, yp_, args.pool_max, seed)
    Xd_fit, yd_fit = stratified_cap(Xd, yd, args.dsel_max, seed)
    if args.dsel_equals_train:
        Xd_fit, yd_fit = stratified_cap(Xp_fit, yp_fit, args.dsel_max, seed)

    pool_dict = make_pool(seed, args.n_jobs)
    fit_times = {}
    for n, m in pool_dict.items():
        t0 = time.perf_counter()
        m.fit(Xp_fit, yp_fit)
        fit_times[n] = time.perf_counter() - t0
        log(f'fitted {n} in {fit_times[n]:.0f}s')
    pool = list(pool_dict.values())

    k_rows = []
    from deslib.des import KNORAU
    grid = args.k_grid if seed in args.seeds[:args.k_grid_seeds] else []
    for k in grid:
        des = KNORAU(pool_classifiers=pool, k=k, random_state=seed,
                     n_jobs=args.n_jobs).fit(Xd_fit, yd_fit)
        pv = des.predict(Xv)
        mv, _, _ = compute_metrics(yv, pv, None, C, benign)
        k_rows.append({"seed": seed, "k": k, **mv})
        log(f'k={k}: val macro-F1 {mv["macro_f1"]:.5f}')
    k_df = pd.DataFrame(k_rows)
    if args.k is not None:
        k_best = args.k
    elif len(k_df):
        k_best = int(k_df.sort_values(["macro_f1", "k"], ascending=[False, True]).iloc[0]["k"])
        SELECTED_K.append(k_best)
    else:
        k_best = int(pd.Series(SELECTED_K).mode().min())

    models = {}
    for n, m in pool_dict.items():
        models[n] = m
    for kind in ("hard_voting", "soft_voting", "stacking"):
        models[kind] = StaticFusion(pool, kind, C, seed).fit(Xd_fit, yd_fit)
    for name in args.methods:
        t0 = time.perf_counter()
        models[name] = make_ds(name, pool, k_best, seed, args.n_jobs).fit(Xd_fit, yd_fit)
        fit_times[name] = time.perf_counter() - t0
        log(f'fitted {name} in {fit_times[name]:.0f}s')

    rows, per_class_all, preds = [], {}, {}
    for name, m in models.items():
        t0 = time.perf_counter()
        yhat = np.asarray(m.predict(Xt)).astype(int)
        t_pred = time.perf_counter() - t0
        P = None
        want_proba = (name in pool_dict or name in ("hard_voting", "soft_voting", "stacking")
                      or name == "KNORAU" or seed == args.seeds[0])
        if want_proba:
            try:
                P = m.predict_proba(Xt)
            except Exception:
                P = None
        met, pc, cm = compute_metrics(yt, yhat, P, C, benign)
        met.update({"seed": seed, "method": name, "k": k_best,
                    "test_predict_seconds": t_pred,
                    "test_throughput_pred_per_s": len(Xt) / t_pred})
        rows.append(met)
        log(f'{name}: test macro-F1 {met["macro_f1"]:.5f} avg-FPR {met["avg_fpr"]:.6f} ({t_pred:.0f}s)')
        per_class_all[name] = pc
        preds[name] = yhat
        if seed == args.seeds[0]:
            np.save(os.path.join(out, f"cm_{name}_seed{seed}.npy"), cm)

    thr = []
    if "KNORAU" in models:
        for b in args.batches:
            v = throughput(models["KNORAU"], Xt, b, args.thr_runs, seed)
            thr.append({"seed": seed, "batch": b, "mean": v.mean(), "std": v.std(ddof=1),
                        "min": v.min(), "max": v.max(),
                        "latency_ms_per_flow": 1000.0 / v.mean()})

    info = {"seed": seed, "k_selected_on_validation": k_best,
            "sizes": {"pool_train": len(i_pool), "pool_train_used": len(yp_fit),
                      "dsel": len(i_dsel), "dsel_used": len(yd_fit),
                      "validation": len(i_val), "test": len(i_test)},
            "test_vectors_identical_to_a_training_or_dsel_vector": overlap,
            "oracle_accuracy_test": oracle_accuracy(pool, Xt, yt),
            "fit_seconds": fit_times, "preprocessing": pre.summary()}
    np.savez_compressed(os.path.join(out, f"preds_seed{seed}.npz"), y_true=yt,
                        **{k: v for k, v in preds.items()})
    return pd.DataFrame(rows), k_df, pd.DataFrame(thr), per_class_all, info


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def summarize(res: pd.DataFrame, out: str, ref: str = "KNORAU"):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_f2", "mcc",
               "kappa", "avg_fpr", "benign_false_alarm_rate", "attack_miss_rate",
               "macro_auc_ovr", "log_loss", "test_throughput_pred_per_s"]
    metrics = [m for m in metrics if m in res.columns]
    lines = []
    for (meth), g in res.groupby("method"):
        row = {"method": meth, "n_runs": len(g)}
        for m in metrics:
            v = g[m].dropna().to_numpy()
            mean, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
            row[f"{m}_mean"], row[f"{m}_std"] = mean, sd
            row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = mean - h, mean + h
        lines.append(row)
    summ = pd.DataFrame(lines).sort_values("macro_f1_mean", ascending=False)
    summ.to_csv(os.path.join(out, "summary_mean_std_ci95.csv"), index=False)

    tests = {}
    if res["seed"].nunique() >= 3 and ref in res["method"].values:
        for metric, better in (("macro_f1", "greater"), ("mcc", "greater"),
                               ("avg_fpr", "less"),
                               ("benign_false_alarm_rate", "less")):
            if metric not in res.columns:
                continue
            piv = res.pivot(index="seed", columns="method", values=metric).dropna(axis=1)
            fr = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
            raw = {}
            for other in piv.columns:
                if other == ref:
                    continue
                d = piv[ref] - piv[other]
                if np.allclose(d, 0):
                    raw[other] = 1.0
                    continue
                raw[other] = stats.wilcoxon(piv[ref], piv[other],
                                            alternative=better,
                                            zero_method="zsplit").pvalue
            tests[metric] = {"friedman_stat": fr.statistic,
                             "friedman_p": fr.pvalue,
                             "wilcoxon_one_sided_p_raw": raw,
                             "wilcoxon_holm_adjusted": holm(raw),
                             "mean_rank": piv.rank(axis=1, ascending=(better == "less"))
                                             .mean().sort_values().to_dict()}
    with open(os.path.join(out, "significance_tests.json"), "w") as f:
        json.dump(tests, f, indent=2, default=float)
    return summ, tests


def mcnemar_tests(out, seed, ref="KNORAU"):
    d = np.load(os.path.join(out, f"preds_seed{seed}.npz"))
    y = d["y_true"]
    if ref not in d.files:
        return {}
    a = d[ref] == y
    res = {}
    for k in d.files:
        if k in ("y_true", ref):
            continue
        b = d[k] == y
        n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
        n = n01 + n10
        p = stats.binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
        res[k] = {"ref_correct_other_wrong": n01, "ref_wrong_other_correct": n10,
                  "exact_mcnemar_p": p}
    with open(os.path.join(out, f"mcnemar_seed{seed}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def environment(args):
    import importlib
    env = {"python": sys.version, "platform": f"{platform.system()}-{platform.machine()}",
           "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "args": vars(args)}
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    for lib in ("numpy", "pandas", "scipy", "sklearn", "deslib", "xgboost",
                "matplotlib"):
        try:
            env[lib] = importlib.import_module(lib).__version__
        except Exception:
            env[lib] = "n/a"
    return env


def make_figures(out, le, res, kdf, thr, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(le.classes_)
    p = os.path.join(out, f"cm_KNORAU_seed{seed}.npy")
    if os.path.exists(p):
        cm = np.load(p)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.3f})", ha="center",
                        va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        fig.colorbar(im, fraction=0.046, label="Row-normalised share")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_confusion_matrix.png"), dpi=300)
        plt.close(fig)
    if len(kdf):
        g = kdf.groupby("k")[["macro_f1", "balanced_accuracy", "accuracy"]].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m in ("macro_f1", "balanced_accuracy", "accuracy"):
            ax.errorbar(g.index, g[(m, "mean")], yerr=g[(m, "std")].fillna(0),
                        marker="o", capsize=3, label=m)
        ax.set_xlabel("k (neighbourhood size)"); ax.set_ylabel("Validation score")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_k_sensitivity.png"), dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="glob of CSV files")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--classes", default=None, help="comma-separated class names to keep")
    ap.add_argument("--dedup", choices=["exact", "none"], default="exact")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--dsel-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--corr", type=float, default=0.95)
    ap.add_argument("--k-grid", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15])
    ap.add_argument("--k", type=int, default=None, help="fix k instead of selecting it")
    ap.add_argument("--k-grid-seeds", type=int, default=3,
                    help="run the k grid on the first N seeds; later seeds reuse the modal k")
    ap.add_argument("--methods", nargs="+",
                    default=["KNORAU", "KNORAE", "METADES", "DESP", "OLA", "LCA",
                             "SingleBest"])
    ap.add_argument("--pool-max", type=int, default=None,
                    help="stratified cap on pool-training rows (runtime control)")
    ap.add_argument("--dsel-max", type=int, default=None,
                    help="stratified cap on DSEL rows (runtime control)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 100, 1000])
    ap.add_argument("--thr-runs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-n", type=int, default=30000)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--dsel-equals-train", action="store_true",
                    help="reproduce the first-submission protocol (DSEL = pool training data)")
    ap.add_argument("--repeat-files", default=None,
                    help="load files whose name contains this text twice "
                         "(reproduces a duplicated-file error, for the control study)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "environment.json"), "w") as f:
        json.dump(environment(args), f, indent=2, default=str)

    print("Loading data ...")
    df = load_data(args)
    df = audit_and_dedup(df, args.dedup, args.out)
    if args.audit_only:
        return
    le = LabelEncoder().fit(df["Label"])

    all_res, all_k, all_thr, infos = [], [], [], []
    for s in args.seeds:
        print(f"\n=== seed {s} ===")
        r, k, t, pc, info = run_seed(df, le, args, s, args.out)
        all_res.append(r); all_k.append(k); all_thr.append(t); infos.append(info)
        r.to_csv(os.path.join(args.out, f"results_seed{s}.csv"), index=False)
        k.to_csv(os.path.join(args.out, f"k_seed{s}.csv"), index=False)
        t.to_csv(os.path.join(args.out, f"throughput_seed{s}.csv"), index=False)
        with open(os.path.join(args.out, f"info_seed{s}.json"), "w") as f:
            json.dump(info, f, indent=2, default=float)
        with open(os.path.join(args.out, f"per_class_seed{s}.json"), "w") as f:
            json.dump({"classes": list(le.classes_), **pc}, f, indent=2)
        cols = ["method", "accuracy", "macro_f1", "mcc", "kappa", "avg_fpr"]
        print(r[cols].sort_values("macro_f1", ascending=False).to_string(index=False))

    res = pd.concat(all_res, ignore_index=True)
    kdf = pd.concat(all_k, ignore_index=True)
    thr = pd.concat(all_thr, ignore_index=True)
    res.to_csv(os.path.join(args.out, "results_all_runs.csv"), index=False)
    kdf.to_csv(os.path.join(args.out, "k_sensitivity_validation.csv"), index=False)
    if len(kdf):
        kdf.drop(columns=["seed"]).groupby("k").agg(["mean", "std"]).to_csv(
            os.path.join(args.out, "k_sensitivity_summary.csv"))
    thr.to_csv(os.path.join(args.out, "throughput_knorau.csv"), index=False)
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(infos, f, indent=2, default=float)

    summ, tests = summarize(res, args.out)
    mcnemar_tests(args.out, args.seeds[0])
    make_figures(args.out, le, res, kdf, thr, args.seeds[0])
    pd.set_option("display.width", 200)
    print("\nSUMMARY (mean over seeds):")
    print(summ[["method", "n_runs", "accuracy_mean", "macro_f1_mean", "macro_f1_std",
                "mcc_mean", "avg_fpr_mean"]].to_string(index=False))
    print(f"\nAll outputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
