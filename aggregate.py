#!/usr/bin/env python3
import glob, json, os, sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import des_nids_revision as d

out = sys.argv[1]
def cat(pattern):
    fs = sorted(glob.glob(os.path.join(out, pattern)))
    fs = [f for f in fs if os.path.getsize(f) > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

res = cat("results_seed*.csv")
kdf, thr = cat("k_seed*.csv"), cat("throughput_seed*.csv")
infos = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "info_seed*.json")))]
res.to_csv(os.path.join(out, "results_all_runs.csv"), index=False)
if len(kdf):
    kdf.to_csv(os.path.join(out, "k_sensitivity_validation.csv"), index=False)
    kdf.drop(columns=["seed"]).groupby("k").agg(["mean", "std"]).to_csv(
        os.path.join(out, "k_sensitivity_summary.csv"))
if len(thr):
    thr.to_csv(os.path.join(out, "throughput_knorau.csv"), index=False)
json.dump(infos, open(os.path.join(out, "run_info.json"), "w"), indent=2)
summ, tests = d.summarize(res, out)
seeds = sorted(res["seed"].unique())
d.mcnemar_tests(out, int(seeds[0]))
classes = json.load(open(os.path.join(out, f"per_class_seed{seeds[0]}.json")))["classes"]
le = LabelEncoder().fit(classes)
d.make_figures(out, le, res, kdf, thr, int(seeds[0]))
pd.set_option("display.width", 220)
print(summ[["method", "n_runs", "accuracy_mean", "macro_f1_mean", "macro_f1_std",
            "mcc_mean", "avg_fpr_mean", "benign_false_alarm_rate_mean"]].to_string(index=False))
