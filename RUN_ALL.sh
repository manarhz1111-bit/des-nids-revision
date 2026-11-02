#!/usr/bin/env bash
set -euo pipefail
C17="BENIGN,DoS Hulk,DoS GoldenEye,DoS slowloris,DoS Slowhttptest,FTP-Patator,SSH-Patator"
C18="Benign,DoS attacks-Hulk,DoS attacks-GoldenEye,DoS attacks-Slowloris,SSH-Bruteforce"
COMMON="--pool-max 600000 --dsel-max 200000 --n-jobs 2"

python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup exact --audit-only --out results/audit
python des_nids_revision.py --data "data/cicids2018/*.csv" \
  --classes "Benign,DoS attacks-Hulk,DoS attacks-GoldenEye,DoS attacks-Slowloris,DoS attacks-SlowHTTPTest,FTP-BruteForce,SSH-Bruteforce" \
  --dedup exact --audit-only --out results/audit2018

python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup exact --seeds 0 --k-grid-seeds 1 $COMMON --out results/main_dedup
python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup exact --seeds 1 2 --k-grid-seeds 2 $COMMON --out results/main_dedup
for s in "3 4" "5 6" "7 8" "9"; do
  python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup exact --seeds $s --k 5 --k-grid-seeds 0 $COMMON --out results/main_dedup
done
python aggregate.py results/main_dedup

python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup none --seeds 0 1 2 3 4 --k 5 --k-grid-seeds 0 \
  --methods KNORAU KNORAE --thr-runs 10 $COMMON --out results/control_nodedup
python aggregate.py results/control_nodedup

python des_nids_revision.py --data "data/cicids2017/*.csv" --classes "$C17" --dedup none --repeat-files wednesday --dsel-equals-train \
  --seeds 0 --k 5 --k-grid-seeds 0 --methods KNORAU KNORAE --thr-runs 10 $COMMON --out results/repro_first_version

python des_nids_revision.py --data "data/cicids2018/*.csv" --classes "$C18" --dedup exact --seeds 0 1 2 3 4 --k 5 --k-grid-seeds 0 \
  --thr-runs 10 $COMMON --out results/ext_cicids2018
python aggregate.py results/ext_cicids2018
