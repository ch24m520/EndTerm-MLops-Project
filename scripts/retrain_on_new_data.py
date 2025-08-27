#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retraining Orchestrator
- Checks drift (PSI/Chi-square) between baseline and new data
- If drift detected (exit code 2 from drift_detect.py), triggers retraining
- After training, calls FastAPI /reload-model to hot-reload Production model

Usage:
  python scripts/retrain_on_new_data.py \
    --new-csv data/incoming/new_batch.csv \
    --baseline data/processed/train_baseline.csv \
    --api http://127.0.0.1:8000

Options:
  --force         retrain even when no drift is detected
  --use-dvc       DVC add/commit/push the new CSV (requires DVC remote configured)
  --drift-report  path to write drift JSON report (default: reports/drift_report.json)
"""

import argparse
import subprocess
import sys
import requests
from pathlib import Path

def run(cmd: str) -> int:
    """Run a shell command and return its exit code (no raise)."""
    print(f"[cmd] {cmd}")
    return subprocess.call(cmd, shell=True)

def must_run(cmd: str) -> None:
    """Run a shell command and exit on failure."""
    rc = run(cmd)
    if rc != 0:
        print(f"[err] command failed ({rc}) -> {cmd}")
        sys.exit(rc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-csv", required=True, help="Incoming batch to compare/train on")
    ap.add_argument("--baseline", default="data/processed/train_baseline.csv", help="Baseline (train) CSV")
    ap.add_argument("--drift-report", default="reports/drift_report.json", help="Where to write drift report JSON")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="FastAPI base URL")
    ap.add_argument("--use-dvc", action="store_true", help="Track new data with DVC add/commit/push")
    ap.add_argument("--force", action="store_true", help="Retrain even if no drift is detected")
    args = ap.parse_args()

    Path("reports").mkdir(parents=True, exist_ok=True)

    # 0) Optionally track new data with DVC (if you use DVC for data)
    if args.use_dvc:
        must_run(f'dvc add "{args.new_csv}"')
        must_run('git add .')
        must_run('git commit -m "data: add new batch via DVC"')
        # dvc push requires remote already configured
        must_run('dvc push')

    # 1) Drift detection gate (scripts/drift_detect.py returns:
    #    0 -> no significant drift, 2 -> drift detected)
    drift_rc = run(
        f'python scripts/drift_detect.py --baseline "{args.baseline}" '
        f'--new "{args.new_csv}" --out "{args.drift_report}"'
    )

    retrain_needed = False
    if drift_rc == 2:
        print("[drift] significant drift detected -> retrain will be triggered")
        retrain_needed = True
    elif drift_rc == 0:
        print("[drift] no significant drift")
        retrain_needed = args.force
        if args.force:
            print("[retrain] --force specified -> retrain anyway")
    else:
        print("[warn] drift check returned unexpected code "
              f"({drift_rc}); proceed cautiously (use --force to retrain regardless).")
        if not args.force:
            print("[exit] neither drift nor --force: stopping without retrain")
            sys.exit(0)
        retrain_needed = True

    if not retrain_needed:
        print("[skip] retrain skipped (no drift and no --force)")
        sys.exit(0)

    # 2) Retrain (your trainer should log to MLflow and register/promote to Production)
    must_run('python src/training/train_spark_mllib.py')

    # 3) Ask API to hot-reload Production model
    try:
        r = requests.post(f"{args.api}/reload-model", timeout=15)
        r.raise_for_status()
        print("[api] reload ok:", r.json())
    except Exception as e:
        print("[api] reload failed:", e)
        sys.exit(1)

    print("[done] retrain + promote + hot reload complete")

if __name__ == "__main__":
    main()
