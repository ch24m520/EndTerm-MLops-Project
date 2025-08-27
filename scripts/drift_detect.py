#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drift detector (PSI for numeric, Chi-square for categorical).
Exit code 0 => no significant drift, 2 => drift flagged.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency  # if you want to avoid SciPy, see comment below

# ---------- CONFIG ----------
NUMERIC_PSI_THRESHOLD = 0.25   # 0.1-0.25 = moderate; >0.25 = significant drift
CATEG_P_THRESHOLD     = 0.01   # p < 0.01 => reject H0 (distribution changed)
N_BINS                = 10     # PSI bin count for numeric
# ----------------------------

def psi(expected, actual, bins=10):
    """Population Stability Index for two 1-D arrays."""
    expected = np.asarray(expected).astype(float)
    actual   = np.asarray(actual).astype(float)
    # Remove nans
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    # Bin edges from expected
    bin_edges = np.quantile(expected, np.linspace(0, 1, bins+1))
    # Avoid duplicate edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:  # too few distinct values
        bin_edges = np.linspace(expected.min(), expected.max(), bins+1)

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual,   bins=bin_edges)

    exp_perc = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1.0)
    act_perc = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1.0)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

def chi_square_pvalue(base_cat: pd.Series, new_cat: pd.Series):
    """Chi-square p-value on categorical distributions."""
    bvc = base_cat.value_counts()
    nvc = new_cat.value_counts()
    cats = sorted(set(bvc.index).union(nvc.index))
    obs = np.array([ [bvc.get(c,0), nvc.get(c,0)] for c in cats ])
    # If all zeros or single category, return 1.0 (no evidence)
    if obs.sum() == 0 or obs.shape[0] < 2:
        return 1.0
    chi2, p, _, _ = chi2_contingency(obs.T)  # compare distributions
    return float(p)

def detect_dtypes(df: pd.DataFrame):
    num_cols, cat_cols = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="CSV path used for training (baseline)")
    ap.add_argument("--new", required=True, help="New CSV path to compare")
    ap.add_argument("--id-cols", nargs="*", default=[], help="Columns to ignore (ids, targets, etc.)")
    ap.add_argument("--out", default="drift_report.json", help="Output report path")
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    new  = pd.read_csv(args.new)
    base = base.drop(columns=[c for c in args.id_cols if c in base.columns], errors="ignore")
    new  = new.drop(columns=[c for c in args.id_cols if c in new.columns], errors="ignore")

    # keep only intersecting columns
    common = [c for c in base.columns if c in new.columns]
    base = base[common]
    new  = new[common]

    num_cols, cat_cols = detect_dtypes(base)

    results = {"numeric": {}, "categorical": {}, "thresholds": {
        "psi": NUMERIC_PSI_THRESHOLD, "chi_square_p": CATEG_P_THRESHOLD
    }}

    drift_flag = False

    # Numeric drift via PSI
    for c in num_cols:
        score = psi(base[c].values, new[c].values, bins=N_BINS)
        if np.isnan(score):
            status = "insufficient_data"
        else:
            status = "drift" if score > NUMERIC_PSI_THRESHOLD else "ok"
            drift_flag = drift_flag or (status == "drift")
        results["numeric"][c] = {"psi": None if np.isnan(score) else round(score, 4), "status": status}

    # Categorical drift via Chi-square p-value
    for c in cat_cols:
        p = chi_square_pvalue(base[c].astype(str), new[c].astype(str))
        status = "drift" if p < CATEG_P_THRESHOLD else "ok"
        results["categorical"][c] = {"p_value": round(p, 6), "status": status}
        drift_flag = drift_flag or (status == "drift")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[drift] report saved -> {args.out}")
    if drift_flag:
        print("[drift] significant drift detected")
        sys.exit(2)
    print("[drift] no significant drift")
    sys.exit(0)

if __name__ == "__main__":
    main()
