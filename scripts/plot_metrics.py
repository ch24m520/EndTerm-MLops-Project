# scripts/plot_metrics.py
import csv, os
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATH = Path("reports/training/runs.csv")
OUT_DIR  = Path("reports/training/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

if not rows:
    raise SystemExit("No rows in runs.csv")

# Convert types
def to_float(x, default=None):
    try: return float(x)
    except: return default

xs = list(range(1, len(rows)+1))
fit_time = [to_float(r["fit_time_sec"], 0.0) for r in rows]
acc      = [to_float(r["accuracy"], 0.0) for r in rows]
parts    = [int(r.get("shuffle_partitions", 0)) for r in rows]

# Plot 1: Accuracy over runs
plt.figure()
plt.plot(xs, acc, marker="o")
plt.title("Accuracy over runs")
plt.xlabel("Run index (time-ordered)")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_over_runs.png", dpi=160)

# Plot 2: Fit time vs shuffle partitions
plt.figure()
plt.scatter(parts, fit_time)
for i, (p, t) in enumerate(zip(parts, fit_time), start=1):
    plt.annotate(i, (p, t), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.title("Fit time vs shuffle_partitions")
plt.xlabel("shuffle_partitions")
plt.ylabel("fit_time_sec")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_DIR / "fit_time_vs_shuffle_partitions.png", dpi=160)

print(f"[ok] saved plots in {OUT_DIR}")
