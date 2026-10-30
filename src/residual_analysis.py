import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_ROOT, PRICE_BANDS, regression_metrics

PRED = OUT_ROOT / "baseline_comparison" / "test_predictions.csv"
OUT = OUT_ROOT / "residual_analysis"
OUT.mkdir(parents=True, exist_ok=True)
if not PRED.exists():
    raise SystemExit(f"{PRED} not found - run baseline_comparison.py first.")

test = pd.read_csv(PRED, parse_dates=["local_date"])
for key in ["baseline", "expanded"]:
    test[f"residual_{key}"] = test["price"] - test[f"pred_{key}"]   # positive = underprediction
    test[f"abs_error_{key}"] = test[f"residual_{key}"].abs()
test["price_band"] = pd.cut(test["price"], **PRICE_BANDS)

summary = pd.DataFrame([{"model": key.capitalize(), "test_rows": len(test),
                         **regression_metrics(test["price"], test[f"pred_{key}"]),
                         "mean_residual": test[f"residual_{key}"].mean()}
                        for key in ["baseline", "expanded"]])
summary.to_csv(OUT / "model_residual_summary.csv", index=False)
print("Held-out model performance and overall bias:\n", summary.round(4).to_string(index=False))

def grouped_summary(frame, by):
    result = frame.groupby(by, observed=True).agg(
        count=("price", "size"), mean_actual_price=("price", "mean"),
        baseline_mean_residual=("residual_baseline", "mean"),
        expanded_mean_residual=("residual_expanded", "mean"),
        baseline_MAE=("abs_error_baseline", "mean"),
        expanded_MAE=("abs_error_expanded", "mean"),
    ).reset_index()
    result["MAE_improvement_USD"] = result["baseline_MAE"] - result["expanded_MAE"]
    return result

by_company = grouped_summary(test, ["cab_type"])
by_band = grouped_summary(test, ["price_band"])
by_surge = grouped_summary(test.loc[test["cab_type"] == "Lyft"], ["surge_multiplier"])
by_hour = grouped_summary(test, ["local_hour"])
by_company.to_csv(OUT / "residuals_by_company.csv", index=False)
by_band.to_csv(OUT / "residuals_by_price_band.csv", index=False)
by_surge.to_csv(OUT / "lyft_residuals_by_surge.csv", index=False)
by_hour.to_csv(OUT / "residuals_by_hour.csv", index=False)

print("\nResiduals by company (positive = underprediction):\n", by_company.round(3).to_string(index=False))
print("\nResiduals by actual price range:\n", by_band.round(3).to_string(index=False))
visible = by_surge.loc[by_surge["count"] >= 30].sort_values("surge_multiplier")
print("\nLyft residuals by surge multiplier (count >= 30):\n", visible.round(3).to_string(index=False))

if not visible.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    pos, width = np.arange(len(visible)), 0.38
    ax.bar(pos - width / 2, visible["baseline_mean_residual"], width, label="Product + distance")
    ax.bar(pos + width / 2, visible["expanded_mean_residual"], width, label="+ Secondary factors")
    ax.set_xticks(pos, [f"{v:g}×\n(n={n:,})" for v, n in zip(visible["surge_multiplier"], visible["count"])])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="Mean residual: actual − predicted ($)",
           title="Lyft: does surge explain residual price differences?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lyft_residuals_by_surge.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
pos, width = np.arange(len(by_band)), 0.38
ax.bar(pos - width / 2, by_band["baseline_MAE"], width, label="Product + distance")
ax.bar(pos + width / 2, by_band["expanded_MAE"], width, label="+ Secondary factors")
ax.set_xticks(pos, by_band["price_band"].astype(str))
ax.set(ylabel="Held-out MAE ($)", title="Prediction errors by quoted fare range")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mae_by_price_band.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-15, 15, 90)
for key in ["baseline", "expanded"]:
    ax.hist(test[f"residual_{key}"], bins=bins, density=True, alpha=0.45, label=key.capitalize())
ax.axvline(0, color="black", linewidth=1)
ax.set(xlabel="Residual: actual − predicted ($)", ylabel="Density",
       title="Held-out residual distribution (display restricted to ±$15)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "residual_distribution.png", dpi=160)

print(f"\nSaved residual diagnostics to: {OUT}")
print("Interpretation: residual patterns show associations, not causal effects; surge > 1 is Lyft-only.")
