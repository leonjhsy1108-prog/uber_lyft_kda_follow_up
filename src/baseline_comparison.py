import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CONTROLS, FEATURES, MAIN_TRAIN_ROWS, OUT_ROOT, TARGET, chronological_split,
                    load_data, make_forest, regression_metrics, sample_rows)

OUT = OUT_ROOT / "baseline_comparison"
OUT.mkdir(parents=True, exist_ok=True)

data = load_data()
train, test, split_day = chronological_split(data)
train_sample = sample_rows(train, MAIN_TRAIN_ROWS)
print(f"Both models fitted on the same {len(train_sample):,}-row training sample.")

specs = {"Product + distance": ("pred_baseline", CONTROLS),
         "Product + distance + secondary": ("pred_expanded", FEATURES)}
rows = []
predictions = test[["id", "local_date", "cab_type", "name", "price", "distance",
                    "surge_multiplier", "local_hour"]].copy()
for label, (key, cols) in specs.items():
    start = time.time()
    model = make_forest(cols)
    model.fit(train_sample[cols], train_sample[TARGET])
    predictions[key] = model.predict(test[cols])
    rows.append({"model": label, "train_sample_rows": len(train_sample), "test_rows": len(test),
                 "test_start": str(split_day), **regression_metrics(test[TARGET], predictions[key])})
    print(f"  {label}: fitted in {time.time() - start:,.0f}s")

performance = pd.DataFrame(rows)
base, expanded = performance.iloc[0], performance.iloc[1]
summary = pd.DataFrame([{
    "delta_test_R2": expanded["R2"] - base["R2"],
    "MAE_reduction_USD": base["MAE"] - expanded["MAE"],
    "remaining_baseline_MSE_reduced_pct": 100 * (base["MSE"] - expanded["MSE"]) / base["MSE"],
}])

print("\nChronological held-out test performance:")
print(performance[["model", "MAE", "RMSE", "R2"]].round(4).to_string(index=False))
print(f"\nSecondary factors change test R² by {summary.delta_test_R2[0]:+.4f}, "
      f"MAE by ${-summary.MAE_reduction_USD[0]:+.4f}, and reduce baseline MSE by "
      f"{summary.remaining_baseline_MSE_reduced_pct[0]:.2f}%.")
print("These are predictive improvements, not causal shares of price variation.")

performance.to_csv(OUT / "model_performance.csv", index=False)
summary.to_csv(OUT / "incremental_value.csv", index=False)
predictions.to_csv(OUT / "test_predictions.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
labels = ["Controls only", "+ Secondary drivers"]
for i, metric in enumerate(["MAE", "RMSE"]):
    axes[0].bar(np.arange(2) + (i - 0.5) * 0.32, performance[metric], width=0.32, label=metric)
axes[0].set_xticks(range(2), labels)
axes[0].set(ylabel="Prediction error ($)", title="Held-out prediction error")
axes[0].legend()
axes[1].bar(labels, performance["R2"])
axes[1].set(ylim=(0, 1.05), ylabel="Test R²", title="Held-out explanatory performance")
for i, v in enumerate(performance["R2"]):
    axes[1].text(i, min(v + 0.015, 1.02), f"{v:.4f}", ha="center")
fig.suptitle("Does adding secondary factors improve fare prediction?")
fig.tight_layout()
fig.savefig(OUT / "baseline_comparison.png", dpi=160)
print(f"\nSaved results to: {OUT}")
