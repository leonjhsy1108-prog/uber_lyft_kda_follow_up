import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from common import (CONTROLS, FAST_TRAIN_ROWS, FAST_TREES, FEATURES, OUT_ROOT, PERM_REPEATS,
                    PERM_ROWS, SECONDARY, SECONDARY_GROUPS, SEED, TARGET, barh_with_error,
                    chronological_split, grouped_permutation_importance, load_data,
                    make_forest, regression_metrics, sample_rows)

OUT = OUT_ROOT / "secondary_importance"
OUT.mkdir(parents=True, exist_ok=True)
data = load_data()
train, test, split_day = chronological_split(data)
train_sample = sample_rows(train, FAST_TRAIN_ROWS)
print(f"Grouped-ablation configuration: {len(train_sample):,} training quotes, {FAST_TREES} trees per model.")

def evaluate(columns, label):
    print(f"  training: {label}", flush=True)
    model = make_forest(columns, n_estimators=FAST_TREES)
    model.fit(train_sample[columns], train_sample[TARGET])
    m = regression_metrics(test[TARGET], model.predict(test[columns]))
    return model, {"model": label, "test_R2": m["R2"], "test_MAE": m["MAE"],
                   "test_RMSE": m["RMSE"], "test_MSE": m["MSE"]}

_, baseline = evaluate(CONTROLS, "Product + distance")
full_model, full = evaluate(FEATURES, "All drivers")
rows, ablations = [baseline, full], []
for group, excluded in SECONDARY_GROUPS.items():
    kept = [c for c in FEATURES if c not in excluded]
    _, without = evaluate(kept, f"Without {group}")
    rows.append(without)
    ablations.append({"group": group,
                      "delta_R2_when_removed": full["test_R2"] - without["test_R2"],
                      "increase_in_MAE_when_removed_USD": without["test_MAE"] - full["test_MAE"],
                      "increase_in_MSE_when_removed": without["test_MSE"] - full["test_MSE"]})

performance = pd.DataFrame(rows)
ablation = pd.DataFrame(ablations).sort_values("delta_R2_when_removed", ascending=False)
whole = (baseline["test_MSE"] - full["test_MSE"]) / baseline["test_MSE"]
print("\nModel comparison:\n", performance[["model", "test_MAE", "test_RMSE", "test_R2"]].round(4).to_string(index=False))
print(f"\nAll secondary features together reduce baseline test MSE by {100 * whole:.2f}%.")
print("\nSecondary group ablation (R² lost after excluding a group and retraining):\n",
      ablation.round(4).to_string(index=False))

perm_rows = sample_rows(test, PERM_ROWS)
group_perm = grouped_permutation_importance(full_model, perm_rows[FEATURES], perm_rows[TARGET],
                                            SECONDARY_GROUPS)
individual = permutation_importance(full_model, perm_rows[FEATURES], perm_rows[TARGET], scoring="r2",
                                    n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
individual = (pd.DataFrame({"driver": FEATURES, "importance_mean_R2_drop": individual.importances_mean,
                            "importance_std": individual.importances_std})
              .query("driver in @SECONDARY").sort_values("importance_mean_R2_drop", ascending=False))
print(f"\nGrouped permutation importance ({len(perm_rows):,} rows x {PERM_REPEATS} repeats), "
      "product and distance left unchanged:\n", group_perm.round(4).to_string(index=False))
print("\nIndividual secondary driver permutation importance:\n", individual.round(4).to_string(index=False))

performance.to_csv(OUT / "model_performance.csv", index=False)
ablation.to_csv(OUT / "secondary_group_ablation.csv", index=False)
group_perm.rename(columns={"importance_mean_r2_drop": "importance_mean_R2_drop"}).to_csv(
    OUT / "group_permutation_importance.csv", index=False)
individual.to_csv(OUT / "individual_permutation_importance.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
d = ablation.sort_values("delta_R2_when_removed")
barh_with_error(axes[0], d["group"], d["delta_R2_when_removed"], None,
                "Test R² decrease when removed", "Group ablation (retrained models)")
d = group_perm.sort_values("importance_mean_r2_drop")
barh_with_error(axes[1], d["group"], d["importance_mean_r2_drop"], d["importance_std"],
                "Test-sample R² decrease when permuted (± SD)", "Grouped permutation (full model)")
fig.suptitle("Importance of secondary drivers beyond ride product and distance")
fig.tight_layout()
fig.savefig(OUT / "secondary_group_importance.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
d = individual.sort_values("importance_mean_R2_drop")
barh_with_error(ax, d["driver"], d["importance_mean_R2_drop"], d["importance_std"],
                "Test-sample R² decrease when permuted (± SD)", "Individual secondary driver importance")
fig.tight_layout()
fig.savefig(OUT / "individual_secondary_importance.png", dpi=160)

print("\nNotes:")
print("- Ride product and distance are controls, not candidate secondary drivers.")
print("- Ablation and permutation scores are not directly comparable or additive.")
print("- Recorded Uber surge multiplier is always 1, so pooled surge importance is Lyft-driven.")
print(f"\nSaved outputs to: {OUT}")
