# ===== step_11.py =====
# Compact diagnostics for the public reproducibility workflow.
stable_models = e1_boot_effects[(e1_boot_effects["ci_low"] > 0) | (e1_boot_effects["ci_high"] < 0)]
strict_coverage = E2[(E2["scenario"] == "strict_grounded") & (E2["min_history"] == PRIMARY_MIN_HISTORY)]
coverage_nonzero = bool(len(strict_coverage) and strict_coverage.iloc[0]["treated_lost_origins"] > 0)

hh = e1_heterogeneity.pivot(index="model", columns="subgroup", values="delta_MAE")
robust_not_tail = False
if "all_strict" in hh.columns and "delay<=7" in hh.columns:
    z = hh.loc[hh.index.intersection(stable_models["model"])]
    robust_not_tail = bool(
        ((np.sign(z["all_strict"]) == np.sign(z["delay<=7"])) & z["delay<=7"].notna()).any()
    ) if len(z) else False

summary = {
    **summary_gate,
    "primary_min_history": PRIMARY_MIN_HISTORY,
    "e1_target_days": E1_TARGET_DAYS,
    "global_target_days": GLOBAL_TARGET_DAYS,
    "stable_effect_models": stable_models["model"].tolist(),
    "local_winner_change_bootstrap_probability": float(e1_winner_change_prob),
    "local_bootstrap_mean_kendall_tau": float(e1_tau_summary.iloc[0]["mean_tau"]),
    "coverage_nonzero": coverage_nonzero,
    "tail_robustness_pass": robust_not_tail,
    "persistent_data_cache": str(CORE_PARQUET),
}

audit.to_parquet(OUTPUT_DIR/"series_audit.parquet", index=False)
E1.to_parquet(OUTPUT_DIR/"E1_local_predictions.parquet", index=False)
e1_series_loss.to_csv(OUTPUT_DIR/"E1_series_losses.csv", index=False)
e1_rank.to_csv(OUTPUT_DIR/"E1_local_rankings.csv", index=False)
e1_target_profile.to_csv(OUTPUT_DIR/"E1_target_day_effect_profile.csv", index=False)
e1_boot_effects.to_csv(OUTPUT_DIR/"E1_product_bootstrap_effects.csv", index=False)
e1_winner.to_csv(OUTPUT_DIR/"E1_winner_probabilities.csv", index=False)
e1_tau_summary.to_csv(OUTPUT_DIR/"E1_bootstrap_kendall_tau_summary.csv", index=False)
e1_heterogeneity.to_csv(OUTPUT_DIR/"E1_delay_heterogeneity.csv", index=False)
E2.to_csv(OUTPUT_DIR/"E2_eligibility.csv", index=False)
scaled_metrics.to_csv(OUTPUT_DIR/"scaled_metric_sensitivity.csv", index=False)
min_history_sensitivity.to_csv(OUTPUT_DIR/"minimum_history_sensitivity.csv", index=False)
if RUN_GLOBAL_E3 and len(E3_summary):
    E3_summary.to_csv(OUTPUT_DIR/"E3_global_target_summaries.csv", index=False)
    E3_rank.to_csv(OUTPUT_DIR/"E3_global_rankings.csv", index=False)
    E3_bootstrap.to_csv(OUTPUT_DIR/"E3_global_bootstrap.csv", index=False)
with open(OUTPUT_DIR/"main_experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Models with bootstrap intervals excluding zero:", stable_models["model"].tolist())
print("Eligibility loss observed:", coverage_nonzero)
print("Tail robustness diagnostic:", robust_not_tail)
print(json.dumps(summary, indent=2))
print("Saved:", OUTPUT_DIR)
print("Persistent data cache:", CORE_PARQUET)
