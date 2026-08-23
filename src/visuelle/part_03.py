# ===== step_11.py =====
scaled_rows = []
x = primary_pred

for model, g in x.groupby("model"):
    for metric in ["mase", "rmsse"]:
        fr = g[f"fixed_{metric}_release"].replace([np.inf, -np.inf], np.nan)
        ft = g[f"fixed_{metric}_trim"].replace([np.inf, -np.inf], np.nan)
        cr = g[f"conv_{metric}_release"].replace([np.inf, -np.inf], np.nan)
        ct = g[f"conv_{metric}_trim"].replace([np.inf, -np.inf], np.nan)

        m_fixed = fr.notna() & ft.notna()
        m_conv = cr.notna() & ct.notna()

        scaled_rows.append({
            "model": model,
            "metric": metric.upper(),
            "fixed_release": fr[m_fixed].mean(),
            "fixed_trim": ft[m_fixed].mean(),
            "fixed_delta": (ft[m_fixed] - fr[m_fixed]).mean(),
            "convention_release": cr[m_conv].mean(),
            "convention_trim": ct[m_conv].mean(),
            "convention_delta": (ct[m_conv] - cr[m_conv]).mean(),
            "fixed_valid_n": int(m_fixed.sum()),
            "convention_valid_n": int(m_conv.sum()),
        })

scaled_metrics = pd.DataFrame(scaled_rows)
display(scaled_metrics)

heterogeneity = (
    primary_pred.groupby(["first_sale_week", "model"], as_index=False)
    .agg(
        n_predictions=("delta_ae", "size"),
        n_products=("product_id", "nunique"),
        mean_delta_ae=("delta_ae", "mean"),
        median_delta_ae=("delta_ae", "median"),
        mae_release=("ae_release", "mean"),
        mae_trim=("ae_trim", "mean"),
    )
)
heterogeneity["pct_mae_change"] = (
    heterogeneity["mae_trim"] / heterogeneity["mae_release"] - 1
) * 100

display(heterogeneity.head(30))

# ===== step_12.py =====
sens_rows = []
for mh in MIN_HISTORY_SENSITIVITY:
    _, sl, rk = summarize_local(local_pred, mh)
    best_rel = rk.sort_values("MAE_release").iloc[0]["model"]
    best_trim = rk.sort_values("MAE_trim").iloc[0]["model"]
    tau = stats.kendalltau(rk["rank_release"], rk["rank_trim"]).statistic
    rank_order_changed = bool((rk["rank_release"] != rk["rank_trim"]).any())
    sens_rows.append({
        "min_history": mh,
        "best_release": best_rel,
        "best_trim": best_trim,
        "winner_changed": best_rel != best_trim,
        "rank_order_changed": rank_order_changed,
        "kendall_tau": tau,
        "max_abs_rank_shift": float((rk["rank_trim"] - rk["rank_release"]).abs().max()),
    })
sensitivity_ranks = pd.DataFrame(sens_rows)
display(sensitivity_ranks)

tsb_alpha_rows = []
if RUN_TSB_ALPHA_SENSITIVITY:
    tsb_base = primary_pred.drop_duplicates(["product_id", "store_id", "target_week"])[
        ["product_id", "store_id", "target_week", "first_sale_week", "y"]
    ].copy()
    key_to_rid = {
        (str(r.external_code), str(r.retail)): int(r.row_id)
        for r in strict.itertuples(index=False)
    }
    for alpha in TSB_ALPHAS:
        vals = []
        for r in tsb_base.itertuples(index=False):
            rid = key_to_rid[(str(r.product_id), str(r.store_id))]
            y = Y[rid]
            f = int(r.first_sale_week)
            t = int(r.target_week)
            m1 = TSB(alpha_d=alpha, alpha_p=alpha, alias="TSB")
            m2 = TSB(alpha_d=alpha, alpha_p=alpha, alias="TSB")
            pr = one_step_model(y[:t], m1)
            pt = one_step_model(y[f:t], m2)
            vals.append(abs(float(r.y)-pt) - abs(float(r.y)-pr))
        tsb_alpha_rows.append({
            "alpha_d": alpha,
            "alpha_p": alpha,
            "n_predictions": len(vals),
            "mean_delta_ae": float(np.mean(vals)),
            "median_delta_ae": float(np.median(vals)),
        })
tsb_alpha_sensitivity = pd.DataFrame(tsb_alpha_rows)
if len(tsb_alpha_sensitivity):
    display(tsb_alpha_sensitivity)

primary_failures = failures[
    (failures["release_history_len"] >= PRIMARY_MIN_HISTORY)
    & (failures["trim_history_len"] >= PRIMARY_MIN_HISTORY)
] if len(failures) else failures
primary_success_n = int(len(primary_pred))
failure_rate = len(primary_failures) / max(1, primary_success_n + len(primary_failures))

stable_effects = effects[(effects["ci_low"] > 0) | (effects["ci_high"] < 0)]
rank_reordering_stable = bool(sensitivity_ranks["rank_order_changed"].sum() >= 2)
winner_switch_stable = bool(sensitivity_ranks["winner_changed"].sum() >= 2)

if failure_rate > 0.01:
    verdict = "REVIEW"
    reason = f"Canonical model failure rate too high: {failure_rate:.2%}"
elif len(stable_effects) == 0:
    verdict = "KILL_OR_REDUCE"
    reason = "Canonical implementations show no stable product-bootstrap history-start effect."
elif not rank_reordering_stable:
    verdict = "REVIEW"
    reason = "Model-level effects exist, but rank reordering is not robust to min-history sensitivity."
elif winner_switch_stable:
    verdict = "SURVIVE"
    reason = "Canonical effects, rank reordering, and winner switching all survive robustness checks."
else:
    verdict = "SURVIVE_MUTATE"
    reason = "Canonical model effects and rank reordering survive, while the top-ranked winner remains stable."

print("#"*72)
print("MAIN EXPERIMENT DECISION:", verdict)
print(reason)
print("#"*72)

# ===== step_13.py =====
audit.to_csv(OUTPUT_DIR/"series_audit.csv", index=False)
local_pred.to_parquet(OUTPUT_DIR/"E1_local_predictions.parquet", index=False)
series_loss.to_csv(OUTPUT_DIR/"E1_series_losses.csv", index=False)
local_rank.to_csv(OUTPUT_DIR/"E1_local_rankings.csv", index=False)
rank_shifts.to_csv(OUTPUT_DIR/"E1_rank_shifts.csv", index=False)
effects.to_csv(OUTPUT_DIR/"E1_product_bootstrap_effects.csv", index=False)
winner.to_csv(OUTPUT_DIR/"E1_winner_probabilities.csv", index=False)
bootstrap_tau.to_csv(OUTPUT_DIR/"E1_bootstrap_kendall_tau_replicates.csv", index=False)
bootstrap_tau_summary.to_csv(OUTPUT_DIR/"E1_bootstrap_kendall_tau_summary.csv", index=False)
eligibility.to_csv(OUTPUT_DIR/"E2_eligibility.csv", index=False)
global_summary.to_csv(OUTPUT_DIR/"E3_global_target_summaries.csv", index=False)
global_rank.to_csv(OUTPUT_DIR/"E3_global_rankings.csv", index=False)
global_winner.to_csv(OUTPUT_DIR/"E3_global_winner_probabilities.csv", index=False)
scaled_metrics.to_csv(OUTPUT_DIR/"scaled_metric_sensitivity.csv", index=False)
heterogeneity.to_csv(OUTPUT_DIR/"first_sale_week_effect_heterogeneity.csv", index=False)
sensitivity_ranks.to_csv(OUTPUT_DIR/"minimum_history_sensitivity.csv", index=False)
failures.to_csv(OUTPUT_DIR/"model_failures.csv", index=False)
if len(tsb_alpha_sensitivity):
    tsb_alpha_sensitivity.to_csv(OUTPUT_DIR/"TSB_alpha_sensitivity.csv", index=False)

summary = {
    "all_series_n": int(len(audit)),
    "global_nonnegative_n": int(len(global_pop)),
    "strict_treated_n": int(len(strict)),
    "strict_treated_share_all": float(len(strict)/len(audit)),
    "strict_unique_products": int(strict["external_code"].nunique()),
    "strict_unique_stores": int(strict["retail"].nunique()),
    "primary_min_history": PRIMARY_MIN_HISTORY,
    "model_failure_rate": float(failure_rate),
    "local_winner_change_bootstrap_probability": float(winner_change_prob),
    "local_bootstrap_kendall_tau_mean": float(bootstrap_tau_summary.iloc[0]["mean_tau"]),
    "global_winner_change_bootstrap_probability": float(global_winner_change_prob),
    "rank_reordering_stable": bool(rank_reordering_stable),
    "winner_switch_stable": bool(winner_switch_stable),
    "verdict": verdict,
    "verdict_reason": reason,
    "persistent_data_cache": str(SALES_CACHE_PARQUET),
}
with open(OUTPUT_DIR/"main_experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("Saved:", OUTPUT_DIR)
print("Persistent data cache:", SALES_CACHE_PARQUET)
