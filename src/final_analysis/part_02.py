# ===== step_06.py =====
def pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return np.nan

table1 = pd.DataFrame([
    {"dataset": "Visuelle 2.0", "all_series": pick(vis_summary, "all_series_n"), "analysis_series": pick(vis_summary, "global_nonnegative_n"), "strict_treated": pick(vis_summary, "strict_treated_n"), "strict_share": pick(vis_summary, "strict_treated_share_all"), "unique_products": pick(vis_summary, "strict_unique_products"), "unique_stores": pick(vis_summary, "strict_unique_stores"), "history_interpretation": "release-aligned lifecycle history"},
    {"dataset": "FreshRetailNet-50K", "all_series": pick(fresh_summary, "series_n"), "analysis_series": pick(fresh_summary, "series_n"), "strict_treated": pick(fresh_summary, "strict_treated_n"), "strict_share": pick(fresh_summary, "strict_treated_share"), "unique_products": pick(fresh_summary, "strict_unique_products"), "unique_stores": pick(fresh_summary, "strict_unique_stores"), "history_interpretation": "stock-confirmed active-zero observation-window history"},
])

table2 = pd.concat([vis_point.assign(dataset="Visuelle").merge(vis_effects[["model", "ci_low", "ci_high", "ci_excludes_zero"]], on="model", how="left"), fresh_point.assign(dataset="FreshRetail").merge(fresh_effects[["model", "ci_low", "ci_high", "ci_excludes_zero"]], on="model", how="left")], ignore_index=True)
table3 = pd.concat([vis_winner.assign(dataset="Visuelle"), fresh_winner.assign(dataset="FreshRetail")], ignore_index=True)
table3b = pd.concat([vis_tau.assign(dataset="Visuelle"), fresh_tau.assign(dataset="FreshRetail")], ignore_index=True)
table4 = pd.concat([vis_e2.assign(dataset="Visuelle"), fresh_e2.assign(dataset="FreshRetail")], ignore_index=True)
table5 = pd.concat([vis_e3.assign(dataset="Visuelle"), fresh_e3.assign(dataset="FreshRetail")], ignore_index=True)
table6 = pd.concat([vis_scaled.assign(dataset="Visuelle"), fresh_scaled.assign(dataset="FreshRetail")], ignore_index=True)

exports = {"Table1_dataset_summary.csv": table1, "Table2_local_effects_corrected.csv": table2, "Table3_winner_probabilities_corrected.csv": table3, "Table3b_rank_stability_corrected.csv": table3b, "Table4_coverage_effect.csv": table4, "Table5_global_propagation.csv": table5, "Table6_scaled_metric_sensitivity.csv": table6, "FreshRetail_full_target_day_effect_profile.csv": fresh_profile, "FreshRetail_target_day_sign_audit.csv": fresh_time_audit, "Bootstrap_estimand_comparison.csv": estimand_comparison}
for name, df in exports.items():
    df.to_csv(TABLE_DIR/name, index=False)
vis_effects.to_csv(TABLE_DIR/"Visuelle_E1_bootstrap_series_weighted.csv", index=False)
fresh_effects.to_csv(TABLE_DIR/"FreshRetail_E1_bootstrap_series_weighted.csv", index=False)
vis_winner.to_csv(TABLE_DIR/"Visuelle_E1_winner_series_weighted.csv", index=False)
fresh_winner.to_csv(TABLE_DIR/"FreshRetail_E1_winner_series_weighted.csv", index=False)
vis_tau.to_csv(TABLE_DIR/"Visuelle_E1_tau_series_weighted.csv", index=False)
fresh_tau.to_csv(TABLE_DIR/"FreshRetail_E1_tau_series_weighted.csv", index=False)
print("Saved tables:", TABLE_DIR)

# ===== step_07.py =====
def plot_effects(effects, title, filename):
    d = effects.sort_values("delta_MAE").reset_index(drop=True)
    y = np.arange(len(d)); x = d["delta_MAE"].to_numpy(dtype=float)
    lo = d["ci_low"].to_numpy(dtype=float); hi = d["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3); ax.axvline(0, linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(d["model"]); ax.set_xlabel("ΔMAE (trimmed − grounded/release)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG_DIR/filename, dpi=220, bbox_inches="tight"); plt.show()

plot_effects(vis_effects, "Visuelle: local history-start effects", "visuelle_local_effects.png")
plot_effects(fresh_effects, "FreshRetail: local history-start effects", "freshretail_local_effects.png")

# ===== step_08.py =====
fig, ax = plt.subplots(figsize=(9, 5.8))
for model, g in fresh_profile.groupby("model"):
    g = g.sort_values("target_day"); ax.plot(g["target_day"], g["delta_MAE"], marker="o", label=model)
ax.axhline(0, linewidth=1); ax.set_xlabel("Target day / evaluation origin"); ax.set_ylabel("ΔMAE (trimmed − grounded)"); ax.set_title("FreshRetail: history-start sensitivity across evaluation origins"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR/"freshretail_target_day_profile.png", dpi=220, bbox_inches="tight"); plt.show()

# ===== step_09.py =====
stability = pd.DataFrame([{"dataset": "Visuelle", "mean_kendall_tau": vis_tau.iloc[0]["mean_tau"], "winner_change_prob": vis_tau.iloc[0]["winner_change_prob"]}, {"dataset": "FreshRetail", "mean_kendall_tau": fresh_tau.iloc[0]["mean_tau"], "winner_change_prob": fresh_tau.iloc[0]["winner_change_prob"]}])
x = np.arange(len(stability)); width = 0.36
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.bar(x - width/2, stability["mean_kendall_tau"], width, label="Mean Kendall τ"); ax.bar(x + width/2, stability["winner_change_prob"], width, label="Winner-change probability")
ax.set_xticks(x); ax.set_xticklabels(stability["dataset"]); ax.set_ylim(0, 1); ax.set_title("Local ranking sensitivity under corrected cluster bootstrap"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR/"local_rank_stability.png", dpi=220, bbox_inches="tight"); plt.show()

# ===== step_10.py =====
def plot_metric_sensitivity(df, dataset, filename):
    x = df[df["metric"].astype(str).str.upper().eq("MASE")].copy().sort_values("model").reset_index(drop=True); pos = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(9, 5.2)); ax.plot(pos, x["fixed_delta"], marker="o", label="Fixed denominator"); ax.plot(pos, x["convention_delta"], marker="o", label="Convention-specific denominator"); ax.axhline(0, linewidth=1)
    ax.set_xticks(pos); ax.set_xticklabels(x["model"], rotation=45, ha="right"); ax.set_ylabel("ΔMASE (trimmed − grounded/release)"); ax.set_title(f"{dataset}: forecast vs metric-normalization sensitivity"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR/filename, dpi=220, bbox_inches="tight"); plt.show()

plot_metric_sensitivity(vis_scaled, "Visuelle", "visuelle_metric_normalization.png")
plot_metric_sensitivity(fresh_scaled, "FreshRetail", "freshretail_metric_normalization.png")

# ===== step_11.py =====
def fmt(x, digits=3): return f"{float(x):.{digits}f}"
v = vis_tau.iloc[0]; f = fresh_tau.iloc[0]
vstable = vis_effects.loc[vis_effects["ci_excludes_zero"], "model"].tolist(); fstable = fresh_effects.loc[fresh_effects["ci_excludes_zero"], "model"].tolist()
story_lines = ["# arXiv v1 — FINAL TOPIC LOCK", "", "## Recommended title", "", "**When Does History Start Matter? Model-Specific Sensitivity and Global Robustness in Retail Forecast Evaluation**", "", "## Final FFR status", "", "**SURVIVE → MUTATE → LOCK**", "", "No additional dataset or SOTA forecasting model is required for arXiv v1.", "", "## Corrected E1 bootstrap", "", "Cluster unit: **product_id**", "Estimand: **series-weighted**", "", "### Visuelle", f"- mean Kendall tau: {fmt(v['mean_tau'])}", f"- 95% tau interval: [{fmt(v['ci_low'])}, {fmt(v['ci_high'])}]", f"- winner-change probability: {float(v['winner_change_prob']):.1%}", f"- models with ΔMAE CI excluding zero: {', '.join(vstable) if vstable else 'none'}", "", "### FreshRetail", f"- mean Kendall tau: {fmt(f['mean_tau'])}", f"- 95% tau interval: [{fmt(f['ci_low'])}, {fmt(f['ci_high'])}]", f"- winner-change probability: {float(f['winner_change_prob']):.1%}", f"- models with ΔMAE CI excluding zero: {', '.join(fstable) if fstable else 'none'}"]
story = "\n".join(story_lines); (PAPER_DIR/"paper_lock_summary.md").write_text(story); print(story)

# ===== step_12.py =====
manifest = {"paper_dir": str(PAPER_DIR), "figures": sorted([p.name for p in FIG_DIR.glob("*.png")]), "tables": sorted([p.name for p in TABLE_DIR.glob("*.csv")]), "bootstrap_reps": BOOT_REPS, "cluster_unit": "product_id", "local_estimand": "series_weighted", "e3_primary_evidence": "deterministic global score/rank propagation", "topic_status": "FINAL_LOCK"}
with open(PAPER_DIR/"final_manifest.json", "w") as f: json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2)); print("\n✅ Final postprocessing complete.")
