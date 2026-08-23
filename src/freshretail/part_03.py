# ===== step_08.py =====
scaled_rows = []
for model, g in primary_e1.groupby("model"):
    for metric in ["mase", "rmsse"]:
        fr = g[f"fixed_{metric}_release"].replace([np.inf, -np.inf], np.nan)
        ft = g[f"fixed_{metric}_trim"].replace([np.inf, -np.inf], np.nan)
        cr = g[f"conv_{metric}_release"].replace([np.inf, -np.inf], np.nan)
        ct = g[f"conv_{metric}_trim"].replace([np.inf, -np.inf], np.nan)
        mf = fr.notna() & ft.notna(); mc = cr.notna() & ct.notna()
        scaled_rows.append({
            "model": model, "metric": metric.upper(),
            "fixed_release": fr[mf].mean(), "fixed_trim": ft[mf].mean(),
            "fixed_delta": (ft[mf]-fr[mf]).mean(), "fixed_valid_n": int(mf.sum()),
            "convention_release": cr[mc].mean(), "convention_trim": ct[mc].mean(),
            "convention_delta": (ct[mc]-cr[mc]).mean(), "convention_valid_n": int(mc.sum()),
        })
scaled_metrics = pd.DataFrame(scaled_rows)
display(scaled_metrics)

sens = []
for mh in MIN_HISTORY_SENSITIVITY:
    _, _, rk = summarize_e1(E1, mh)
    br = rk.sort_values("MAE_release").iloc[0]["model"]
    bt = rk.sort_values("MAE_trim").iloc[0]["model"]
    sens.append({
        "min_history": mh,
        "best_release": br,
        "best_trim": bt,
        "winner_changed": br != bt,
        "rank_order_changed": bool((rk["rank_release"] != rk["rank_trim"]).any()),
        "kendall_tau": float(stats.kendalltau(rk["rank_release"], rk["rank_trim"]).statistic),
        "max_abs_rank_shift": float((rk["rank_trim"]-rk["rank_release"]).abs().max()),
    })
min_history_sensitivity = pd.DataFrame(sens)
display(min_history_sensitivity)

# ===== step_09.py =====
def e3_checkpoint_path(target_day, scenario):
    return CHECKPOINT_DIR / f"E3_{scenario}_target_day_{target_day:02d}.parquet"

def e3_baseline_checkpoint_path(target_day):
    return CHECKPOINT_DIR / f"E3_baseline_target_day_{target_day:02d}.parquet"

def make_all_sample():
    x = audit[["row_id", "store_id", "product_id", "first_positive_day"]].copy()
    x["first_positive_day"] = x["first_positive_day"].fillna(0).astype(int)
    return x

all_sample = make_all_sample()

def run_e3_target(target_day, scenario, trim_sample):
    cp = e3_checkpoint_path(target_day, scenario)
    if RESUME_CHECKPOINTS and cp.exists() and cp.stat().st_size > 0:
        print(f"✅ E3 {scenario} day {target_day}: checkpoint hit")
        return pd.read_parquet(cp)

    baseline_cp = e3_baseline_checkpoint_path(target_day)
    if RESUME_CHECKPOINTS and baseline_cp.exists() and baseline_cp.stat().st_size > 0:
        print(f"✅ E3 baseline day {target_day}: checkpoint hit")
        rel_fc = pd.read_parquet(baseline_cp)
    else:
        print(f"E3 baseline day {target_day}: forecasting {len(all_sample):,} series ONCE")
        rel_long = build_long(all_sample, target_day, trimmed=False)
        rel_fc = sf_batch_forecast(rel_long).drop(columns=["ds"], errors="ignore")
        rel_fc.to_parquet(baseline_cp, index=False)
        del rel_long
        gc.collect()

    eligible_trim = trim_sample[
        (target_day - trim_sample["first_positive_day"].astype(int)) >= PRIMARY_MIN_HISTORY
    ].copy()
    trim_long = build_long(eligible_trim, target_day, trimmed=True)
    trim_fc = sf_batch_forecast(trim_long).drop(columns=["ds"], errors="ignore") if len(trim_long) else pd.DataFrame()

    meta = all_sample[["row_id", "store_id", "product_id"]].copy()
    meta["unique_id"] = meta["store_id"].astype(str) + "__" + meta["product_id"].astype(str)
    meta["y_true"] = Y[meta["row_id"].to_numpy(), target_day]
    wide = meta.merge(rel_fc, on="unique_id", how="inner", validate="one_to_one")

    summaries = []
    detail_cols = ["unique_id", "product_id", "store_id", "y_true"]
    detail = wide[detail_cols].copy()
    trim_maps = {}
    if len(trim_fc):
        trim_maps = {m: trim_fc.set_index("unique_id")[m] for m in MODEL_ALIASES}

    for m in MODEL_ALIASES:
        pred_r = wide[m].astype(float)
        pred_t = pred_r.copy()
        if len(trim_fc):
            mp = trim_maps[m]
            mask = wide["unique_id"].isin(mp.index)
            pred_t.loc[mask] = wide.loc[mask, "unique_id"].map(mp)
        ae_r = (wide["y_true"] - pred_r).abs()
        ae_t = (wide["y_true"] - pred_t).abs()
        detail[f"ae_{m}_release"] = ae_r.to_numpy()
        detail[f"ae_{m}_trim"] = ae_t.to_numpy()
        denom = wide["y_true"].sum()
        summaries.append({
            "scenario": scenario,
            "target_day": target_day,
            "model": m,
            "MAE_release": float(ae_r.mean()),
            "MAE_trim": float(ae_t.mean()),
            "delta_MAE": float(ae_t.mean()-ae_r.mean()),
            "WAPE_release": float(ae_r.sum()/denom) if denom else np.nan,
            "WAPE_trim": float(ae_t.sum()/denom) if denom else np.nan,
            "delta_WAPE": float((ae_t.sum()-ae_r.sum())/denom) if denom else np.nan,
            "n_series": int(len(wide)),
            "n_trim_reforecasted": int(len(eligible_trim)),
        })

    detail["scenario"] = scenario
    detail["target_day"] = target_day
    detail.to_parquet(cp, index=False)
    pd.DataFrame(summaries).to_csv(cp.with_suffix(".summary.csv"), index=False)

    del rel_fc, trim_long, trim_fc, wide
    gc.collect()
    return detail

E3_summary_parts = []
E3_detail_parts = []
if RUN_GLOBAL_E3:
    scenarios = [("strict_grounded", strict)]
    if RUN_BLANKET_FIRST_POSITIVE_E3:
        scenarios.append(("blanket_delayed_zero", blanket))

    for scenario, sample in scenarios:
        for t in GLOBAL_TARGET_DAYS:
            detail = run_e3_target(t, scenario, sample)
            E3_detail_parts.append(detail)
            sp = e3_checkpoint_path(t, scenario).with_suffix(".summary.csv")
            if sp.exists():
                E3_summary_parts.append(pd.read_csv(sp))

    E3_summary = pd.concat(E3_summary_parts, ignore_index=True)
    E3_detail = pd.concat(E3_detail_parts, ignore_index=True)

    E3_rank = (
        E3_summary.groupby(["scenario", "model"], as_index=False)
        .agg(
            MAE_release=("MAE_release", "mean"),
            MAE_trim=("MAE_trim", "mean"),
            delta_MAE=("delta_MAE", "mean"),
            WAPE_release=("WAPE_release", "mean"),
            WAPE_trim=("WAPE_trim", "mean"),
            delta_WAPE=("delta_WAPE", "mean"),
        )
    )
    E3_rank["rank_release"] = E3_rank.groupby("scenario")["MAE_release"].rank(method="min")
    E3_rank["rank_trim"] = E3_rank.groupby("scenario")["MAE_trim"].rank(method="min")
    display(E3_rank.sort_values(["scenario", "rank_release"]))
else:
    E3_summary = pd.DataFrame(); E3_detail = pd.DataFrame(); E3_rank = pd.DataFrame()

# ===== step_10.py =====
e3_boot_rows = []
if RUN_GLOBAL_E3 and len(E3_detail):
    for scenario in E3_detail["scenario"].unique():
        g = E3_detail[(E3_detail["scenario"] == scenario) & (E3_detail["target_day"] == max(GLOBAL_TARGET_DAYS))].copy()
        rel_cols = [f"ae_{m}_release" for m in MODEL_ALIASES]
        trim_cols = [f"ae_{m}_trim" for m in MODEL_ALIASES]
        pr = g.groupby(g["product_id"].astype(str))[rel_cols].mean()
        pt = g.groupby(g["product_id"].astype(str))[trim_cols].mean()
        common = pr.index.intersection(pt.index)
        R = pr.loc[common, rel_cols].to_numpy()
        T = pt.loc[common, trim_cols].to_numpy()
        rng = np.random.default_rng(RNG_SEED + 100 + len(e3_boot_rows))
        wr = np.zeros(len(MODEL_ALIASES), int); wt = np.zeros(len(MODEL_ALIASES), int)
        changed = 0; taus = []
        for b in range(BOOT_REPS):
            idx = rng.integers(0, len(common), size=len(common))
            mr = np.nanmean(R[idx], axis=0); mt = np.nanmean(T[idx], axis=0)
            ir, it = int(np.nanargmin(mr)), int(np.nanargmin(mt))
            wr[ir] += 1; wt[it] += 1; changed += int(ir != it)
            taus.append(stats.kendalltau(stats.rankdata(mr), stats.rankdata(mt)).statistic)
        for j, m in enumerate(MODEL_ALIASES):
            e3_boot_rows.append({
                "scenario": scenario, "model": m,
                "release_win_prob": wr[j]/BOOT_REPS,
                "trim_win_prob": wt[j]/BOOT_REPS,
                "winner_change_prob_all_models": changed/BOOT_REPS,
                "mean_kendall_tau": float(np.nanmean(taus)),
                "tau_ci_low": float(np.nanquantile(taus, .025)),
                "tau_ci_high": float(np.nanquantile(taus, .975)),
            })
E3_bootstrap = pd.DataFrame(e3_boot_rows)
if len(E3_bootstrap):
    display(E3_bootstrap.sort_values(["scenario", "release_win_prob"], ascending=[True, False]))
