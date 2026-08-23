# ===== step_05.py =====
def e1_checkpoint_path(target_day):
    return CHECKPOINT_DIR / f"E1_target_day_{target_day:02d}.parquet"

def run_e1_target(target_day):
    cp = e1_checkpoint_path(target_day)
    if RESUME_CHECKPOINTS and cp.exists() and cp.stat().st_size > 0:
        print(f"✅ E1 day {target_day}: checkpoint hit")
        return pd.read_parquet(cp)

    eligible = strict[
        (target_day >= MIN_HISTORY_BUILD)
        & ((target_day - strict["first_positive_day"].astype(int)) >= MIN_HISTORY_BUILD)
    ].copy()
    if eligible.empty:
        print(f"E1 day {target_day}: no eligible strict series")
        return pd.DataFrame()

    print(f"E1 day {target_day}: {len(eligible):,} eligible series")
    rel_long = build_long(eligible, target_day, trimmed=False)
    trim_long = build_long(eligible, target_day, trimmed=True)
    rel_fc = sf_batch_forecast(rel_long)
    trim_fc = sf_batch_forecast(trim_long)

    meta = uid_frame(eligible)
    meta["y_true"] = Y[meta["row_id"].to_numpy(), target_day]
    meta["target_day"] = target_day
    meta["release_history_len"] = target_day
    meta["trim_history_len"] = target_day - meta["first_positive_day"].astype(int)

    rel_fc = rel_fc.drop(columns=["ds"], errors="ignore")
    trim_fc = trim_fc.drop(columns=["ds"], errors="ignore")
    rel_fc = rel_fc.rename(columns={m: f"pred_release_{m}" for m in MODEL_ALIASES})
    trim_fc = trim_fc.rename(columns={m: f"pred_trim_{m}" for m in MODEL_ALIASES})
    wide = meta.merge(rel_fc, on="unique_id", how="inner", validate="one_to_one")
    wide = wide.merge(trim_fc, on="unique_id", how="inner", validate="one_to_one")

    scale_rows = []
    for r in eligible.itertuples(index=False):
        rid = int(r.row_id); f = int(r.first_positive_day)
        h_rel = Y[rid, :target_day]
        h_trim = Y[rid, f:target_day]
        scale_rows.append({
            "unique_id": str(r.store_id)+"__"+str(r.product_id),
            "scale_mae_release": scale_mae(h_rel),
            "scale_mae_trim": scale_mae(h_trim),
            "scale_rmse_release": scale_rmse(h_rel),
            "scale_rmse_trim": scale_rmse(h_trim),
        })
    wide = wide.merge(pd.DataFrame(scale_rows), on="unique_id", how="left", validate="one_to_one")

    rows = []
    for m in MODEL_ALIASES:
        g = wide[[
            "row_id", "store_id", "product_id", "unique_id", "first_positive_day",
            "target_day", "release_history_len", "trim_history_len", "y_true",
            "scale_mae_release", "scale_mae_trim", "scale_rmse_release", "scale_rmse_trim",
            f"pred_release_{m}", f"pred_trim_{m}"
        ]].copy()
        g["model"] = m
        g = g.rename(columns={f"pred_release_{m}": "pred_release", f"pred_trim_{m}": "pred_trim"})
        g["ae_release"] = (g["y_true"] - g["pred_release"]).abs()
        g["ae_trim"] = (g["y_true"] - g["pred_trim"]).abs()
        g["delta_ae"] = g["ae_trim"] - g["ae_release"]
        g["se_release"] = (g["y_true"] - g["pred_release"])**2
        g["se_trim"] = (g["y_true"] - g["pred_trim"])**2
        g["fixed_mase_release"] = g["ae_release"] / g["scale_mae_release"].replace(0, np.nan)
        g["fixed_mase_trim"] = g["ae_trim"] / g["scale_mae_release"].replace(0, np.nan)
        g["conv_mase_release"] = g["ae_release"] / g["scale_mae_release"].replace(0, np.nan)
        g["conv_mase_trim"] = g["ae_trim"] / g["scale_mae_trim"].replace(0, np.nan)
        g["fixed_rmsse_release"] = np.sqrt(g["se_release"]) / g["scale_rmse_release"].replace(0, np.nan)
        g["fixed_rmsse_trim"] = np.sqrt(g["se_trim"]) / g["scale_rmse_release"].replace(0, np.nan)
        g["conv_rmsse_release"] = np.sqrt(g["se_release"]) / g["scale_rmse_release"].replace(0, np.nan)
        g["conv_rmsse_trim"] = np.sqrt(g["se_trim"]) / g["scale_rmse_trim"].replace(0, np.nan)
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(cp, index=False)
    del rel_long, trim_long, rel_fc, trim_fc, wide
    gc.collect()
    return out

e1_parts = []
for t in E1_TARGET_DAYS:
    part = run_e1_target(t)
    if len(part):
        e1_parts.append(part)

E1 = pd.concat(e1_parts, ignore_index=True)
print("E1 rows:", f"{len(E1):,}")
print("E1 unique strict series:", E1["unique_id"].nunique())

# ===== step_06.py =====
def summarize_e1(pred, min_hist=PRIMARY_MIN_HISTORY):
    x = pred[
        (pred["release_history_len"] >= min_hist)
        & (pred["trim_history_len"] >= min_hist)
    ].copy()
    series_loss = (
        x.groupby(["store_id", "product_id", "model"], as_index=False)
         .agg(
             mae_release=("ae_release", "mean"),
             mae_trim=("ae_trim", "mean"),
             delta_mae=("delta_ae", "mean"),
             n_targets=("target_day", "nunique"),
         )
    )
    rank = (
        series_loss.groupby("model", as_index=False)
        .agg(
            MAE_release=("mae_release", "mean"),
            MAE_trim=("mae_trim", "mean"),
            mean_delta=("delta_mae", "mean"),
            median_delta=("delta_mae", "median"),
            n_series=("product_id", "size"),
        )
    )
    rank["rank_release"] = rank["MAE_release"].rank(method="min")
    rank["rank_trim"] = rank["MAE_trim"].rank(method="min")
    rank["rank_shift"] = rank["rank_trim"] - rank["rank_release"]
    return x, series_loss, rank.sort_values("rank_release")

primary_e1, e1_series_loss, e1_rank = summarize_e1(E1)
display(e1_rank)

e1_target_profile = (
    primary_e1.groupby(["target_day", "model"], as_index=False)
    .agg(
        n_series=("unique_id", "nunique"),
        MAE_release=("ae_release", "mean"),
        MAE_trim=("ae_trim", "mean"),
        delta_MAE=("delta_ae", "mean"),
        median_delta=("delta_ae", "median"),
    )
)
e1_target_profile["pct_MAE_change"] = (
    e1_target_profile["MAE_trim"] / e1_target_profile["MAE_release"] - 1
) * 100

display(e1_target_profile.head(30))

prod_loss = (
    e1_series_loss.groupby(["product_id", "model"], as_index=False)
    .agg(
        mae_release=("mae_release", "mean"),
        mae_trim=("mae_trim", "mean"),
        delta_mae=("delta_mae", "mean"),
    )
)
products = np.array(sorted(prod_loss["product_id"].astype(str).unique()))
models = MODEL_ALIASES
rel_mat = prod_loss.assign(product_id=prod_loss["product_id"].astype(str)).pivot(index="product_id", columns="model", values="mae_release").reindex(products)[models].to_numpy()
trim_mat = prod_loss.assign(product_id=prod_loss["product_id"].astype(str)).pivot(index="product_id", columns="model", values="mae_trim").reindex(products)[models].to_numpy()
delta_mat = prod_loss.assign(product_id=prod_loss["product_id"].astype(str)).pivot(index="product_id", columns="model", values="delta_mae").reindex(products)[models].to_numpy()

rng = np.random.default_rng(RNG_SEED)
effect_reps = np.empty((BOOT_REPS, len(models)))
wr = np.zeros(len(models), int); wt = np.zeros(len(models), int)
changed = 0
tau_reps = np.empty(BOOT_REPS)
for b in range(BOOT_REPS):
    idx = rng.integers(0, len(products), size=len(products))
    r = np.nanmean(rel_mat[idx], axis=0)
    t = np.nanmean(trim_mat[idx], axis=0)
    d = np.nanmean(delta_mat[idx], axis=0)
    effect_reps[b] = d
    ir, it = int(np.nanargmin(r)), int(np.nanargmin(t))
    wr[ir] += 1; wt[it] += 1; changed += int(ir != it)
    tau_reps[b] = stats.kendalltau(stats.rankdata(r), stats.rankdata(t)).statistic

e1_boot_effects = pd.DataFrame([
    {
        "model": m,
        "delta_MAE": float(np.nanmean(delta_mat[:, j])),
        "ci_low": float(np.nanquantile(effect_reps[:, j], .025)),
        "ci_high": float(np.nanquantile(effect_reps[:, j], .975)),
        "n_products": len(products),
    }
    for j, m in enumerate(models)
]).sort_values("delta_MAE")
e1_winner = pd.DataFrame({
    "model": models,
    "release_win_prob": wr/BOOT_REPS,
    "trim_win_prob": wt/BOOT_REPS,
})
e1_winner_change_prob = changed/BOOT_REPS
e1_tau_summary = pd.DataFrame([{
    "mean_tau": float(np.nanmean(tau_reps)),
    "median_tau": float(np.nanmedian(tau_reps)),
    "ci_low": float(np.nanquantile(tau_reps, .025)),
    "ci_high": float(np.nanquantile(tau_reps, .975)),
    "p_tau_below_1": float(np.mean(tau_reps < 1-1e-12)),
    "p_tau_below_0_8": float(np.mean(tau_reps < .8)),
    "winner_change_prob": float(e1_winner_change_prob),
}])

display(e1_boot_effects)
display(e1_winner.sort_values("release_win_prob", ascending=False))
display(e1_tau_summary)

hetero_frames = []
for name, mask in {
    "all_strict": np.ones(len(primary_e1), dtype=bool),
    "gap=1": primary_e1["first_positive_day"].astype(int).eq(1).to_numpy(),
    "gap>=2": primary_e1["first_positive_day"].astype(int).ge(2).to_numpy(),
    "gap>=3": primary_e1["first_positive_day"].astype(int).ge(3).to_numpy(),
    "delay<=7": primary_e1["first_positive_day"].astype(int).le(7).to_numpy(),
    "exclude_day18": primary_e1["first_positive_day"].astype(int).ne(18).to_numpy(),
}.items():
    g = primary_e1.loc[mask]
    s = g.groupby("model", as_index=False).agg(
        n_predictions=("delta_ae", "size"),
        n_series=("unique_id", "nunique"),
        n_products=("product_id", "nunique"),
        MAE_release=("ae_release", "mean"),
        MAE_trim=("ae_trim", "mean"),
        delta_MAE=("delta_ae", "mean"),
    )
    s["subgroup"] = name
    hetero_frames.append(s)
e1_heterogeneity = pd.concat(hetero_frames, ignore_index=True)
display(e1_heterogeneity.head(30))

# ===== step_07.py =====
def coverage_for_sample(sample, min_hist):
    release = 0; trim = 0; lost = 0
    for r in sample.itertuples(index=False):
        f = int(r.first_positive_day)
        rel_n = max(0, 90 - min_hist)
        trim_n = max(0, 90 - (f + min_hist))
        release += rel_n
        trim += trim_n
        lost += rel_n - trim_n
    return release, trim, lost

elig_rows = []
for mh in MIN_HISTORY_SENSITIVITY:
    for label, sample in [("strict_grounded", strict), ("blanket_delayed_zero", blanket)]:
        rel, tr, lost = coverage_for_sample(sample, mh)
        global_rel = len(audit) * max(0, 90 - mh)
        elig_rows.append({
            "scenario": label,
            "min_history": mh,
            "treated_release_eligible_origins": rel,
            "treated_trim_eligible_origins": tr,
            "treated_lost_origins": lost,
            "treated_coverage_loss_share": lost/rel if rel else np.nan,
            "global_release_eligible_origins": global_rel,
            "global_lost_origins": lost,
            "global_coverage_loss_share": lost/global_rel if global_rel else np.nan,
        })
E2 = pd.DataFrame(elig_rows)
display(E2)
