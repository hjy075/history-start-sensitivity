# ===== step_06.py =====
def summarize_local(pred, min_hist=PRIMARY_MIN_HISTORY):
    x = pred[
        (pred["release_history_len"] >= min_hist)
        & (pred["trim_history_len"] >= min_hist)
    ].copy()

    series_loss = (
        x.groupby(["product_id", "store_id", "model"], as_index=False)
         .agg(
             mae_release=("ae_release", "mean"),
             mae_trim=("ae_trim", "mean"),
             delta_mae=("delta_ae", "mean"),
             n_origins=("target_week", "size"),
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
    rank["rank_shift_trim_minus_release"] = rank["rank_trim"] - rank["rank_release"]
    rank["abs_rank_shift"] = rank["rank_shift_trim_minus_release"].abs()

    wape_rows = []
    for model, g in x.groupby("model"):
        denom = g["y"].sum()
        wape_rows.append({
            "model": model,
            "WAPE_release": g["ae_release"].sum()/denom if denom else np.nan,
            "WAPE_trim": g["ae_trim"].sum()/denom if denom else np.nan,
        })
    rank = rank.merge(pd.DataFrame(wape_rows), on="model", how="left")
    return x, series_loss, rank.sort_values("rank_release")

primary_pred, series_loss, local_rank = summarize_local(local_pred)
rank_shifts = local_rank[[
    "model", "rank_release", "rank_trim", "rank_shift_trim_minus_release", "abs_rank_shift"
]].sort_values("abs_rank_shift", ascending=False)

display(local_rank)
display(rank_shifts)

# ===== step_07.py =====
rng = np.random.default_rng(RNG_SEED)

prod_loss = (
    series_loss.groupby(["product_id", "model"], as_index=False)
    .agg(
        mae_release=("mae_release", "mean"),
        mae_trim=("mae_trim", "mean"),
        delta_mae=("delta_mae", "mean"),
    )
)

products = np.array(sorted(prod_loss["product_id"].unique()))
models = MODEL_ALIASES

rel_mat = prod_loss.pivot(index="product_id", columns="model", values="mae_release").reindex(products)[models].to_numpy()
trim_mat = prod_loss.pivot(index="product_id", columns="model", values="mae_trim").reindex(products)[models].to_numpy()
delta_mat = prod_loss.pivot(index="product_id", columns="model", values="delta_mae").reindex(products)[models].to_numpy()

effect_reps = np.empty((BOOT_REPS, len(models)))
winner_rel = np.zeros(len(models), dtype=int)
winner_trim = np.zeros(len(models), dtype=int)
winner_changed = 0
tau_reps = np.empty(BOOT_REPS, dtype=float)

for b in range(BOOT_REPS):
    idx = rng.integers(0, len(products), size=len(products))
    r = np.nanmean(rel_mat[idx], axis=0)
    t = np.nanmean(trim_mat[idx], axis=0)
    d = np.nanmean(delta_mat[idx], axis=0)
    effect_reps[b] = d
    wr, wt = int(np.nanargmin(r)), int(np.nanargmin(t))
    winner_rel[wr] += 1
    winner_trim[wt] += 1
    winner_changed += int(wr != wt)
    tau_reps[b] = stats.kendalltau(stats.rankdata(r), stats.rankdata(t)).statistic

effects = []
for j, m in enumerate(models):
    vals = effect_reps[:, j]
    effects.append({
        "model": m,
        "delta_MAE_trim_minus_release": float(np.nanmean(delta_mat[:, j])),
        "ci_low": float(np.nanquantile(vals, 0.025)),
        "ci_high": float(np.nanquantile(vals, 0.975)),
        "n_products": int(len(products)),
    })
effects = pd.DataFrame(effects).sort_values("delta_MAE_trim_minus_release")

winner = pd.DataFrame({
    "model": models,
    "release_win_prob": winner_rel / BOOT_REPS,
    "trim_win_prob": winner_trim / BOOT_REPS,
})
winner_change_prob = winner_changed / BOOT_REPS

bootstrap_tau = pd.DataFrame({"kendall_tau": tau_reps})
bootstrap_tau_summary = pd.DataFrame([{
    "mean_tau": float(np.nanmean(tau_reps)),
    "median_tau": float(np.nanmedian(tau_reps)),
    "ci_low": float(np.nanquantile(tau_reps, 0.025)),
    "ci_high": float(np.nanquantile(tau_reps, 0.975)),
    "p_tau_below_1": float(np.mean(tau_reps < 1 - 1e-12)),
    "p_tau_below_0_8": float(np.mean(tau_reps < 0.8)),
}])

display(effects)
display(winner.sort_values("release_win_prob", ascending=False))
display(bootstrap_tau_summary)
print("Bootstrap probability winner changes:", f"{winner_change_prob:.1%}")

# ===== step_08.py =====
elig_rows = []

for min_hist in MIN_HISTORY_SENSITIVITY:
    local_release = 0
    local_trim = 0
    local_lost = 0

    for _, a in strict.iterrows():
        f = int(a["first_sale_week"])
        for t in range(1, 12):
            rel_ok = t >= min_hist
            trim_ok = (t - f) >= min_hist
            local_release += int(rel_ok)
            local_trim += int(rel_ok and trim_ok)
            local_lost += int(rel_ok and not trim_ok)

    n_global = len(global_pop)
    release_origins_global = n_global * sum(t >= min_hist for t in range(1,12))
    lost_global = local_lost

    elig_rows.append({
        "min_history": min_hist,
        "treated_release_eligible_origins": local_release,
        "treated_trim_eligible_origins": local_trim,
        "treated_lost_origins": local_lost,
        "treated_coverage_loss_share": local_lost/local_release if local_release else np.nan,
        "global_release_eligible_origins": release_origins_global,
        "global_lost_origins": lost_global,
        "global_coverage_loss_share": lost_global/release_origins_global if release_origins_global else np.nan,
    })

eligibility = pd.DataFrame(elig_rows)
display(eligibility)

# ===== step_09.py =====
def build_release_long(target_week):
    pop = global_pop[["row_id","external_code","retail"]].copy()
    parts = []
    base_date = pd.Timestamp("2000-01-02")
    for w in range(target_week):
        p = pop.copy()
        p["unique_id"] = p["external_code"].astype(str) + "__" + p["retail"].astype(str)
        p["ds"] = base_date + pd.to_timedelta(w*7, unit="D")
        p["y"] = Y[p["row_id"].to_numpy(), w]
        parts.append(p[["unique_id","ds","y"]])
    return pd.concat(parts, ignore_index=True)

def build_trim_long(target_week):
    rows = []
    base_date = pd.Timestamp("2000-01-02")
    for _, a in strict.iterrows():
        rid = int(a["row_id"])
        f = int(a["first_sale_week"])
        if target_week - f < PRIMARY_MIN_HISTORY:
            continue
        uid = str(a["external_code"]) + "__" + str(a["retail"])
        for w in range(f, target_week):
            rows.append({
                "unique_id": uid,
                "ds": base_date + pd.Timedelta(days=w*7),
                "y": float(Y[rid,w]),
            })
    return pd.DataFrame(rows)

def sf_batch_forecast(long_df):
    sf = StatsForecast(
        models=make_models(),
        freq="W",
        n_jobs=-1,
    )
    out = sf.forecast(df=long_df, h=1)
    if "unique_id" not in out.columns:
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)
    return out

global_detail = []
global_summaries = []

for target_week in GLOBAL_TARGET_WEEKS:
    print("\nGLOBAL TARGET WEEK", target_week)
    rel_long = build_release_long(target_week)
    rel_fc = sf_batch_forecast(rel_long)

    actual = global_pop[["row_id","external_code","retail"]].copy()
    actual["unique_id"] = actual["external_code"].astype(str) + "__" + actual["retail"].astype(str)
    actual["y_true"] = Y[actual["row_id"].to_numpy(), target_week]
    actual = actual[["unique_id","external_code","retail","y_true"]]

    rel = actual.merge(rel_fc.drop(columns=["ds"], errors="ignore"), on="unique_id", how="inner")

    trim_long = build_trim_long(target_week)
    trim_fc = sf_batch_forecast(trim_long) if len(trim_long) else pd.DataFrame()

    for model in MODEL_ALIASES:
        if model not in rel.columns:
            raise RuntimeError(f"Missing StatsForecast output column: {model}")

        rel[f"ae_{model}_release"] = (rel["y_true"] - rel[model]).abs()
        rel[f"pred_{model}_trim"] = rel[model]

        if len(trim_fc) and model in trim_fc.columns:
            map_trim = trim_fc.set_index("unique_id")[model]
            mask = rel["unique_id"].isin(map_trim.index)
            rel.loc[mask, f"pred_{model}_trim"] = rel.loc[mask, "unique_id"].map(map_trim)

        rel[f"ae_{model}_trim"] = (rel["y_true"] - rel[f"pred_{model}_trim"]).abs()

        denom = rel["y_true"].sum()
        mae_r = rel[f"ae_{model}_release"].mean()
        mae_t = rel[f"ae_{model}_trim"].mean()
        wape_r = rel[f"ae_{model}_release"].sum()/denom if denom else np.nan
        wape_t = rel[f"ae_{model}_trim"].sum()/denom if denom else np.nan

        global_summaries.append({
            "target_week": target_week,
            "model": model,
            "MAE_release": mae_r,
            "MAE_trim": mae_t,
            "delta_MAE": mae_t-mae_r,
            "WAPE_release": wape_r,
            "WAPE_trim": wape_t,
            "delta_WAPE": wape_t-wape_r,
            "n_series": len(rel),
        })

    keep_cols = ["unique_id","external_code","retail","y_true"] + [
        c for c in rel.columns if c.startswith("ae_")
    ]
    d = rel[keep_cols].copy()
    d["target_week"] = target_week
    global_detail.append(d)

    del rel_long, rel_fc, trim_long, trim_fc, rel
    gc.collect()

global_summary = pd.DataFrame(global_summaries)
global_errors = pd.concat(global_detail, ignore_index=True)

global_rank = (
    global_summary.groupby("model", as_index=False)
    .agg(
        MAE_release=("MAE_release","mean"),
        MAE_trim=("MAE_trim","mean"),
        delta_MAE=("delta_MAE","mean"),
        WAPE_release=("WAPE_release","mean"),
        WAPE_trim=("WAPE_trim","mean"),
        delta_WAPE=("delta_WAPE","mean"),
    )
)
global_rank["rank_release"] = global_rank["MAE_release"].rank(method="min")
global_rank["rank_trim"] = global_rank["MAE_trim"].rank(method="min")
global_rank = global_rank.sort_values("rank_release")

display(global_rank)

# ===== step_10.py =====
g11 = global_errors[global_errors["target_week"] == 11].copy()

rel_cols = {m: f"ae_{m}_release" for m in MODEL_ALIASES}
trim_cols = {m: f"ae_{m}_trim" for m in MODEL_ALIASES}

prod_rel = g11.groupby(g11["external_code"].astype(str))[
    list(rel_cols.values())
].mean()
prod_trim = g11.groupby(g11["external_code"].astype(str))[
    list(trim_cols.values())
].mean()

common_products = prod_rel.index.intersection(prod_trim.index)
R = prod_rel.loc[common_products, list(rel_cols.values())].to_numpy()
T = prod_trim.loc[common_products, list(trim_cols.values())].to_numpy()

rng = np.random.default_rng(RNG_SEED + 1)
wr = np.zeros(len(MODEL_ALIASES), int)
wt = np.zeros(len(MODEL_ALIASES), int)
changed = 0

for b in range(BOOT_REPS):
    idx = rng.integers(0, len(common_products), size=len(common_products))
    mr = np.nanmean(R[idx], axis=0)
    mt = np.nanmean(T[idx], axis=0)
    ir, it = int(np.nanargmin(mr)), int(np.nanargmin(mt))
    wr[ir] += 1
    wt[it] += 1
    changed += int(ir != it)

global_winner = pd.DataFrame({
    "model": MODEL_ALIASES,
    "release_win_prob": wr/BOOT_REPS,
    "trim_win_prob": wt/BOOT_REPS,
})
global_winner_change_prob = changed/BOOT_REPS

display(global_winner.sort_values("release_win_prob", ascending=False))
print("Global benchmark winner-change probability:", f"{global_winner_change_prob:.1%}")
