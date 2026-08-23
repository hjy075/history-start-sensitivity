# ===== step_01.py =====
# 0. Setup
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive")
else:
    DRIVE_ROOT = Path("/content")

PROJECT_ROOT = DRIVE_ROOT / "history_start_sensitivity"
VIS_DIR = PROJECT_ROOT / "outputs" / "visuelle"
FRESH_DIR = PROJECT_ROOT / "outputs" / "freshretail"
PAPER_DIR = PROJECT_ROOT / "paper_outputs"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

for p in [PAPER_DIR, FIG_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

BOOT_REPS = 5000
RNG_SEED = 20260823

print("VIS_DIR:", VIS_DIR)
print("FRESH_DIR:", FRESH_DIR)
print("PAPER_DIR:", PAPER_DIR)

# ===== step_02.py =====
def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file is missing:\n{path}\n"
            "Run the completed Visuelle v4.1 / FreshRetail Main v1 notebook first."
        )
    return path

# Visuelle
vis_series = pd.read_csv(require(VIS_DIR/"E1_series_losses.csv"))
vis_rank_old = pd.read_csv(require(VIS_DIR/"E1_local_rankings.csv"))
vis_boot_old = pd.read_csv(require(VIS_DIR/"E1_product_bootstrap_effects.csv"))
vis_winner_old = pd.read_csv(require(VIS_DIR/"E1_winner_probabilities.csv"))
vis_tau_old = pd.read_csv(require(VIS_DIR/"E1_bootstrap_kendall_tau_summary.csv"))
vis_e2 = pd.read_csv(require(VIS_DIR/"E2_eligibility.csv"))
vis_e3 = pd.read_csv(require(VIS_DIR/"E3_global_rankings.csv"))
vis_scaled = pd.read_csv(require(VIS_DIR/"scaled_metric_sensitivity.csv"))
with open(require(VIS_DIR/"main_experiment_summary.json")) as f:
    vis_summary = json.load(f)

# FreshRetail
fresh_series = pd.read_csv(require(FRESH_DIR/"E1_series_losses.csv"))
fresh_rank_old = pd.read_csv(require(FRESH_DIR/"E1_local_rankings.csv"))
fresh_boot_old = pd.read_csv(require(FRESH_DIR/"E1_product_bootstrap_effects.csv"))
fresh_winner_old = pd.read_csv(require(FRESH_DIR/"E1_winner_probabilities.csv"))
fresh_tau_old = pd.read_csv(require(FRESH_DIR/"E1_bootstrap_kendall_tau_summary.csv"))
fresh_profile = pd.read_csv(require(FRESH_DIR/"E1_target_day_effect_profile.csv"))
fresh_hetero = pd.read_csv(require(FRESH_DIR/"E1_delay_heterogeneity.csv"))
fresh_e2 = pd.read_csv(require(FRESH_DIR/"E2_eligibility.csv"))
fresh_e3 = pd.read_csv(require(FRESH_DIR/"E3_global_rankings.csv"))
fresh_scaled = pd.read_csv(require(FRESH_DIR/"scaled_metric_sensitivity.csv"))
with open(require(FRESH_DIR/"main_experiment_summary.json")) as f:
    fresh_summary = json.load(f)

print("Visuelle series-loss rows:", len(vis_series))
print("FreshRetail series-loss rows:", len(fresh_series))
print("FreshRetail target-profile rows:", len(fresh_profile))

# ===== step_03.py =====
def canonicalize_series_loss(df):
    x = df.copy()
    required = {"product_id", "model", "mae_release", "mae_trim", "delta_mae"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"Missing series-loss columns: {sorted(missing)}")
    x["product_id"] = x["product_id"].astype(str)
    return x

def series_weighted_product_cluster_bootstrap(series_loss, reps=5000, seed=0):
    x = canonicalize_series_loss(series_loss)

    point = (
        x.groupby("model", as_index=False)
         .agg(
             MAE_release=("mae_release", "mean"),
             MAE_trim=("mae_trim", "mean"),
             n_series=("mae_release", "size"),
         )
    )
    point["delta_MAE"] = point["MAE_trim"] - point["MAE_release"]
    point["rank_release"] = point["MAE_release"].rank(method="min")
    point["rank_trim"] = point["MAE_trim"].rank(method="min")
    point["rank_shift"] = point["rank_trim"] - point["rank_release"]

    models = list(point.sort_values("MAE_release")["model"])
    products = np.array(sorted(x["product_id"].unique()))

    gp = (
        x.groupby(["product_id", "model"], as_index=False)
         .agg(
             sum_release=("mae_release", "sum"),
             sum_trim=("mae_trim", "sum"),
             n_series=("mae_release", "size"),
         )
    )

    def make_matrix(value):
        return (
            gp.pivot(index="product_id", columns="model", values=value)
              .reindex(index=products, columns=models)
              .to_numpy(dtype=float)
        )

    SR = np.nan_to_num(make_matrix("sum_release"), nan=0.0)
    ST = np.nan_to_num(make_matrix("sum_trim"), nan=0.0)
    NN = np.nan_to_num(make_matrix("n_series"), nan=0.0)

    rng = np.random.default_rng(seed)
    effect_reps = np.empty((reps, len(models)), dtype=float)
    tau_reps = np.empty(reps, dtype=float)
    wr = np.zeros(len(models), dtype=int)
    wt = np.zeros(len(models), dtype=int)
    changed = 0

    for b in range(reps):
        idx = rng.integers(0, len(products), size=len(products))
        sr = SR[idx].sum(axis=0)
        st = ST[idx].sum(axis=0)
        nn = NN[idx].sum(axis=0)

        mr = np.divide(sr, nn, out=np.full_like(sr, np.nan), where=nn > 0)
        mt = np.divide(st, nn, out=np.full_like(st, np.nan), where=nn > 0)
        effect_reps[b] = mt - mr

        ir = int(np.nanargmin(mr))
        it = int(np.nanargmin(mt))
        wr[ir] += 1
        wt[it] += 1
        changed += int(ir != it)

        tau_reps[b] = stats.kendalltau(
            stats.rankdata(mr),
            stats.rankdata(mt),
        ).statistic

    effects = point.set_index("model").reindex(models).reset_index()
    effects = effects[["model", "delta_MAE", "n_series"]]
    effects["ci_low"] = np.nanquantile(effect_reps, .025, axis=0)
    effects["ci_high"] = np.nanquantile(effect_reps, .975, axis=0)
    effects["n_products"] = len(products)
    effects["ci_excludes_zero"] = (
        (effects["ci_low"] > 0) | (effects["ci_high"] < 0)
    )

    winner = pd.DataFrame({
        "model": models,
        "release_win_prob": wr / reps,
        "trim_win_prob": wt / reps,
    })

    tau_summary = pd.DataFrame([{
        "mean_tau": float(np.nanmean(tau_reps)),
        "median_tau": float(np.nanmedian(tau_reps)),
        "ci_low": float(np.nanquantile(tau_reps, .025)),
        "ci_high": float(np.nanquantile(tau_reps, .975)),
        "p_tau_below_1": float(np.mean(tau_reps < 1 - 1e-12)),
        "p_tau_below_0_8": float(np.mean(tau_reps < .8)),
        "winner_change_prob": float(changed / reps),
        "bootstrap_reps": int(reps),
        "cluster_unit": "product_id",
        "estimand": "series_weighted",
    }])

    point = point.sort_values("rank_release").reset_index(drop=True)
    return point, effects, winner, tau_summary

vis_point, vis_effects, vis_winner, vis_tau = series_weighted_product_cluster_bootstrap(
    vis_series, reps=BOOT_REPS, seed=RNG_SEED
)
fresh_point, fresh_effects, fresh_winner, fresh_tau = series_weighted_product_cluster_bootstrap(
    fresh_series, reps=BOOT_REPS, seed=RNG_SEED + 1
)

print("VISUELLE — corrected")
display(vis_point)
display(vis_effects.sort_values("delta_MAE"))
display(vis_winner.sort_values("release_win_prob", ascending=False))
display(vis_tau)

print("FRESHRETAIL — corrected")
display(fresh_point)
display(fresh_effects.sort_values("delta_MAE"))
display(fresh_winner.sort_values("release_win_prob", ascending=False))
display(fresh_tau)

# ===== step_04.py =====
def normalize_old_effect(df):
    x = df.copy()
    if "delta_MAE_trim_minus_release" in x.columns:
        x = x.rename(columns={"delta_MAE_trim_minus_release": "old_delta_MAE"})
    elif "delta_MAE" in x.columns:
        x = x.rename(columns={"delta_MAE": "old_delta_MAE"})
    else:
        raise RuntimeError("Old effect file has no recognized delta column.")
    if "ci_low" in x.columns:
        x = x.rename(columns={"ci_low": "old_ci_low"})
    if "ci_high" in x.columns:
        x = x.rename(columns={"ci_high": "old_ci_high"})
    return x

def effect_comparison(dataset, old_df, new_df):
    old = normalize_old_effect(old_df)
    keep = [c for c in ["model", "old_delta_MAE", "old_ci_low", "old_ci_high"] if c in old.columns]
    new = new_df.rename(columns={
        "delta_MAE": "new_delta_MAE",
        "ci_low": "new_ci_low",
        "ci_high": "new_ci_high",
    })
    z = new.merge(old[keep], on="model", how="left")
    z["dataset"] = dataset
    z["delta_estimand_difference"] = z["new_delta_MAE"] - z["old_delta_MAE"]
    return z

cmp_vis = effect_comparison("Visuelle", vis_boot_old, vis_effects)
cmp_fresh = effect_comparison("FreshRetail", fresh_boot_old, fresh_effects)
estimand_comparison = pd.concat([cmp_vis, cmp_fresh], ignore_index=True)

display(estimand_comparison.sort_values(["dataset", "model"]))

# ===== step_05.py =====
fresh_profile = fresh_profile.sort_values(["model", "target_day"]).reset_index(drop=True)
display(fresh_profile)

audit_rows = []
for model, g in fresh_profile.groupby("model"):
    g = g.sort_values("target_day")
    vals = g["delta_MAE"].to_numpy(dtype=float)
    nz = np.sign(vals[np.abs(vals) > 1e-12])
    audit_rows.append({
        "model": model,
        "n_target_days": len(g),
        "min_delta_MAE": float(np.nanmin(vals)),
        "max_delta_MAE": float(np.nanmax(vals)),
        "first_delta_MAE": float(vals[0]),
        "last_delta_MAE": float(vals[-1]),
        "sign_changes": int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0,
        "non_monotonic": bool(
            not (
                np.all(np.diff(vals) >= -1e-12)
                or np.all(np.diff(vals) <= 1e-12)
            )
        ),
    })
fresh_time_audit = pd.DataFrame(audit_rows)
display(fresh_time_audit)
