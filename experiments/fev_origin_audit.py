import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260829
H = 168
N_WINDOWS = 20
INNER_FOLDS = 4
OUT = Path("tmp_fev_audit_results")
OUT.mkdir(exist_ok=True)


def load_fev_gfc14():
    obj = load_dataset("autogluon/fev_datasets", "proenfo_gfc14")
    split_name = next(iter(obj.keys()))
    ds = obj[split_name]
    if len(ds) != 1:
        raise RuntimeError(f"Expected one series for gfc14, got {len(ds)}")
    row = ds[0]
    print("HF_SPLIT", split_name)
    print("HF_COLUMNS", ds.column_names)
    print("HF_FEATURES", ds.features)

    def pick(names):
        for n in names:
            if n in row:
                return n
        return None

    ts_col = pick(["timestamp", "timestamps", "date", "datetime"])
    target_col = pick(["target", "load", "Load", "y"])
    temp_col = "airtemperature"
    if ts_col is None or target_col is None or temp_col not in row:
        list_cols = [k for k, v in row.items() if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 1000]
        print("LIST_COLUMNS", list_cols)
        if ts_col is None:
            for k in list_cols:
                if isinstance(row[k][0], str):
                    ts_col = k
                    break
        if target_col is None:
            numeric_lists = [k for k in list_cols if k not in {ts_col, temp_col}]
            if len(numeric_lists) != 1:
                raise RuntimeError(f"Cannot infer target column: {numeric_lists}")
            target_col = numeric_lists[0]

    ts = pd.to_datetime(pd.Series(row[ts_col]))
    y = np.asarray(row[target_col], dtype=float)
    temp = np.asarray(row[temp_col], dtype=float)
    if not (len(ts) == len(y) == len(temp)):
        raise RuntimeError("Length mismatch")
    frame = pd.DataFrame({"timestamp": ts, "y": y, "temp": temp}).sort_values("timestamp").reset_index(drop=True)
    print("N_ROWS", len(frame), "START", frame.timestamp.iloc[0], "END", frame.timestamp.iloc[-1])
    return frame


def calendar_features(ts):
    ts = pd.DatetimeIndex(ts)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()
    raw = ts.view("int64")
    year_frac = (raw - raw.min()) / max(1, raw.max() - raw.min())
    return np.column_stack([
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
        year_frac,
    ])


def temp_pred_simple(method, temp, cutoff):
    a = temp[cutoff-H:cutoff]
    b = temp[cutoff-2*H:cutoff-H]
    if method == "snaive168":
        return a.copy()
    if method == "avg2":
        return 0.5 * (a + b)
    if method == "trend2":
        return np.clip(2*a - b, -50, 130)
    raise KeyError(method)


def temp_model_features(ts, temp, idx):
    idx = np.asarray(idx)
    cal = calendar_features(ts.iloc[idx])
    lag168 = temp[idx-H]
    lag336 = temp[idx-2*H]
    return np.column_stack([cal, lag168, lag336, 0.5*(lag168+lag336), lag168-lag336])


def temp_pred_ridge(ts, temp, cutoff):
    start = max(2*H, cutoff - 24*365*2)
    train_idx = np.arange(start, cutoff)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(temp_model_features(ts, temp, train_idx), temp[train_idx])
    fut_idx = np.arange(cutoff, cutoff+H)
    return model.predict(temp_model_features(ts, temp, fut_idx))


def make_temp_pred(method, ts, temp, cutoff):
    if method in {"snaive168", "avg2", "trend2"}:
        return temp_pred_simple(method, temp, cutoff)
    if method == "ridge_temp":
        return temp_pred_ridge(ts, temp, cutoff)
    raise KeyError(method)


def select_temp_method(ts, temp, cutoff):
    methods = ["snaive168", "avg2", "trend2", "ridge_temp"]
    scores = {m: [] for m in methods}
    for j in range(INNER_FOLDS, 0, -1):
        c = cutoff - j*H
        if c < 2*H + 500:
            continue
        truth = temp[c:c+H]
        for m in methods:
            scores[m].append(mean_absolute_error(truth, make_temp_pred(m, ts, temp, c)))
    means = {m: float(np.mean(v)) if v else float("inf") for m, v in scores.items()}
    return min(means, key=means.get), means


def load_features(df, idx, future_temp=None, include_temp=True):
    idx = np.asarray(idx)
    cal = calendar_features(df.timestamp.iloc[idx])
    y = df.y.to_numpy()
    lag168 = y[idx-H]
    lag336 = y[idx-2*H]
    X = np.column_stack([cal, lag168, lag336, 0.5*(lag168+lag336), lag168-lag336])
    if include_temp:
        t = df.temp.to_numpy()[idx] if future_temp is None else np.asarray(future_temp)
        X = np.column_stack([X, t, t*t/100.0, np.maximum(65.0-t, 0.0), np.maximum(t-65.0, 0.0)])
    return X


def model_factories():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=31, learning_rate=0.06, l2_regularization=1.0, random_state=SEED),
        "rf": lambda: RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=SEED),
        "extra": lambda: ExtraTreesRegressor(n_estimators=80, max_depth=18, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=SEED),
    }


def wape(y, p):
    return float(np.sum(np.abs(y-p)) / max(np.sum(np.abs(y)), 1e-12))


def bootstrap_ci(x, B=3000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    sims = np.mean(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))]


def run():
    df = load_fev_gfc14()
    n = len(df)
    first_cutoff = n - H*N_WINDOWS
    print("FIRST_CUTOFF", first_cutoff, df.timestamp.iloc[first_cutoff], "LAST_CUTOFF", n-H, df.timestamp.iloc[n-H])
    rows, temp_rows = [], []

    for w in range(N_WINDOWS):
        cutoff = first_cutoff + w*H
        test_idx = np.arange(cutoff, cutoff+H)
        y_true = df.y.to_numpy()[test_idx]
        t_actual = df.temp.to_numpy()[test_idx]
        selected, cv_scores = select_temp_method(df.timestamp, df.temp.to_numpy(), cutoff)
        t_origin = make_temp_pred(selected, df.timestamp, df.temp.to_numpy(), cutoff)
        t_sn = make_temp_pred("snaive168", df.timestamp, df.temp.to_numpy(), cutoff)
        temp_mae = mean_absolute_error(t_actual, t_origin)
        temp_rows.append({
            "window": w, "cutoff": str(df.timestamp.iloc[cutoff]), "selected": selected,
            "origin_temp_mae": temp_mae, "snaive_temp_mae": mean_absolute_error(t_actual, t_sn),
            **{f"cv_{k}": v for k, v in cv_scores.items()},
            "actual_temp_mean": float(np.mean(t_actual)), "actual_temp_min": float(np.min(t_actual)), "actual_temp_max": float(np.max(t_actual)),
        })
        print(f"WINDOW {w:02d} cutoff={df.timestamp.iloc[cutoff]} selected={selected} tempMAE={temp_mae:.3f}")

        train_idx = np.arange(2*H, cutoff)
        Xtr_temp = load_features(df, train_idx, include_temp=True)
        Xtr_none = load_features(df, train_idx, include_temp=False)
        X_oracle = load_features(df, test_idx, future_temp=t_actual, include_temp=True)
        X_origin = load_features(df, test_idx, future_temp=t_origin, include_temp=True)
        X_sn = load_features(df, test_idx, future_temp=t_sn, include_temp=True)
        X_none = load_features(df, test_idx, include_temp=False)

        for name, factory in model_factories().items():
            m = factory(); m.fit(Xtr_temp, df.y.to_numpy()[train_idx])
            preds = {"oracle": m.predict(X_oracle), "origin": m.predict(X_origin), "snaive_temp": m.predict(X_sn)}
            m0 = factory(); m0.fit(Xtr_none, df.y.to_numpy()[train_idx])
            preds["none"] = m0.predict(X_none)
            for cond, pred in preds.items():
                rows.append({"window": w, "cutoff": str(df.timestamp.iloc[cutoff]), "model": name, "condition": cond,
                             "mae": mean_absolute_error(y_true, pred), "wape": wape(y_true, pred)})

    res = pd.DataFrame(rows)
    tdf = pd.DataFrame(temp_rows)
    res.to_csv(OUT/"window_model_metrics.csv", index=False)
    tdf.to_csv(OUT/"temperature_selection.csv", index=False)
    wide = res.pivot_table(index=["window","model"], columns="condition", values="mae").reset_index()
    wide["optimism"] = (wide["origin"] - wide["oracle"]) / wide["origin"]
    wide["oracle_lift"] = (wide["none"] - wide["oracle"]) / wide["none"]
    wide["origin_lift"] = (wide["none"] - wide["origin"]) / wide["none"]
    wide["lift_shrinkage"] = wide["oracle_lift"] - wide["origin_lift"]
    wide.to_csv(OUT/"paired_effects.csv", index=False)

    model_summary = []
    for model, g in wide.groupby("model"):
        delta = g["origin"] - g["oracle"]
        model_summary.append({
            "model": model, "oracle_mae_mean": float(g.oracle.mean()), "origin_mae_mean": float(g.origin.mean()), "none_mae_mean": float(g.none.mean()),
            "mean_optimism": float(g.optimism.mean()), "median_optimism": float(g.optimism.median()),
            "frac_optimism_ge_20pct": float((g.optimism >= .20).mean()), "mean_oracle_lift": float(g.oracle_lift.mean()),
            "mean_origin_lift": float(g.origin_lift.mean()), "mean_lift_shrinkage": float(g.lift_shrinkage.mean()),
            "mean_abs_mae_gap": float(delta.mean()), "gap_ci95": bootstrap_ci(delta),
        })
    ms = pd.DataFrame(model_summary)
    ms.to_csv(OUT/"model_summary.csv", index=False)

    rank_rows = []
    models = sorted(res.model.unique())
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]
    for w in range(N_WINDOWS):
        z = res[res.window.eq(w)].pivot(index="model", columns="condition", values="mae")
        ro, rr = z.oracle.rank(), z.origin.rank()
        reversals = 0
        for a, b in pairs:
            so = np.sign(z.loc[a,"oracle"] - z.loc[b,"oracle"])
            sr = np.sign(z.loc[a,"origin"] - z.loc[b,"origin"])
            reversals += int(so != 0 and sr != 0 and so != sr)
        rank_rows.append({"window": w, "spearman": float(spearmanr(ro, rr).statistic), "kendall": float(kendalltau(ro, rr).statistic),
                          "top1_oracle": z.oracle.idxmin(), "top1_origin": z.origin.idxmin(),
                          "top1_switch": int(z.oracle.idxmin() != z.origin.idxmin()), "pairwise_reversal_rate": reversals/len(pairs)})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(OUT/"rank_distortion.csv", index=False)

    gap_by_window = wide.groupby("window").apply(lambda x: np.mean(x.origin-x.oracle), include_groups=False)
    diag = tdf.set_index("window").copy(); diag["mean_load_gap"] = gap_by_window
    diag.to_csv(OUT/"window_diagnostics.csv")
    corr_temp = spearmanr(diag.origin_temp_mae, diag.mean_load_gap).statistic

    summary = {
        "dataset": "autogluon/fev_datasets::proenfo_gfc14", "horizon": H, "num_windows": N_WINDOWS,
        "model_summary": model_summary,
        "ranking": {"mean_spearman": float(ranks.spearman.mean()), "median_spearman": float(ranks.spearman.median()),
                    "mean_kendall": float(ranks.kendall.mean()), "top1_switch_rate": float(ranks.top1_switch.mean()),
                    "mean_pairwise_reversal_rate": float(ranks.pairwise_reversal_rate.mean()), "windows_spearman_lt_0_9": int((ranks.spearman < .9).sum())},
        "temperature": {"mean_origin_temp_mae": float(tdf.origin_temp_mae.mean()), "median_origin_temp_mae": float(tdf.origin_temp_mae.median()),
                        "selected_counts": {str(k): int(v) for k,v in tdf.selected.value_counts().items()}, "rho_temp_mae_vs_load_gap": float(corr_temp)},
        "gates": {"any_model_mean_optimism_ge_20pct": bool((ms.mean_optimism >= .20).any()),
                  "any_model_mean_lift_shrinkage_ge_30pct": bool((ms.mean_lift_shrinkage >= .30).any()),
                  "any_rank_reversal": bool((ranks.pairwise_reversal_rate > 0).any()), "top1_switch_rate": float(ranks.top1_switch.mean())}
    }
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== MODEL SUMMARY ==="); print(ms.to_string(index=False))
    print("=== RANK SUMMARY ==="); print(ranks.to_string(index=False))
    print("=== FINAL SUMMARY JSON ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
