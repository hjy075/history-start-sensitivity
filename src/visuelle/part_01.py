# ===== step_01.py =====
# 0. Setup
import sys, subprocess, os, json, gc, shutil, zipfile, hashlib
from pathlib import Path
from datetime import datetime, timezone

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "statsforecast==2.1.1", "pyarrow>=15"],
    check=True,
)

import numpy as np
import pandas as pd
from scipy import stats

import statsforecast
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    HistoricAverage,
    SimpleExponentialSmoothingOptimized,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    TSB,
    ADIDA,
    IMAPA,
)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive")
else:
    DRIVE_ROOT = Path("/content")

PROJECT_ROOT = DRIVE_ROOT / "history_start_sensitivity"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visuelle"
CACHE_ROOT = PROJECT_ROOT / "cache" / "visuelle2"
WORK_ROOT = Path("/content/visuelle2_work")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Obtain Visuelle 2.0 through the official project page:
# https://humaticslab.github.io/forecasting/visuelle
# Then place either the downloaded archive or sales.csv at this path (or edit it).
VISUELLE_SOURCE = DRIVE_ROOT / "history_start_sensitivity" / "data" / "visuelle2.zip"
SALES_CACHE_PARQUET = CACHE_ROOT / "sales_core_v1.parquet"
SALES_CACHE_CSV = CACHE_ROOT / "sales.csv"
CACHE_MANIFEST = CACHE_ROOT / "cache_manifest.json"

LOCAL_SALES_CSV = WORK_ROOT / "sales.csv"

FORCE_DATA_REFRESH = False
VERIFY_CACHE_SHA256 = False

PRIMARY_MIN_HISTORY = 3
MIN_HISTORY_SENSITIVITY = [2, 3, 4]
MIN_HISTORY_BUILD = min(MIN_HISTORY_SENSITIVITY)
GLOBAL_TARGET_WEEKS = [8, 9, 10, 11]
BOOT_REPS = 2000
RNG_SEED = 20260823
RUN_TSB_ALPHA_SENSITIVITY = True
TSB_ALPHAS = [0.1, 0.2, 0.3]

print("OUTPUT_DIR:", OUTPUT_DIR)
print("CACHE_ROOT:", CACHE_ROOT)
print("StatsForecast version:", statsforecast.__version__)

# ===== step_02.py =====
def human_bytes(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

def validate_sales_schema(df):
    cols = {str(c) for c in df.columns}
    required = {str(i) for i in range(12)}
    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(f"Not a valid Visuelle sales table; missing weekly columns: {missing}")
    return True

def load_valid_sales_csv(path):
    path = Path(path)
    head = pd.read_csv(path, nrows=3)
    validate_sales_schema(head)
    df = pd.read_csv(path)
    validate_sales_schema(df)
    return df

def write_parquet_atomic(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False, compression="snappy")
    check = pd.read_parquet(tmp)
    validate_sales_schema(check)
    del check
    tmp.replace(path)

def extract_sales_from_zip(zip_path, out_csv):
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [m for m in zf.namelist() if Path(m).name.lower() == "sales.csv"]
        if not candidates:
            raise FileNotFoundError(f"sales.csv not found inside {zip_path}")
        member = sorted(candidates, key=len)[0]
        with zf.open(member) as src, open(out_csv, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_csv

def ensure_visuelle_sales():
    if FORCE_DATA_REFRESH:
        SALES_CACHE_PARQUET.unlink(missing_ok=True)

    if SALES_CACHE_PARQUET.exists():
        df = pd.read_parquet(SALES_CACHE_PARQUET)
        validate_sales_schema(df)
        print("Persistent Visuelle cache hit:", SALES_CACHE_PARQUET)
        return df

    source = Path(VISUELLE_SOURCE)
    if not source.exists():
        raise FileNotFoundError(
            "Visuelle 2.0 is not bundled with this repository. Obtain it from the official project page "
            "(https://humaticslab.github.io/forecasting/visuelle), then set VISUELLE_SOURCE to the downloaded "
            "archive, sales.csv, or a directory containing sales.csv."
        )

    if source.is_dir():
        candidates = list(source.rglob("sales.csv"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one sales.csv under {source}; found {len(candidates)}")
        csv_path = candidates[0]
    elif source.suffix.lower() == ".csv":
        csv_path = source
    elif source.suffix.lower() == ".zip":
        csv_path = WORK_ROOT / "sales.csv"
        extract_sales_from_zip(source, csv_path)
    else:
        raise ValueError("VISUELLE_SOURCE must point to a .zip, sales.csv, or directory containing sales.csv")

    df = load_valid_sales_csv(csv_path)
    write_parquet_atomic(df, SALES_CACHE_PARQUET)
    print("Created persistent Visuelle cache:", SALES_CACHE_PARQUET)
    return df

sales = ensure_visuelle_sales()
print("sales:", sales.shape)
print("cache parquet:", SALES_CACHE_PARQUET, human_bytes(SALES_CACHE_PARQUET.stat().st_size))

# ===== step_03.py =====
WEEKS = [str(i) for i in range(12)]
missing = [c for c in WEEKS if c not in sales.columns]
if missing:
    raise RuntimeError(f"Missing weekly sales columns: {missing}")

for c in WEEKS:
    sales[c] = pd.to_numeric(sales[c], errors="coerce")

Y = sales[WEEKS].to_numpy(float)
has_negative = np.any(Y < 0, axis=1)
has_missing = np.any(np.isnan(Y), axis=1)
first_pos = np.full(len(sales), np.nan)

for i, y in enumerate(Y):
    pos = np.flatnonzero(y > 0)
    if len(pos):
        first_pos[i] = int(pos[0])

pre_zero = np.zeros(len(sales), dtype=bool)
for i, f in enumerate(first_pos):
    if np.isfinite(f) and int(f) > 0:
        pre_zero[i] = bool(np.all(Y[i, :int(f)] == 0))

meta_cols = [c for c in ["external_code","retail","season","category","release_date"] if c in sales.columns]
audit = sales[meta_cols].copy()
audit["row_id"] = np.arange(len(sales))
audit["first_sale_week"] = first_pos
audit["has_negative"] = has_negative
audit["has_missing"] = has_missing
audit["strict_treated"] = (
    np.isfinite(first_pos) & (first_pos > 0) & pre_zero & (~has_negative) & (~has_missing)
)
audit["nonnegative_population"] = (~has_negative) & (~has_missing)

strict = audit[audit["strict_treated"]].copy()
global_pop = audit[audit["nonnegative_population"]].copy()

print("All series:", f"{len(audit):,}")
print("Negative-series excluded:", f"{has_negative.sum():,}")
print("Missing-series excluded:", f"{has_missing.sum():,}")
print("Global primary population:", f"{len(global_pop):,}")
print("Strict treated:", f"{len(strict):,}", f"({len(strict)/len(audit):.3%})")
print("Unique strict products:", strict["external_code"].nunique())
print("Unique strict stores:", strict["retail"].nunique())

# ===== step_04.py =====
MODEL_ALIASES = [
    "Naive", "HistAvg", "SESOpt", "CrostonClassic", "CrostonOpt",
    "CrostonSBA", "TSB", "ADIDA", "IMAPA"
]

def make_models():
    return [
        Naive(alias="Naive"),
        HistoricAverage(alias="HistAvg"),
        SimpleExponentialSmoothingOptimized(alias="SESOpt"),
        CrostonClassic(alias="CrostonClassic"),
        CrostonOptimized(alias="CrostonOpt"),
        CrostonSBA(alias="CrostonSBA"),
        TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB"),
        ADIDA(alias="ADIDA"),
        IMAPA(alias="IMAPA"),
    ]

def one_step_model(history, model):
    y = np.asarray(history, dtype=np.float64)
    out = model.forecast(y=y, h=1)
    pred = float(np.asarray(out["mean"]).reshape(-1)[0])
    return max(0.0, pred)

smoke_y = np.array([0., 1., 0., 0., 2., 0., 1.])
smoke = {}
for m in make_models():
    try:
        smoke[m.alias] = one_step_model(smoke_y, m)
    except Exception as e:
        smoke[m.alias] = f"ERROR: {e}"
print(smoke)

errors = [k for k,v in smoke.items() if isinstance(v, str)]
if errors:
    raise RuntimeError(f"StatsForecast smoke test failed for: {errors}")

# ===== step_05.py =====
def scale_mae(h):
    h = np.asarray(h, float)
    if len(h) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(h))))

def scale_rmse(h):
    h = np.asarray(h, float)
    if len(h) < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(h) ** 2)))

local_rows = []
failure_rows = []

for _, a in strict.iterrows():
    rid = int(a["row_id"])
    y = Y[rid]
    f = int(a["first_sale_week"])
    product = str(a["external_code"])
    store = str(a["retail"])

    for t in range(f + MIN_HISTORY_BUILD, 12):
        h_rel = y[:t]
        h_trim = y[f:t]
        actual = float(y[t])

        smae_rel = scale_mae(h_rel)
        smae_trim = scale_mae(h_trim)
        srmse_rel = scale_rmse(h_rel)
        srmse_trim = scale_rmse(h_trim)

        for model in make_models():
            try:
                pr = one_step_model(h_rel, model)
                model2 = next(m for m in make_models() if m.alias == model.alias)
                pt = one_step_model(h_trim, model2)
            except Exception as e:
                failure_rows.append({
                    "product_id": product, "store_id": store,
                    "target_week": t, "first_sale_week": f,
                    "release_history_len": len(h_rel),
                    "trim_history_len": len(h_trim),
                    "model": model.alias, "error": repr(e),
                })
                continue

            ae_r = abs(actual - pr)
            ae_t = abs(actual - pt)
            se_r = (actual - pr) ** 2
            se_t = (actual - pt) ** 2

            local_rows.append({
                "product_id": product,
                "store_id": store,
                "target_week": t,
                "first_sale_week": f,
                "release_history_len": len(h_rel),
                "trim_history_len": len(h_trim),
                "model": model.alias,
                "y": actual,
                "pred_release": pr,
                "pred_trim": pt,
                "ae_release": ae_r,
                "ae_trim": ae_t,
                "delta_ae": ae_t - ae_r,
                "se_release": se_r,
                "se_trim": se_t,
                "fixed_mase_release": ae_r/smae_rel if np.isfinite(smae_rel) and smae_rel > 0 else np.nan,
                "fixed_mase_trim": ae_t/smae_rel if np.isfinite(smae_rel) and smae_rel > 0 else np.nan,
                "conv_mase_release": ae_r/smae_rel if np.isfinite(smae_rel) and smae_rel > 0 else np.nan,
                "conv_mase_trim": ae_t/smae_trim if np.isfinite(smae_trim) and smae_trim > 0 else np.nan,
                "fixed_rmsse_release": np.sqrt(se_r)/srmse_rel if np.isfinite(srmse_rel) and srmse_rel > 0 else np.nan,
                "fixed_rmsse_trim": np.sqrt(se_t)/srmse_rel if np.isfinite(srmse_rel) and srmse_rel > 0 else np.nan,
                "conv_rmsse_release": np.sqrt(se_r)/srmse_rel if np.isfinite(srmse_rel) and srmse_rel > 0 else np.nan,
                "conv_rmsse_trim": np.sqrt(se_t)/srmse_trim if np.isfinite(srmse_trim) and srmse_trim > 0 else np.nan,
            })

local_pred = pd.DataFrame(local_rows)
failures = pd.DataFrame(failure_rows)

print("Unique local prediction rows:", f"{len(local_pred):,}")
print("Failures:", f"{len(failures):,}")
if len(failures):
    display(failures.groupby("model").size().sort_values(ascending=False))
