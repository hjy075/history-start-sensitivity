# ===== step_01.py =====
# 0. Setup and persistent paths
import sys, subprocess, os, json, gc, shutil
from pathlib import Path
from datetime import datetime, timezone

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "statsforecast==2.1.1", "datasets>=3.0", "pyarrow>=15", "pandas>=2.0"],
    check=True,
)

import numpy as np
import pandas as pd
from scipy import stats
from datasets import load_dataset

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

DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
KEYS = ["store_id", "product_id"]
KEEP = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]

PROJECT_ROOT = DRIVE_ROOT / "history_start_sensitivity"
CACHE_ROOT = PROJECT_ROOT / "cache" / "freshretail50k"
HF_CACHE_ROOT = CACHE_ROOT / "hf_cache"
CORE_PARQUET = CACHE_ROOT / "train_core_5cols_v1.parquet"
CACHE_MANIFEST = CACHE_ROOT / "cache_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "freshretail"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for p in [CACHE_ROOT, HF_CACHE_ROOT, OUTPUT_DIR, CHECKPOINT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FORCE_DATA_REFRESH = False
RESUME_CHECKPOINTS = True
PRIMARY_MIN_HISTORY = 3
MIN_HISTORY_SENSITIVITY = [2, 3, 4, 7]
MIN_HISTORY_BUILD = min(MIN_HISTORY_SENSITIVITY)
E1_TARGET_DAYS = [4, 7, 14, 21, 28, 42, 56, 70, 84, 89]
GLOBAL_TARGET_DAYS = [28, 56, 89]

BOOT_REPS = 2000
RNG_SEED = 20260823
RUN_GLOBAL_E3 = True
RUN_BLANKET_FIRST_POSITIVE_E3 = True

print("OUTPUT_DIR:", OUTPUT_DIR)
print("CACHE_ROOT:", CACHE_ROOT)
print("StatsForecast version:", statsforecast.__version__)

# ===== step_02.py =====
def validate_core(df):
    missing = sorted(set(KEEP) - set(df.columns))
    if missing:
        raise RuntimeError(f"FreshRetail cache missing expected fields: {missing}")
    if len(df) == 0:
        raise RuntimeError("FreshRetail cache is empty")
    return True

if FORCE_DATA_REFRESH and CORE_PARQUET.exists():
    CORE_PARQUET.unlink()

if CORE_PARQUET.exists() and CORE_PARQUET.stat().st_size > 0:
    print("✅ Persistent FreshRetail Parquet cache hit:", CORE_PARQUET)
    df = pd.read_parquet(CORE_PARQUET, columns=KEEP)
    validate_core(df)
else:
    print("⬇️ ONE-TIME FreshRetail download/load from Hugging Face...")
    ds = load_dataset(
        DATASET_ID,
        split="train",
        keep_in_memory=False,
        cache_dir=str(HF_CACHE_ROOT),
    )
    missing = sorted(set(KEEP) - set(ds.column_names))
    if missing:
        raise RuntimeError(f"Missing expected fields in public dataset: {missing}")
    small = ds.select_columns(KEEP)
    df = small.to_pandas()
    del ds, small
    gc.collect()
    validate_core(df)
    print("Writing persistent 5-column Parquet cache to Drive...")
    df.to_parquet(CORE_PARQUET, index=False, compression="snappy")
    with open(CACHE_MANIFEST, "w") as f:
        json.dump({
            "dataset_id": DATASET_ID,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(df)),
            "columns": KEEP,
            "parquet_bytes": int(CORE_PARQUET.stat().st_size),
        }, f, indent=2)

df["dt"] = pd.to_datetime(df["dt"], errors="raise")
df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce").astype("float32")
df["stock_hour6_22_cnt"] = pd.to_numeric(df["stock_hour6_22_cnt"], errors="coerce")
df = df.sort_values(KEYS + ["dt"]).reset_index(drop=True)
df["day_idx"] = df.groupby(KEYS, sort=False).cumcount().astype("int16")

print("Rows:", f"{len(df):,}")
print("Date range:", df["dt"].min(), "to", df["dt"].max())
print("Cache size MB:", round(CORE_PARQUET.stat().st_size/1024**2, 1))

# ===== step_03.py =====
sizes = df.groupby(KEYS, sort=False).size().rename("n_days")
if not (sizes == 90).all():
    raise RuntimeError("Expected exactly 90 daily rows for every store-product series.")
if df["sale_amount"].isna().any():
    raise RuntimeError("Missing sale_amount values detected.")
if (df["sale_amount"] < 0).any():
    raise RuntimeError("Negative sale_amount values detected.")
if df["stock_hour6_22_cnt"].isna().any():
    raise RuntimeError("Missing stock_hour6_22_cnt values detected.")

positive = df["sale_amount"] > 0
first_pos = (
    df.loc[positive]
      .groupby(KEYS, sort=False)["day_idx"]
      .min()
      .rename("first_positive_day")
)
base = sizes.to_frame().join(first_pos, how="left")
base["never_positive"] = base["first_positive_day"].isna()
base["delayed_first_positive"] = base["first_positive_day"].fillna(0) > 0

tmp = df.merge(base[["first_positive_day"]].reset_index(), on=KEYS, how="left")
pre = tmp[
    tmp["first_positive_day"].notna()
    & (tmp["day_idx"] < tmp["first_positive_day"])
].copy()
pre_stats = (
    pre.groupby(KEYS, sort=False)
       .agg(
           pre_days=("day_idx", "size"),
           pre_zero_all=("sale_amount", lambda x: bool((x == 0).all())),
           pre_all_instock=("stock_hour6_22_cnt", lambda x: bool((x == 0).all())),
           pre_any_instock=("stock_hour6_22_cnt", lambda x: bool((x == 0).any())),
       )
)
base = base.join(pre_stats, how="left")
base["pre_days"] = base["pre_days"].fillna(0).astype(int)
for c in ["pre_zero_all", "pre_all_instock", "pre_any_instock"]:
    base[c] = base[c].fillna(False).astype(bool)

base["strict_treated"] = (
    base["delayed_first_positive"]
    & base["pre_zero_all"]
    & base["pre_all_instock"]
)
base["blanket_delayed_zero"] = (
    base["delayed_first_positive"]
    & base["pre_zero_all"]
)

audit = base.reset_index()
strict = audit[audit["strict_treated"]].copy()
blanket = audit[audit["blanket_delayed_zero"]].copy()

summary_gate = {
    "series_n": int(len(audit)),
    "delayed_first_positive_n": int(audit["delayed_first_positive"].sum()),
    "strict_treated_n": int(len(strict)),
    "strict_treated_share": float(len(strict)/len(audit)),
    "strict_unique_products": int(strict["product_id"].nunique()),
    "strict_unique_stores": int(strict["store_id"].nunique()),
    "blanket_delayed_zero_n": int(len(blanket)),
}
print(json.dumps(summary_gate, indent=2))

series_keys = df[KEYS].drop_duplicates().reset_index(drop=True)
if len(series_keys) != len(audit):
    raise RuntimeError("Series key count mismatch")
Y = df["sale_amount"].to_numpy(dtype=np.float64).reshape(len(series_keys), 90)
key_index = series_keys.copy()
key_index["row_id"] = np.arange(len(series_keys), dtype=int)
audit = audit.merge(key_index, on=KEYS, how="left", validate="one_to_one")
strict = audit[audit["strict_treated"]].copy()
blanket = audit[audit["blanket_delayed_zero"]].copy()

if len(strict) != 903:
    print(f"⚠️ Expected 903 strict series from the prior data gate, found {len(strict)}. Check dataset revision/cache.")
else:
    print("✅ Strict sample matches prior gate: 903 series")

del tmp, pre, pre_stats, df
gc.collect()

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

def sf_batch_forecast(long_df):
    if long_df.empty:
        return pd.DataFrame()
    sf = StatsForecast(models=make_models(), freq="D", n_jobs=-1)
    out = sf.forecast(df=long_df, h=1)
    if "unique_id" not in out.columns:
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)
    return out

def uid_frame(sample):
    x = sample[["row_id", "store_id", "product_id", "first_positive_day"]].copy()
    x["unique_id"] = x["store_id"].astype(str) + "__" + x["product_id"].astype(str)
    return x

def build_long(sample, target_day, trimmed=False):
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for r in sample.itertuples(index=False):
        rid = int(r.row_id)
        f = int(r.first_positive_day)
        start = f if trimmed else 0
        uid = str(r.store_id) + "__" + str(r.product_id)
        for d in range(start, target_day):
            rows.append({
                "unique_id": uid,
                "ds": base_date + pd.Timedelta(days=d),
                "y": float(Y[rid, d]),
            })
    return pd.DataFrame(rows)

def scale_mae(h):
    h = np.asarray(h, float)
    return float(np.mean(np.abs(np.diff(h)))) if len(h) >= 2 else np.nan

def scale_rmse(h):
    h = np.asarray(h, float)
    return float(np.sqrt(np.mean(np.diff(h)**2))) if len(h) >= 2 else np.nan

smoke_y = np.array([0., 1., 0., 0., 2., 0., 1.])
for m in make_models():
    out = m.forecast(y=smoke_y, h=1)
    _ = float(np.asarray(out["mean"]).reshape(-1)[0])
print("✅ Canonical model smoke test passed")
