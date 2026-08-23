# ===== step_01.py =====
# 0. Colab setup
import sys, subprocess, json, os, gc
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "datasets>=3.0", "pyarrow>=15", "pandas>=2.0"],
    check=True,
)

import numpy as np
import pandas as pd
from datasets import load_dataset

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive")
    PROJECT_ROOT = DRIVE_ROOT / "history_start_sensitivity"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "freshretail_data_gate"
else:
    DRIVE_ROOT = Path("/content")
    PROJECT_ROOT = DRIVE_ROOT / "history_start_sensitivity"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "freshretail_data_gate"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
KEYS = ["store_id", "product_id"]

print("OUTPUT_DIR:", OUTPUT_DIR)

# ===== step_02.py =====
KEEP = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]

print("Downloading/loading FreshRetailNet-50K train split...")
ds = load_dataset(DATASET_ID, split="train", keep_in_memory=False)

missing = sorted(set(KEEP) - set(ds.column_names))
if missing:
    raise RuntimeError(f"Missing expected fields: {missing}")

small = ds.select_columns(KEEP)
df = small.to_pandas()
del ds, small
gc.collect()

df["dt"] = pd.to_datetime(df["dt"], errors="raise")
df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce")
df["stock_hour6_22_cnt"] = pd.to_numeric(df["stock_hour6_22_cnt"], errors="coerce")

df = df.sort_values(KEYS + ["dt"]).reset_index(drop=True)
df["day_idx"] = df.groupby(KEYS, sort=False).cumcount()

print("Rows:", f"{len(df):,}")
print("Date range:", df["dt"].min(), "to", df["dt"].max())
print("Memory MB:", round(df.memory_usage(deep=True).sum()/1024**2, 1))
display(df.head())

# ===== step_03.py =====
sizes = df.groupby(KEYS, sort=False).size().rename("n_days")
series_n = len(sizes)

integrity = pd.DataFrame({
    "metric": [
        "rows",
        "series",
        "min_days",
        "median_days",
        "max_days",
        "share_exactly_90_days",
        "missing_sale_rows",
        "negative_sale_rows",
        "missing_stock_status_rows",
    ],
    "value": [
        len(df),
        series_n,
        int(sizes.min()),
        float(sizes.median()),
        int(sizes.max()),
        float((sizes == 90).mean()),
        int(df["sale_amount"].isna().sum()),
        int((df["sale_amount"] < 0).sum()),
        int(df["stock_hour6_22_cnt"].isna().sum()),
    ],
})
display(integrity)

if (sizes == 90).mean() < 0.99:
    print("⚠️ WARNING: fewer than 99% of series have exactly 90 train days.")
else:
    print("✅ Panel length is consistent with the published 90-day encoder history.")

# ===== step_04.py =====
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

neg = (
    df.assign(_neg=df["sale_amount"] < 0)
      .groupby(KEYS, sort=False)["_neg"]
      .any()
      .rename("has_negative")
)
base = base.join(neg, how="left")

tmp = df.merge(
    base[["first_positive_day"]].reset_index(),
    on=KEYS,
    how="left",
)
pre = tmp[
    tmp["first_positive_day"].notna()
    & (tmp["day_idx"] < tmp["first_positive_day"])
].copy()

if len(pre):
    pre_stats = (
        pre.groupby(KEYS, sort=False)
           .agg(
               pre_days=("day_idx", "size"),
               pre_zero_all=("sale_amount", lambda x: bool((x == 0).all())),
               pre_all_instock=("stock_hour6_22_cnt", lambda x: bool((x == 0).all())),
               pre_any_instock=("stock_hour6_22_cnt", lambda x: bool((x == 0).any())),
               pre_stockout_hours_sum=("stock_hour6_22_cnt", "sum"),
           )
    )
    base = base.join(pre_stats, how="left")
else:
    for c in ["pre_days","pre_zero_all","pre_all_instock","pre_any_instock","pre_stockout_hours_sum"]:
        base[c] = np.nan

base["pre_days"] = base["pre_days"].fillna(0).astype(int)
for c in ["pre_zero_all", "pre_all_instock", "pre_any_instock"]:
    base[c] = base[c].fillna(False).astype(bool)

base["strict_treated"] = (
    base["delayed_first_positive"]
    & base["pre_zero_all"]
    & base["pre_all_instock"]
    & (~base["has_negative"])
)
base["any_confirmed_active_zero"] = (
    base["delayed_first_positive"]
    & base["pre_zero_all"]
    & base["pre_any_instock"]
    & (~base["has_negative"])
)

audit = base.reset_index()
strict = audit[audit["strict_treated"]].copy()
any_conf = audit[audit["any_confirmed_active_zero"]].copy()

summary = {
    "series_n": int(len(audit)),
    "never_positive_n": int(audit["never_positive"].sum()),
    "never_positive_share": float(audit["never_positive"].mean()),
    "delayed_first_positive_n": int(audit["delayed_first_positive"].sum()),
    "delayed_first_positive_share": float(audit["delayed_first_positive"].mean()),
    "strict_treated_n": int(len(strict)),
    "strict_treated_share": float(len(strict)/len(audit)),
    "strict_unique_products": int(strict["product_id"].nunique()),
    "strict_unique_stores": int(strict["store_id"].nunique()),
    "any_confirmed_n": int(len(any_conf)),
    "any_confirmed_share": float(len(any_conf)/len(audit)),
    "negative_series_n": int(audit["has_negative"].sum()),
}

print(json.dumps(summary, indent=2))
display(strict.head(20))

# ===== step_05.py =====
gap = (
    audit.loc[audit["first_positive_day"].notna(), "first_positive_day"]
    .astype(int)
    .value_counts()
    .sort_index()
    .rename_axis("first_positive_day")
    .reset_index(name="series_n")
)
gap["share_ever_positive"] = gap["series_n"] / (~audit["never_positive"]).sum()

display(gap.head(30))

if len(strict):
    print("Strict treated first-positive delay:")
    display(strict["first_positive_day"].describe())
    print("Strict treated products:", strict["product_id"].nunique())
    print("Strict treated stores:", strict["store_id"].nunique())

# ===== step_06.py =====
panel_ok = float((sizes == 90).mean()) >= 0.99

strict_n = summary["strict_treated_n"]
strict_share = summary["strict_treated_share"]
strict_products = summary["strict_unique_products"]
any_n = summary["any_confirmed_n"]

if panel_ok and strict_n >= 500 and strict_share >= 0.01 and strict_products >= 100:
    verdict = "PASS"
    reason = "Strong stock-confirmed active-zero exposure with broad product coverage."
elif panel_ok and strict_n >= 250 and strict_share >= 0.005 and strict_products >= 50:
    verdict = "PASS-SMALL"
    reason = "Usable external replication sample, but weaker scale than the strong gate."
elif any_n >= 100:
    verdict = "REVIEW"
    reason = "Mechanism exists, but strict fully-in-stock leading-zero sample is limited."
else:
    verdict = "KILL"
    reason = "Too little stock-confirmed leading-zero exposure for a useful replication."

print("#"*72)
print("FRESHRETAIL EXTERNAL-REPLICATION DATA GATE:", verdict)
print(reason)
print("#"*72)

summary["verdict"] = verdict
summary["verdict_reason"] = reason
summary["share_exactly_90_days"] = float((sizes == 90).mean())

# ===== step_07.py =====
audit.to_parquet(OUTPUT_DIR / "freshretail_series_audit.parquet", index=False)
gap.to_csv(OUTPUT_DIR / "freshretail_first_positive_gap.csv", index=False)
integrity.to_csv(OUTPUT_DIR / "freshretail_panel_integrity.csv", index=False)

with open(OUTPUT_DIR / "freshretail_datagate_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

if len(strict):
    sample_keys = strict[KEYS].head(30)
    sample = df.merge(sample_keys, on=KEYS, how="inner")
    sample = sample[sample["day_idx"] < 15].copy()
    sample.to_csv(OUTPUT_DIR / "freshretail_strict_sample_first15days.csv", index=False)
    display(sample.head(100))

print("Saved to:", OUTPUT_DIR)
