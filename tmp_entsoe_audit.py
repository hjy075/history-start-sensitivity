import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

URL = "https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/entsoe/1H/train-00000-of-00001.parquet?download=true"
OUT = Path("entsoe_audit_results.csv")
RAW = Path("entsoe_1H.parquet")
H = 168
N_WINDOWS = 20
WEATHER = ["temperature", "radiation_direct_horizontal", "radiation_diffuse_horizontal"]
BASE_FEATURES = ["lag1", "lag24", "lag168", "lag336", "roll24", "roll168", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


def download():
    if RAW.exists():
        return
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    RAW.write_bytes(r.content)
    print("DOWNLOADED", len(r.content), flush=True)


def flatten_nested(df):
    parts=[]
    for _, r in df.iterrows():
        ts = pd.to_datetime(r["timestamp"])
        n=len(ts)
        d={"id":np.repeat(str(r["id"]), n), "timestamp":ts}
        for c in ["target","solar_generation_actual","wind_onshore_generation_actual"]+WEATHER:
            d[c]=np.asarray(r[c], dtype=float)
        parts.append(pd.DataFrame(d))
    x=pd.concat(parts, ignore_index=True)
    x=x.sort_values(["id","timestamp"]).reset_index(drop=True)
    return x


def add_static_history_features(x):
    x=x.copy()
    g=x.groupby("id", sort=False)["target"]
    for lag in [1,24,168,336]:
        x[f"lag{lag}"]=g.shift(lag)
    x["roll24"]=g.transform(lambda s: s.shift(1).rolling(24, min_periods=24).mean())
    x["roll168"]=g.transform(lambda s: s.shift(1).rolling(168, min_periods=168).mean())
    hr=x.timestamp.dt.hour
    dow=x.timestamp.dt.dayofweek
    doy=x.timestamp.dt.dayofyear
    x["hour_sin"]=np.sin(2*np.pi*hr/24); x["hour_cos"]=np.cos(2*np.pi*hr/24)
    x["dow_sin"]=np.sin(2*np.pi*dow/7); x["dow_cos"]=np.cos(2*np.pi*dow/7)
    x["doy_sin"]=np.sin(2*np.pi*doy/365.25); x["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    for c in WEATHER:
        gg=x.groupby("id", sort=False)[c]
        x[f"{c}_p7"]=gg.shift(168)
        lagcols=[]
        for k in [168,336,504,672]:
            name=f"__{c}_{k}"
            x[name]=gg.shift(k)
            lagcols.append(name)
        x[f"{c}_pclim"]=x[lagcols].mean(axis=1)
        x.drop(columns=lagcols, inplace=True)
    return x


def dynamic_row_features(hist_y, row):
    # hist_y includes actual history and recursive predictions only up to current step
    arr=np.asarray(hist_y, dtype=float)
    f={
        "lag1":arr[-1], "lag24":arr[-24], "lag168":arr[-168], "lag336":arr[-336],
        "roll24":np.nanmean(arr[-24:]), "roll168":np.nanmean(arr[-168:]),
        "hour_sin":row.hour_sin, "hour_cos":row.hour_cos, "dow_sin":row.dow_sin, "dow_cos":row.dow_cos,
        "doy_sin":row.doy_sin, "doy_cos":row.doy_cos,
    }
    return f


def fit_predict(train, test, condition):
    if condition=="oracle":
        wx=WEATHER
    elif condition=="proxy7":
        wx=[f"{c}_p7" for c in WEATHER]
    elif condition=="proxyclim":
        wx=[f"{c}_pclim" for c in WEATHER]
    elif condition=="nocov":
        wx=[]
    else: raise ValueError(condition)
    feats=BASE_FEATURES+wx
    tr=train.dropna(subset=feats+["target"])
    model=LGBMRegressor(n_estimators=140, learning_rate=0.05, num_leaves=31, max_depth=-1,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=20260829,
                        n_jobs=-1, verbosity=-1)
    model.fit(tr[feats], tr.target)
    preds=[]; ys=[]; ids=[]
    for sid, te in test.groupby("id", sort=False):
        te=te.sort_values("timestamp")
        hist=train.loc[train.id==sid].sort_values("timestamp").target.to_list()
        for row in te.itertuples(index=False):
            f=dynamic_row_features(hist, row)
            if condition=="oracle":
                for c in WEATHER: f[c]=getattr(row,c)
            elif condition=="proxy7":
                for c in WEATHER: f[f"{c}_p7"]=getattr(row,f"{c}_p7")
            elif condition=="proxyclim":
                for c in WEATHER: f[f"{c}_pclim"]=getattr(row,f"{c}_pclim")
            X=pd.DataFrame([[f[k] for k in feats]], columns=feats)
            p=float(model.predict(X)[0])
            preds.append(p); ys.append(float(row.target)); ids.append(sid); hist.append(p)
    return np.asarray(ys), np.asarray(preds), np.asarray(ids)


def metrics(y,p,ids,train):
    e=np.abs(y-p)
    mae=float(np.mean(e))
    wape=float(np.sum(e)/np.sum(np.abs(y)))
    scales={}
    for sid,g in train.groupby("id", sort=False):
        a=g.sort_values("timestamp").target.to_numpy(float)
        scales[sid]=float(np.mean(np.abs(a[24:]-a[:-24])))
    mase=float(np.mean([err/scales[sid] for err,sid in zip(e,ids)]))
    return mae,wape,mase


def main():
    download()
    nested=pd.read_parquet(RAW)
    print("NESTED_SHAPE", nested.shape, "IDS", nested.id.tolist(), flush=True)
    x=add_static_history_features(flatten_nested(nested))
    print("LONG_SHAPE", x.shape, "RANGE", x.timestamp.min(), x.timestamp.max(), flush=True)
    counts=x.groupby('id').size().to_dict(); print("COUNTS", counts, flush=True)
    # Common final timestamp ensures paired windows across all six series.
    common_end=min(g.timestamp.max() for _,g in x.groupby('id'))
    rows=[]
    conditions=["oracle","proxy7","proxyclim","nocov"]
    for w in range(N_WINDOWS):
        # w=0 is earliest of final 20 non-overlapping 168h windows
        test_start=common_end-pd.Timedelta(hours=H*N_WINDOWS-1)+pd.Timedelta(hours=w*H)
        test_end=test_start+pd.Timedelta(hours=H-1)
        train=x[x.timestamp<test_start].copy()
        test=x[(x.timestamp>=test_start)&(x.timestamp<=test_end)].copy()
        print("WINDOW",w,test_start,test_end,"train",len(train),"test",len(test), flush=True)
        if test.groupby('id').size().min()!=H:
            raise RuntimeError(f"Incomplete test window {w}: {test.groupby('id').size().to_dict()}")
        for cond in conditions:
            y,p,ids=fit_predict(train,test,cond)
            mae,wape,mase=metrics(y,p,ids,train)
            row={"window":w,"condition":cond,"test_start":str(test_start),"mae":mae,"wape":wape,"mase":mase}
            rows.append(row)
            print("RESULT",json.dumps(row), flush=True)
    res=pd.DataFrame(rows); res.to_csv(OUT,index=False)
    summary=res.groupby('condition')[["mae","wape","mase"]].mean().sort_values('mae')
    print("SUMMARY\n"+summary.to_string(), flush=True)
    piv=res.pivot(index='window',columns='condition',values='mae')
    for proxy in ["proxy7","proxyclim"]:
        G=float((piv.nocov-piv.oracle).mean())
        P=float((piv[proxy]-piv.oracle).mean())
        R=float(((piv.nocov-piv[proxy]).mean())/G) if G!=0 else np.nan
        wins=int((piv[proxy]>piv.oracle).sum())
        print(f"DECOMP {proxy} G={G:.6f} P={P:.6f} R={R:.6f} P_positive_windows={wins}/{N_WINDOWS}", flush=True)
    # paired bootstrap of mean P over windows
    rng=np.random.default_rng(20260829)
    for proxy in ["proxy7","proxyclim"]:
        d=(piv[proxy]-piv.oracle).to_numpy()
        bs=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)])
        lo,hi=np.quantile(bs,[.025,.975])
        print(f"BOOT {proxy} meanP={d.mean():.6f} CI95=[{lo:.6f},{hi:.6f}]", flush=True)
    print("RESULT_FILE",OUT.resolve(),flush=True)

if __name__=="__main__": main()
