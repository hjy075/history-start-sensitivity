import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

URL='https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/uci_air_quality/1D/train-00000-of-00001.parquet?download=true'
RAW=Path('uci_1D.parquet'); OUT=Path('uci_audit_results.csv')
TARGETS=['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']
WEATHER=['T','RH','AH']
H=28; N_WINDOWS=11
BASE=['lag1','lag7','lag14','lag28','roll7','roll28','dow_sin','dow_cos','doy_sin','doy_cos']

def download():
    r=requests.get(URL,timeout=120); r.raise_for_status(); RAW.write_bytes(r.content); print('DOWNLOADED',len(r.content),flush=True)

def flatten(df):
    print('COLUMNS',df.columns.tolist(),flush=True)
    parts=[]
    for _,r in df.iterrows():
        ts=pd.to_datetime(r['timestamp']); n=len(ts)
        for tgt in TARGETS:
            d={'id':np.repeat(tgt,n),'timestamp':ts,'target':np.asarray(r[tgt],float)}
            for c in WEATHER: d[c]=np.asarray(r[c],float)
            parts.append(pd.DataFrame(d))
    return pd.concat(parts,ignore_index=True).sort_values(['id','timestamp']).reset_index(drop=True)

def prep(x):
    x=x.copy(); g=x.groupby('id',sort=False)['target']
    for l in [1,7,14,28]: x[f'lag{l}']=g.shift(l)
    x['roll7']=g.transform(lambda s:s.shift(1).rolling(7,min_periods=7).mean())
    x['roll28']=g.transform(lambda s:s.shift(1).rolling(28,min_periods=28).mean())
    dow=x.timestamp.dt.dayofweek; doy=x.timestamp.dt.dayofyear
    x['dow_sin']=np.sin(2*np.pi*dow/7); x['dow_cos']=np.cos(2*np.pi*dow/7)
    x['doy_sin']=np.sin(2*np.pi*doy/365.25); x['doy_cos']=np.cos(2*np.pi*doy/365.25)
    for c in WEATHER:
        gg=x.groupby('id',sort=False)[c]
        x[f'{c}_p7']=gg.shift(7)
        ls=[]
        for k in [7,14,21,28]:
            z=f'__{c}_{k}'; x[z]=gg.shift(k); ls.append(z)
        x[f'{c}_pclim']=x[ls].mean(axis=1); x.drop(columns=ls,inplace=True)
    return x

def dyn(hist,row):
    a=np.asarray(hist,float)
    return {'lag1':a[-1],'lag7':a[-7],'lag14':a[-14],'lag28':a[-28],
            'roll7':np.mean(a[-7:]),'roll28':np.mean(a[-28:]),
            'dow_sin':row.dow_sin,'dow_cos':row.dow_cos,'doy_sin':row.doy_sin,'doy_cos':row.doy_cos}

def run(train,test,cond):
    if cond=='oracle': wx=WEATHER
    elif cond=='proxy7': wx=[f'{c}_p7' for c in WEATHER]
    elif cond=='proxyclim': wx=[f'{c}_pclim' for c in WEATHER]
    elif cond=='nocov': wx=[]
    feats=BASE+wx
    tr=train.dropna(subset=feats+['target']).copy()
    m=LGBMRegressor(n_estimators=140,learning_rate=.05,num_leaves=31,subsample=.9,colsample_bytree=.9,reg_lambda=1.,random_state=20260829,n_jobs=-1,verbosity=-1)
    m.fit(tr[feats],tr.target)
    ys=[]; ps=[]; ids=[]
    for sid,te in test.groupby('id',sort=False):
        hist=train[train.id==sid].sort_values('timestamp').target.tolist()
        for row in te.sort_values('timestamp').itertuples(index=False):
            f=dyn(hist,row)
            if cond=='oracle':
                for c in WEATHER:f[c]=getattr(row,c)
            elif cond=='proxy7':
                for c in WEATHER:f[f'{c}_p7']=getattr(row,f'{c}_p7')
            elif cond=='proxyclim':
                for c in WEATHER:f[f'{c}_pclim']=getattr(row,f'{c}_pclim')
            p=float(m.predict(pd.DataFrame([[f[k] for k in feats]],columns=feats))[0])
            ys.append(float(row.target));ps.append(p);ids.append(sid);hist.append(p)
    return np.array(ys),np.array(ps),np.array(ids)

def metrics(y,p,ids,train):
    e=np.abs(y-p); mae=float(e.mean()); wape=float(e.sum()/np.abs(y).sum())
    scales={}
    for sid,g in train.groupby('id',sort=False):
        a=g.sort_values('timestamp').target.to_numpy(float); scales[sid]=float(np.mean(np.abs(a[7:]-a[:-7])))
    mase=float(np.mean([err/scales[s] for err,s in zip(e,ids)]))
    return mae,wape,mase

def main():
    download(); nested=pd.read_parquet(RAW); print('NESTED',nested.shape,flush=True)
    x=prep(flatten(nested)); print('LONG',x.shape,x.timestamp.min(),x.timestamp.max(),x.groupby('id').size().to_dict(),flush=True)
    common_end=min(g.timestamp.max() for _,g in x.groupby('id'))
    rows=[]
    for w in range(N_WINDOWS):
        st=common_end-pd.Timedelta(days=H*N_WINDOWS-1)+pd.Timedelta(days=w*H); en=st+pd.Timedelta(days=H-1)
        tr=x[x.timestamp<st].copy(); te=x[(x.timestamp>=st)&(x.timestamp<=en)].copy()
        print('WINDOW',w,st,en,len(tr),len(te),flush=True)
        for cond in ['oracle','proxy7','proxyclim','nocov']:
            y,p,ids=run(tr,te,cond); mae,wape,mase=metrics(y,p,ids,tr)
            row={'window':w,'condition':cond,'test_start':str(st),'mae':mae,'wape':wape,'mase':mase}; rows.append(row); print('RESULT',json.dumps(row),flush=True)
    res=pd.DataFrame(rows);res.to_csv(OUT,index=False);print('SUMMARY\n',res.groupby('condition')[['mae','wape','mase']].mean(),flush=True)
    piv=res.pivot(index='window',columns='condition',values='mae');rng=np.random.default_rng(20260829)
    G=(piv.nocov-piv.oracle).to_numpy(); bs=np.array([rng.choice(G,len(G),replace=True).mean() for _ in range(20000)]);print('G_BOOT',G.mean(),np.quantile(bs,[.025,.975]),'positive',int((G>0).sum()),flush=True)
    for proxy in ['proxy7','proxyclim']:
        P=(piv[proxy]-piv.oracle).to_numpy();bs=np.array([rng.choice(P,len(P),replace=True).mean() for _ in range(20000)]);gm=G.mean();R=(piv.nocov-piv[proxy]).mean()/gm if gm!=0 else np.nan
        print('DECOMP',proxy,'G',gm,'P',P.mean(),'R',R,'Ppos',int((P>0).sum()),'CI',np.quantile(bs,[.025,.975]),flush=True)
if __name__=='__main__':main()
