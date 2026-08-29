import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

URL="https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/proenfo_gfc14/train-00000-of-00001.parquet?download=true"
RAW=Path('gfc14.parquet'); OUT=Path('gfc14_audit_results.csv')
H=168; N_WINDOWS=20; W='airtemperature'
BASE=['lag1','lag24','lag168','lag336','roll24','roll168','hour_sin','hour_cos','dow_sin','dow_cos','doy_sin','doy_cos']

def download():
    r=requests.get(URL,timeout=120); r.raise_for_status(); RAW.write_bytes(r.content); print('DOWNLOADED',len(r.content),flush=True)

def flatten(df):
    ps=[]
    for _,r in df.iterrows():
        ts=pd.to_datetime(r['timestamp']); n=len(ts)
        ps.append(pd.DataFrame({'id':np.repeat(str(r['id']),n),'timestamp':ts,'target':np.asarray(r['target'],float),W:np.asarray(r[W],float)}))
    return pd.concat(ps,ignore_index=True).sort_values(['id','timestamp']).reset_index(drop=True)

def prep(x):
    x=x.copy(); g=x.groupby('id',sort=False)['target']
    for l in [1,24,168,336]: x[f'lag{l}']=g.shift(l)
    x['roll24']=g.transform(lambda s:s.shift(1).rolling(24,min_periods=24).mean())
    x['roll168']=g.transform(lambda s:s.shift(1).rolling(168,min_periods=168).mean())
    hr=x.timestamp.dt.hour; dow=x.timestamp.dt.dayofweek; doy=x.timestamp.dt.dayofyear
    x['hour_sin']=np.sin(2*np.pi*hr/24); x['hour_cos']=np.cos(2*np.pi*hr/24)
    x['dow_sin']=np.sin(2*np.pi*dow/7); x['dow_cos']=np.cos(2*np.pi*dow/7)
    x['doy_sin']=np.sin(2*np.pi*doy/365.25); x['doy_cos']=np.cos(2*np.pi*doy/365.25)
    gg=x.groupby('id',sort=False)[W]
    x[W+'_p7']=gg.shift(168)
    ls=[]
    for k in [168,336,504,672]:
        c=f'__w{k}'; x[c]=gg.shift(k); ls.append(c)
    x[W+'_pclim']=x[ls].mean(axis=1); x.drop(columns=ls,inplace=True)
    return x

def dyn(hist,row):
    a=np.asarray(hist,float)
    return {'lag1':a[-1],'lag24':a[-24],'lag168':a[-168],'lag336':a[-336],
            'roll24':np.mean(a[-24:]),'roll168':np.mean(a[-168:]),
            'hour_sin':row.hour_sin,'hour_cos':row.hour_cos,'dow_sin':row.dow_sin,'dow_cos':row.dow_cos,
            'doy_sin':row.doy_sin,'doy_cos':row.doy_cos}

def run(train,test,cond):
    extra=[] if cond=='nocov' else [W if cond=='oracle' else W+'_'+('p7' if cond=='proxy7' else 'pclim')]
    feats=BASE+extra
    tr=train.dropna(subset=feats+['target'])
    m=LGBMRegressor(n_estimators=140,learning_rate=.05,num_leaves=31,subsample=.9,colsample_bytree=.9,reg_lambda=1.,random_state=20260829,n_jobs=-1,verbosity=-1)
    m.fit(tr[feats],tr.target)
    ys=[]; ps=[]; ids=[]
    for sid,te in test.groupby('id',sort=False):
        hist=train[train.id==sid].sort_values('timestamp').target.tolist()
        for row in te.sort_values('timestamp').itertuples(index=False):
            f=dyn(hist,row)
            if cond=='oracle': f[W]=getattr(row,W)
            elif cond=='proxy7': f[W+'_p7']=getattr(row,W+'_p7')
            elif cond=='proxyclim': f[W+'_pclim']=getattr(row,W+'_pclim')
            X=pd.DataFrame([[f[k] for k in feats]],columns=feats); p=float(m.predict(X)[0])
            ys.append(float(row.target)); ps.append(p); ids.append(sid); hist.append(p)
    return np.array(ys),np.array(ps),np.array(ids)

def metrics(y,p,ids,train):
    e=np.abs(y-p); mae=e.mean(); wape=e.sum()/np.abs(y).sum(); scales={}
    for sid,g in train.groupby('id'):
        a=g.sort_values('timestamp').target.to_numpy(float); scales[sid]=np.mean(np.abs(a[24:]-a[:-24]))
    mase=np.mean([ee/scales[i] for ee,i in zip(e,ids)])
    return float(mae),float(wape),float(mase)

def main():
    download(); nested=pd.read_parquet(RAW); print('COLUMNS',nested.columns.tolist(),flush=True); print('NESTED',nested.shape,'IDS',nested.id.tolist(),flush=True)
    x=prep(flatten(nested)); print('LONG',x.shape,x.timestamp.min(),x.timestamp.max(),x.groupby('id').size().to_dict(),flush=True)
    common_end=min(g.timestamp.max() for _,g in x.groupby('id'))
    rows=[]
    for w in range(N_WINDOWS):
        st=common_end-pd.Timedelta(hours=H*N_WINDOWS-1)+pd.Timedelta(hours=w*H); en=st+pd.Timedelta(hours=H-1)
        tr=x[x.timestamp<st].copy(); te=x[(x.timestamp>=st)&(x.timestamp<=en)].copy(); print('WINDOW',w,st,en,len(tr),len(te),flush=True)
        for c in ['oracle','proxy7','proxyclim','nocov']:
            y,p,ids=run(tr,te,c); mae,wape,mase=metrics(y,p,ids,tr); row={'window':w,'condition':c,'test_start':str(st),'mae':mae,'wape':wape,'mase':mase}; rows.append(row); print('RESULT',json.dumps(row),flush=True)
    r=pd.DataFrame(rows); r.to_csv(OUT,index=False); print('SUMMARY\n'+r.groupby('condition')[['mae','wape','mase']].mean().sort_values('mae').to_string(),flush=True)
    piv=r.pivot(index='window',columns='condition',values='mae'); rng=np.random.default_rng(20260829)
    G=(piv.nocov-piv.oracle).to_numpy(); bs=np.array([rng.choice(G,len(G),replace=True).mean() for _ in range(20000)]); print('G_BOOT',G.mean(),np.quantile(bs,[.025,.975]),'positive',int((G>0).sum()),flush=True)
    for px in ['proxy7','proxyclim']:
        P=(piv[px]-piv.oracle).to_numpy(); bs=np.array([rng.choice(P,len(P),replace=True).mean() for _ in range(20000)]); g=G.mean(); R=(piv.nocov-piv[px]).mean()/g if g!=0 else np.nan; print('DECOMP',px,'G',g,'P',P.mean(),'R',R,'Ppos',int((P>0).sum()),'CI',np.quantile(bs,[.025,.975]),flush=True)
if __name__=='__main__': main()
