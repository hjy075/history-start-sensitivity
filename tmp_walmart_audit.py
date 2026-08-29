import json, math
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

URL='https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/walmart/train-00000-of-00001.parquet?download=true'
H=39
SEED=20260829
DYN_PLANNED=['IsHoliday','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
DYN_STOCH=['Temperature','Fuel_Price','CPI','Unemployment']
LAGS=[1,2,4,13,26,52]

r=requests.get(URL,timeout=120); r.raise_for_status(); open('/tmp/walmart.parquet','wb').write(r.content)
print('DOWNLOADED',len(r.content),flush=True)
df=pd.read_parquet('/tmp/walmart.parquet')
print('NESTED',df.shape,df.columns.tolist(),flush=True)

# flatten nested FEV format
rows=[]
for _,x in df.iterrows():
    n=len(x['timestamp'])
    z=pd.DataFrame({
        'id':str(x['id']), 'Store':int(x['Store']), 'Dept':int(x['Dept']),
        'Type':str(x['Type']), 'Size':float(x['Size']),
        'timestamp':pd.to_datetime(x['timestamp']), 'target':np.asarray(x['target'],dtype=float)
    })
    for c in DYN_PLANNED+DYN_STOCH:
        z[c]=np.asarray(x[c],dtype=float)
    rows.append(z)
long=pd.concat(rows,ignore_index=True).sort_values(['id','timestamp']).reset_index(drop=True)
print('LONG',long.shape,long.timestamp.min(),long.timestamp.max(),'stores',long.Store.nunique(),'series',long.id.nunique(),flush=True)

# Use the FEV final 39-week horizon. Keep only series with complete horizon and enough history.
all_dates=np.sort(long.timestamp.unique())
test_dates=all_dates[-H:]
cutoff=pd.Timestamp(test_dates[0])
train=long[long.timestamp<cutoff].copy()
test=long[long.timestamp>=cutoff].copy()
counts=test.groupby('id').size()
good=counts[counts==H].index
train=train[train.id.isin(good)].copy(); test=test[test.id.isin(good)].copy()
print('CUTOFF',cutoff,'TRAIN',train.shape,'TEST',test.shape,'GOOD_SERIES',len(good),flush=True)

# MarkDown is competition-provided/planned-like; missing means no recorded markdown. Preserve missingness indicator.
for c in DYN_PLANNED:
    train[c+'_miss']=train[c].isna().astype('int8'); test[c+'_miss']=test[c].isna().astype('int8')
    train[c]=train[c].fillna(0.0); test[c]=test[c].fillna(0.0)

# Stochastic covariate missingness: forward/back fill within store, then global median as last resort.
for c in DYN_STOCH:
    med=float(train[c].median())
    train[c]=train.groupby('Store')[c].ffill().bfill().fillna(med)
    test[c]=test.groupby('Store')[c].ffill().bfill().fillna(med)

# type encoding
all_types=sorted(train.Type.unique())
type_map={v:i for i,v in enumerate(all_types)}
train['TypeCode']=train.Type.map(type_map).astype(int); test['TypeCode']=test.Type.map(type_map).astype(int)

# Build historical target features without leakage.
def add_target_features(frame):
    frame=frame.sort_values(['id','timestamp']).copy()
    g=frame.groupby('id')['target']
    for l in LAGS: frame[f'lag{l}']=g.shift(l)
    frame['roll4']=g.shift(1).rolling(4).mean().reset_index(level=0,drop=True)
    frame['roll13']=g.shift(1).rolling(13).mean().reset_index(level=0,drop=True)
    frame['week_sin']=np.sin(2*np.pi*frame.timestamp.dt.isocalendar().week.astype(float)/52.1775)
    frame['week_cos']=np.cos(2*np.pi*frame.timestamp.dt.isocalendar().week.astype(float)/52.1775)
    return frame
train_feat=add_target_features(train)

planned_feats=DYN_PLANNED+[c+'_miss' for c in DYN_PLANNED]
base_feats=['Store','Dept','TypeCode','Size','week_sin','week_cos']+[f'lag{l}' for l in LAGS]+['roll4','roll13']
full_feats=base_feats+planned_feats+DYN_STOCH
valid_feats=base_feats+planned_feats
train_fit=train_feat.dropna(subset=[f'lag{l}' for l in LAGS]+['roll13']).copy()
print('TRAIN_FIT',train_fit.shape,flush=True)

params=dict(n_estimators=220,learning_rate=.05,num_leaves=63,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,random_state=SEED,n_jobs=-1,verbosity=-1)
full_model=LGBMRegressor(**params).fit(train_fit[full_feats],train_fit.target)
valid_model=LGBMRegressor(**params).fit(train_fit[valid_feats],train_fit.target)
print('MODELS_FIT',flush=True)

# Origin-valid stochastic proxies are store-level, generated solely from history before cutoff.
hist_store=train[['Store','timestamp']+DYN_STOCH].drop_duplicates(['Store','timestamp']).sort_values(['Store','timestamp'])
future_store=test[['Store','timestamp']+DYN_STOCH].drop_duplicates(['Store','timestamp']).sort_values(['Store','timestamp'])

proxy_simple={}; proxy_trend={}
for store,hs in hist_store.groupby('Store'):
    hs=hs.sort_values('timestamp').set_index('timestamp')
    fs=future_store[future_store.Store==store].sort_values('timestamp')
    last=hs.iloc[-1]
    # linear slopes on last 13 observed weeks, evaluated h=1..H
    slopes={}
    for c in ['Fuel_Price','CPI']:
        vals=hs[c].tail(13).to_numpy(float); xx=np.arange(len(vals),dtype=float)
        slopes[c]=float(np.polyfit(xx,vals,1)[0]) if len(vals)>=4 else 0.0
    for h,(_,fr) in enumerate(fs.iterrows(),start=1):
        t=pd.Timestamp(fr.timestamp)
        # 52-week exact seasonal values; use 104-week mean too for structured proxy when available
        vals52={}
        for c in ['Temperature']:
            arr=[]
            for weeks in [52,104]:
                tt=t-pd.Timedelta(weeks=weeks)
                if tt in hs.index: arr.append(float(hs.loc[tt,c]))
            vals52[c]=arr
        temp_simple=vals52['Temperature'][0] if vals52['Temperature'] else float(last['Temperature'])
        temp_trend=float(np.mean(vals52['Temperature'])) if vals52['Temperature'] else temp_simple
        proxy_simple[(store,t)]={
            'Temperature':temp_simple,'Fuel_Price':float(last.Fuel_Price),'CPI':float(last.CPI),'Unemployment':float(last.Unemployment)}
        proxy_trend[(store,t)]={
            'Temperature':temp_trend,
            'Fuel_Price':float(last.Fuel_Price)+slopes['Fuel_Price']*h,
            'CPI':float(last.CPI)+slopes['CPI']*h,
            'Unemployment':float(last.Unemployment)}

# recursive global forecast
history={sid:g.sort_values('timestamp').target.astype(float).tolist() for sid,g in train.groupby('id')}
static=train.groupby('id').tail(1).set_index('id')[['Store','Dept','TypeCode','Size']]
test_idx=test.set_index(['id','timestamp']).sort_index()

def predict_condition(condition):
    hists={k:list(v) for k,v in history.items()}
    outs=[]
    for t in test_dates:
        t=pd.Timestamp(t)
        ids=[sid for sid in good if (sid,t) in test_idx.index]
        rr=[]
        for sid in ids:
            vals=hists[sid]; st=static.loc[sid]; obs=test_idx.loc[(sid,t)]
            row={'Store':st.Store,'Dept':st.Dept,'TypeCode':st.TypeCode,'Size':st.Size,
                 'week_sin':math.sin(2*math.pi*t.isocalendar().week/52.1775),
                 'week_cos':math.cos(2*math.pi*t.isocalendar().week/52.1775)}
            for l in LAGS: row[f'lag{l}']=vals[-l]
            row['roll4']=float(np.mean(vals[-4:])); row['roll13']=float(np.mean(vals[-13:]))
            for c in planned_feats: row[c]=float(obs[c])
            if condition=='oracle':
                for c in DYN_STOCH: row[c]=float(obs[c])
            elif condition=='proxy_simple':
                for c,v in proxy_simple[(int(st.Store),t)].items(): row[c]=v
            elif condition=='proxy_trend':
                for c,v in proxy_trend[(int(st.Store),t)].items(): row[c]=v
            rr.append((sid,row,float(obs.target),float(obs.IsHoliday)))
        X=pd.DataFrame([x[1] for x in rr])
        if condition=='valid_known': pred=valid_model.predict(X[valid_feats])
        else: pred=full_model.predict(X[full_feats])
        for (sid,row,y,hol),yp in zip(rr,pred):
            hists[sid].append(float(yp)); outs.append((sid,t,int(row['Store']),y,float(yp),hol))
    out=pd.DataFrame(outs,columns=['id','timestamp','Store','y','yhat','IsHoliday'])
    out['ae']=(out.y-out.yhat).abs(); out['w']=np.where(out.IsHoliday>0.5,5.0,1.0)
    return out

preds={}
for cond in ['oracle','proxy_simple','proxy_trend','valid_known']:
    preds[cond]=predict_condition(cond)
    d=preds[cond]
    print('RESULT',cond,'MAE',d.ae.mean(),'WMAE',(d.ae*d.w).sum()/d.w.sum(),'N',len(d),flush=True)

# Store-cluster metrics and paired bootstrap across 45 stores.
def store_metric(d):
    return d.groupby('Store').apply(lambda x: pd.Series({'ae_sum':x.ae.sum(),'n':len(x),'wae_sum':(x.ae*x.w).sum(),'w_sum':x.w.sum()}),include_groups=False)
sm={c:store_metric(d) for c,d in preds.items()}
stores=sorted(set.intersection(*[set(x.index) for x in sm.values()]))

def aggregate(tab,ss):
    z=tab.loc[ss]
    return z.ae_sum.sum()/z.n.sum(), z.wae_sum.sum()/z.w_sum.sum()
point={c:aggregate(sm[c],stores) for c in sm}
print('POINT',json.dumps(point),flush=True)

rng=np.random.default_rng(SEED); B=20000
stats={k:[] for k in ['G_mae','P_simple_mae','P_trend_mae','G_wmae','P_simple_wmae','P_trend_wmae']}
for _ in range(B):
    ss=list(rng.choice(stores,size=len(stores),replace=True))
    a={c:aggregate(sm[c],ss) for c in sm}
    stats['G_mae'].append(a['valid_known'][0]-a['oracle'][0])
    stats['P_simple_mae'].append(a['proxy_simple'][0]-a['oracle'][0])
    stats['P_trend_mae'].append(a['proxy_trend'][0]-a['oracle'][0])
    stats['G_wmae'].append(a['valid_known'][1]-a['oracle'][1])
    stats['P_simple_wmae'].append(a['proxy_simple'][1]-a['oracle'][1])
    stats['P_trend_wmae'].append(a['proxy_trend'][1]-a['oracle'][1])
for k,v in stats.items():
    arr=np.asarray(v); print('BOOT',k,'mean',arr.mean(),'CI',np.quantile(arr,[.025,.975]).tolist(),'PrPos',float((arr>0).mean()),flush=True)

for metric,idx in [('MAE',0),('WMAE',1)]:
    G=point['valid_known'][idx]-point['oracle'][idx]
    for p in ['proxy_simple','proxy_trend']:
        P=point[p][idx]-point['oracle'][idx]
        R=(point['valid_known'][idx]-point[p][idx])/G if abs(G)>1e-12 else np.nan
        print('DECOMP',metric,p,'G',G,'P',P,'R',R,flush=True)

# save prediction-level results
pd.concat([d.assign(condition=c) for c,d in preds.items()],ignore_index=True).to_csv('walmart_audit_predictions.csv',index=False)
