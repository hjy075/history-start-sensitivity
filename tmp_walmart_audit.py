import json, math
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

URL='https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/walmart/train-00000-of-00001.parquet?download=true'
H=39; SEED=20260829
DYN_PLANNED=['IsHoliday','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
DYN_STOCH=['Temperature','Fuel_Price','CPI','Unemployment']
LAGS=[1,2,4,13,26,52]
r=requests.get(URL,timeout=120); r.raise_for_status(); open('/tmp/walmart.parquet','wb').write(r.content)
print('DOWNLOADED',len(r.content),flush=True)
df=pd.read_parquet('/tmp/walmart.parquet'); print('NESTED',df.shape,df.columns.tolist(),flush=True)
rows=[]
for _,x in df.iterrows():
 n=len(x['timestamp']); z=pd.DataFrame({'id':str(x.id),'Store':int(x.Store),'Dept':int(x.Dept),'Type':str(x.Type),'Size':float(x.Size),'timestamp':pd.to_datetime(x.timestamp),'target':np.asarray(x.target,float)})
 for c in DYN_PLANNED+DYN_STOCH: z[c]=np.asarray(x[c],float)
 rows.append(z)
long=pd.concat(rows,ignore_index=True).sort_values(['id','timestamp']).reset_index(drop=True)
print('LONG',long.shape,long.timestamp.min(),long.timestamp.max(),'stores',long.Store.nunique(),'series',long.id.nunique(),flush=True)
all_dates=np.sort(long.timestamp.unique()); test_dates=all_dates[-H:]; cutoff=pd.Timestamp(test_dates[0])
train=long[long.timestamp<cutoff].copy(); test=long[long.timestamp>=cutoff].copy()
counts=test.groupby('id').size(); candidates=counts[counts==H].index
# Require enough pre-origin target history for the largest lag. This fixes short/new department series without changing the estimand.
hist_counts=train[train.id.isin(candidates)].groupby('id').size(); good=hist_counts[hist_counts>=max(LAGS)].index
train=train[train.id.isin(good)].copy(); test=test[test.id.isin(good)].copy()
print('CUTOFF',cutoff,'TRAIN',train.shape,'TEST',test.shape,'GOOD_SERIES',len(good),'DROPPED_SHORT',len(candidates)-len(good),flush=True)
for c in DYN_PLANNED:
 train[c+'_miss']=train[c].isna().astype('int8'); test[c+'_miss']=test[c].isna().astype('int8'); train[c]=train[c].fillna(0.); test[c]=test[c].fillna(0.)
for c in DYN_STOCH:
 med=float(train[c].median()); train[c]=train.groupby('Store')[c].ffill().bfill().fillna(med); test[c]=test.groupby('Store')[c].ffill().bfill().fillna(med)
type_map={v:i for i,v in enumerate(sorted(train.Type.unique()))}; train['TypeCode']=train.Type.map(type_map).astype(int); test['TypeCode']=test.Type.map(type_map).astype(int)
def addf(f):
 f=f.sort_values(['id','timestamp']).copy(); g=f.groupby('id')['target']
 for l in LAGS: f[f'lag{l}']=g.shift(l)
 # transform keeps rolling windows strictly within each series
 f['roll4']=g.transform(lambda s:s.shift(1).rolling(4).mean()); f['roll13']=g.transform(lambda s:s.shift(1).rolling(13).mean())
 wk=f.timestamp.dt.isocalendar().week.astype(float); f['week_sin']=np.sin(2*np.pi*wk/52.1775); f['week_cos']=np.cos(2*np.pi*wk/52.1775); return f
train_feat=addf(train)
planned=DYN_PLANNED+[c+'_miss' for c in DYN_PLANNED]
base=['Store','Dept','TypeCode','Size','week_sin','week_cos']+[f'lag{l}' for l in LAGS]+['roll4','roll13']; full=base+planned+DYN_STOCH; valid=base+planned
fit=train_feat.dropna(subset=[f'lag{l}' for l in LAGS]+['roll13']).copy(); print('TRAIN_FIT',fit.shape,flush=True)
params=dict(n_estimators=220,learning_rate=.05,num_leaves=63,subsample=.9,colsample_bytree=.9,reg_lambda=1.,random_state=SEED,n_jobs=-1,verbosity=-1)
fm=LGBMRegressor(**params).fit(fit[full],fit.target); vm=LGBMRegressor(**params).fit(fit[valid],fit.target); print('MODELS_FIT',flush=True)
hs0=train[['Store','timestamp']+DYN_STOCH].drop_duplicates(['Store','timestamp']).sort_values(['Store','timestamp']); fs0=test[['Store','timestamp']+DYN_STOCH].drop_duplicates(['Store','timestamp']).sort_values(['Store','timestamp'])
ps={}; pt={}
for store,hs in hs0.groupby('Store'):
 hs=hs.sort_values('timestamp').set_index('timestamp'); fs=fs0[fs0.Store==store].sort_values('timestamp'); last=hs.iloc[-1]; slopes={}
 for c in ['Fuel_Price','CPI']:
  v=hs[c].tail(13).to_numpy(float); slopes[c]=float(np.polyfit(np.arange(len(v),dtype=float),v,1)[0]) if len(v)>=4 else 0.
 for h,(_,fr) in enumerate(fs.iterrows(),1):
  t=pd.Timestamp(fr.timestamp); arr=[]
  for w in [52,104]:
   tt=t-pd.Timedelta(weeks=w)
   if tt in hs.index: arr.append(float(hs.loc[tt,'Temperature']))
  ts=arr[0] if arr else float(last.Temperature); tt=float(np.mean(arr)) if arr else ts
  ps[(store,t)]={'Temperature':ts,'Fuel_Price':float(last.Fuel_Price),'CPI':float(last.CPI),'Unemployment':float(last.Unemployment)}
  pt[(store,t)]={'Temperature':tt,'Fuel_Price':float(last.Fuel_Price)+slopes['Fuel_Price']*h,'CPI':float(last.CPI)+slopes['CPI']*h,'Unemployment':float(last.Unemployment)}
history={sid:g.sort_values('timestamp').target.astype(float).tolist() for sid,g in train.groupby('id')}; static=train.groupby('id').tail(1).set_index('id')[['Store','Dept','TypeCode','Size']]; ti=test.set_index(['id','timestamp']).sort_index(); good_list=list(good)
def predict(cond):
 hists={k:list(v) for k,v in history.items()}; outs=[]
 for tv in test_dates:
  t=pd.Timestamp(tv); rr=[]
  for sid in good_list:
   if (sid,t) not in ti.index: continue
   vals=hists[sid]; st=static.loc[sid]; obs=ti.loc[(sid,t)]; row={'Store':st.Store,'Dept':st.Dept,'TypeCode':st.TypeCode,'Size':st.Size,'week_sin':math.sin(2*math.pi*t.isocalendar().week/52.1775),'week_cos':math.cos(2*math.pi*t.isocalendar().week/52.1775)}
   for l in LAGS: row[f'lag{l}']=vals[-l]
   row['roll4']=float(np.mean(vals[-4:])); row['roll13']=float(np.mean(vals[-13:]))
   for c in planned: row[c]=float(obs[c])
   if cond=='oracle':
    for c in DYN_STOCH: row[c]=float(obs[c])
   elif cond=='proxy_simple': row.update(ps[(int(st.Store),t)])
   elif cond=='proxy_trend': row.update(pt[(int(st.Store),t)])
   rr.append((sid,row,float(obs.target),float(obs.IsHoliday)))
  X=pd.DataFrame([x[1] for x in rr]); pred=vm.predict(X[valid]) if cond=='valid_known' else fm.predict(X[full])
  for (sid,row,y,hol),yp in zip(rr,pred): hists[sid].append(float(yp)); outs.append((sid,t,int(row['Store']),y,float(yp),hol))
 out=pd.DataFrame(outs,columns=['id','timestamp','Store','y','yhat','IsHoliday']); out['ae']=(out.y-out.yhat).abs(); out['w']=np.where(out.IsHoliday>.5,5.,1.); return out
preds={}
for c in ['oracle','proxy_simple','proxy_trend','valid_known']:
 preds[c]=predict(c); d=preds[c]; print('RESULT',c,'MAE',d.ae.mean(),'WMAE',(d.ae*d.w).sum()/d.w.sum(),'N',len(d),flush=True)
def smetric(d): return d.groupby('Store').apply(lambda x:pd.Series({'ae_sum':x.ae.sum(),'n':len(x),'wae_sum':(x.ae*x.w).sum(),'w_sum':x.w.sum()}),include_groups=False)
sm={c:smetric(d) for c,d in preds.items()}; stores=sorted(set.intersection(*[set(x.index) for x in sm.values()]))
def agg(tab,ss):
 z=tab.loc[ss]; return z.ae_sum.sum()/z.n.sum(),z.wae_sum.sum()/z.w_sum.sum()
point={c:agg(sm[c],stores) for c in sm}; print('POINT',json.dumps(point),flush=True)
rng=np.random.default_rng(SEED); B=20000; stats={k:[] for k in ['G_mae','P_simple_mae','P_trend_mae','G_wmae','P_simple_wmae','P_trend_wmae']}
for _ in range(B):
 ss=list(rng.choice(stores,len(stores),replace=True)); a={c:agg(sm[c],ss) for c in sm}; stats['G_mae'].append(a['valid_known'][0]-a['oracle'][0]); stats['P_simple_mae'].append(a['proxy_simple'][0]-a['oracle'][0]); stats['P_trend_mae'].append(a['proxy_trend'][0]-a['oracle'][0]); stats['G_wmae'].append(a['valid_known'][1]-a['oracle'][1]); stats['P_simple_wmae'].append(a['proxy_simple'][1]-a['oracle'][1]); stats['P_trend_wmae'].append(a['proxy_trend'][1]-a['oracle'][1])
for k,v in stats.items():
 a=np.asarray(v); print('BOOT',k,'mean',a.mean(),'CI',np.quantile(a,[.025,.975]).tolist(),'PrPos',float((a>0).mean()),flush=True)
for metric,ix in [('MAE',0),('WMAE',1)]:
 G=point['valid_known'][ix]-point['oracle'][ix]
 for p in ['proxy_simple','proxy_trend']:
  P=point[p][ix]-point['oracle'][ix]; R=(point['valid_known'][ix]-point[p][ix])/G if abs(G)>1e-12 else np.nan; print('DECOMP',metric,p,'G',G,'P',P,'R',R,flush=True)
pd.concat([d.assign(condition=c) for c,d in preds.items()],ignore_index=True).to_csv('walmart_audit_predictions.csv',index=False)
