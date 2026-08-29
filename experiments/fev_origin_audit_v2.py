import json
from pathlib import Path
import numpy as np, pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=20260829; H=168; N_WINDOWS=20; INNER_FOLDS=4
OUT=Path('tmp_fev_audit_results_v2'); OUT.mkdir(exist_ok=True)
EPOCH=pd.Timestamp('2010-01-01')

def load_data():
    d=load_dataset('autogluon/fev_datasets','proenfo_gfc14'); split=next(iter(d)); ds=d[split]; r=ds[0]
    assert len(ds)==1 and {'timestamp','target','airtemperature'} <= set(ds.column_names)
    df=pd.DataFrame({'timestamp':pd.to_datetime(r['timestamp']),'y':np.asarray(r['target'],float),'temp':np.asarray(r['airtemperature'],float)}).sort_values('timestamp').reset_index(drop=True)
    print('SCHEMA',ds.features); print('N_ROWS',len(df),'START',df.timestamp.iloc[0],'END',df.timestamp.iloc[-1])
    return df

def calendar_features(ts):
    x=pd.DatetimeIndex(ts); hour=x.hour.to_numpy(); dow=x.dayofweek.to_numpy(); doy=x.dayofyear.to_numpy()
    years=((x-EPOCH).total_seconds().to_numpy()/(365.25*24*3600))
    return np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25),years])

def temp_X(ts,temp,idx):
    idx=np.asarray(idx); a=temp[idx-H]; b=temp[idx-2*H]
    return np.column_stack([calendar_features(ts.iloc[idx]),a,b,.5*(a+b),a-b])

def temp_pred(method,ts,temp,c):
    a=temp[c-H:c]; b=temp[c-2*H:c-H]
    if method=='snaive168': return a.copy()
    if method=='avg2': return .5*(a+b)
    if method=='trend2': return np.clip(2*a-b,-50,130)
    start=max(2*H,c-24*365*2); tr=np.arange(start,c); fut=np.arange(c,c+H)
    if method=='ridge_temp': model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    elif method=='hgb_temp': model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=25,learning_rate=.06,l2_regularization=2.0,random_state=SEED)
    else: raise KeyError(method)
    model.fit(temp_X(ts,temp,tr),temp[tr]); return model.predict(temp_X(ts,temp,fut))

def select_temp(ts,temp,c):
    methods=['snaive168','avg2','trend2','ridge_temp','hgb_temp']; scores={m:[] for m in methods}
    for j in range(INNER_FOLDS,0,-1):
        ic=c-j*H; truth=temp[ic:ic+H]
        for m in methods: scores[m].append(mean_absolute_error(truth,temp_pred(m,ts,temp,ic)))
    means={m:float(np.mean(v)) for m,v in scores.items()}; return min(means,key=means.get),means

def load_X(df,idx,temp_future=None,with_temp=True):
    idx=np.asarray(idx); y=df.y.to_numpy(); a=y[idx-H]; b=y[idx-2*H]
    X=np.column_stack([calendar_features(df.timestamp.iloc[idx]),a,b,.5*(a+b),a-b])
    if with_temp:
        t=df.temp.to_numpy()[idx] if temp_future is None else np.asarray(temp_future)
        X=np.column_stack([X,t,t*t/100,np.maximum(65-t,0),np.maximum(t-65,0)])
    return X

def factories():
    return {
      'ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
      'hgb':lambda:HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=31,learning_rate=.06,l2_regularization=1.0,random_state=SEED),
      'rf':lambda:RandomForestRegressor(n_estimators=80,max_depth=16,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=SEED),
      'extra':lambda:ExtraTreesRegressor(n_estimators=80,max_depth=18,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=SEED)}

def boot_ci(x,B=3000):
    x=np.asarray(x,float); rng=np.random.default_rng(SEED); b=np.mean(rng.choice(x,(B,len(x)),replace=True),axis=1); return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def main():
    df=load_data(); n=len(df); first=n-H*N_WINDOWS
    tr_last=calendar_features(df.timestamp.iloc[[first-1]])[0,-1]; te_first=calendar_features(df.timestamp.iloc[[first]])[0,-1]
    assert 0 < te_first-tr_last < 0.001, (tr_last,te_first)
    print('TREND_BOUNDARY_DELTA',te_first-tr_last); print('FIRST_CUTOFF',first,df.timestamp.iloc[first],'LAST',df.timestamp.iloc[n-H])
    rows=[]; trows=[]
    for w in range(N_WINDOWS):
        c=first+w*H; te=np.arange(c,c+H); truth=df.y.to_numpy()[te]; tactual=df.temp.to_numpy()[te]
        sel,cvs=select_temp(df.timestamp,df.temp.to_numpy(),c); torigin=temp_pred(sel,df.timestamp,df.temp.to_numpy(),c); tsn=temp_pred('snaive168',df.timestamp,df.temp.to_numpy(),c)
        trows.append({'window':w,'cutoff':str(df.timestamp.iloc[c]),'selected':sel,'origin_temp_mae':mean_absolute_error(tactual,torigin),'snaive_temp_mae':mean_absolute_error(tactual,tsn),**{f'cv_{k}':v for k,v in cvs.items()},'temp_mean':float(np.mean(tactual)),'temp_min':float(np.min(tactual)),'temp_max':float(np.max(tactual))})
        print('WINDOW',w,'cutoff',df.timestamp.iloc[c],'selected',sel,'tempMAE',round(trows[-1]['origin_temp_mae'],3))
        train=np.arange(2*H,c); Xtr=load_X(df,train,with_temp=True); Xtr0=load_X(df,train,with_temp=False)
        Xs={'oracle':load_X(df,te,tactual,True),'origin':load_X(df,te,torigin,True),'snaive_temp':load_X(df,te,tsn,True),'none':load_X(df,te,with_temp=False)}
        for name,f in factories().items():
            m=f(); m.fit(Xtr,df.y.to_numpy()[train]); m0=f(); m0.fit(Xtr0,df.y.to_numpy()[train])
            for cond,X in Xs.items():
                p=(m0 if cond=='none' else m).predict(X); rows.append({'window':w,'cutoff':str(df.timestamp.iloc[c]),'model':name,'condition':cond,'mae':mean_absolute_error(truth,p),'wape':float(np.sum(np.abs(truth-p))/np.sum(np.abs(truth)))})
    res=pd.DataFrame(rows); td=pd.DataFrame(trows); res.to_csv(OUT/'window_model_metrics.csv',index=False); td.to_csv(OUT/'temperature_selection.csv',index=False)
    wide=res.pivot_table(index=['window','model'],columns='condition',values='mae').reset_index(); wide['optimism']=(wide.origin-wide.oracle)/wide.origin; wide['oracle_lift']=(wide.none-wide.oracle)/wide.none; wide['origin_lift']=(wide.none-wide.origin)/wide.none; wide['lift_shrinkage']=wide.oracle_lift-wide.origin_lift; wide.to_csv(OUT/'paired_effects.csv',index=False)
    summaries=[]
    for model,g in wide.groupby('model'):
        gap=g.origin-g.oracle; summaries.append({'model':model,'oracle_mae_mean':float(g.oracle.mean()),'origin_mae_mean':float(g.origin.mean()),'none_mae_mean':float(g.none.mean()),'mean_optimism':float(g.optimism.mean()),'median_optimism':float(g.optimism.median()),'frac_optimism_ge_20pct':float((g.optimism>=.2).mean()),'mean_oracle_lift':float(g.oracle_lift.mean()),'mean_origin_lift':float(g.origin_lift.mean()),'mean_lift_shrinkage':float(g.lift_shrinkage.mean()),'gap_ci95':boot_ci(gap)})
    ms=pd.DataFrame(summaries); ms.to_csv(OUT/'model_summary.csv',index=False)
    models=sorted(res.model.unique()); pairs=[(a,b) for i,a in enumerate(models) for b in models[i+1:]]; rr=[]
    for w in range(N_WINDOWS):
        z=res[res.window.eq(w)].pivot(index='model',columns='condition',values='mae'); ro=z.oracle.rank(); rg=z.origin.rank(); rev=sum(np.sign(z.loc[a,'oracle']-z.loc[b,'oracle'])!=np.sign(z.loc[a,'origin']-z.loc[b,'origin']) for a,b in pairs)
        rr.append({'window':w,'spearman':float(spearmanr(ro,rg).statistic),'kendall':float(kendalltau(ro,rg).statistic),'top1_oracle':z.oracle.idxmin(),'top1_origin':z.origin.idxmin(),'top1_switch':int(z.oracle.idxmin()!=z.origin.idxmin()),'pairwise_reversal_rate':rev/len(pairs)})
    ranks=pd.DataFrame(rr); ranks.to_csv(OUT/'rank_distortion.csv',index=False)
    gapw=wide.groupby('window').apply(lambda x:np.mean(x.origin-x.oracle),include_groups=False); diag=td.set_index('window').copy(); diag['mean_load_gap']=gapw; diag.to_csv(OUT/'window_diagnostics.csv')
    summary={'dataset':'autogluon/fev_datasets::proenfo_gfc14','horizon':H,'num_windows':N_WINDOWS,'model_summary':summaries,'ranking':{'mean_spearman':float(ranks.spearman.mean()),'median_spearman':float(ranks.spearman.median()),'mean_kendall':float(ranks.kendall.mean()),'top1_switch_rate':float(ranks.top1_switch.mean()),'mean_pairwise_reversal_rate':float(ranks.pairwise_reversal_rate.mean()),'windows_spearman_lt_0_9':int((ranks.spearman<.9).sum())},'temperature':{'mean_origin_temp_mae':float(td.origin_temp_mae.mean()),'median_origin_temp_mae':float(td.origin_temp_mae.median()),'selected_counts':{str(k):int(v) for k,v in td.selected.value_counts().items()},'rho_temp_mae_vs_load_gap':float(spearmanr(diag.origin_temp_mae,diag.mean_load_gap).statistic)},'gates':{'any_model_mean_optimism_ge_20pct':bool((ms.mean_optimism>=.2).any()),'any_model_mean_lift_shrinkage_ge_30pct':bool((ms.mean_lift_shrinkage>=.3).any()),'any_rank_reversal':bool((ranks.pairwise_reversal_rate>0).any()),'top1_switch_rate':float(ranks.top1_switch.mean())}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print('===MODEL==='); print(ms.to_string(index=False)); print('===RANK==='); print(ranks.to_string(index=False)); print('===JSON==='); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
