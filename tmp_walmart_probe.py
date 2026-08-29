from pathlib import Path
import pandas as pd, requests, numpy as np
URL='https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/walmart/train-00000-of-00001.parquet?download=true'
p=Path('walmart.parquet'); r=requests.get(URL,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
df=pd.read_parquet(p)
print('DOWNLOADED',len(r.content),flush=True)
print('SHAPE',df.shape,flush=True)
print('COLUMNS',df.columns.tolist(),flush=True)
print('DTYPES',df.dtypes.astype(str).to_dict(),flush=True)
print('IDS_HEAD',df['id'].head().tolist() if 'id' in df else None,flush=True)
for c in df.columns:
    try:
        v=df.iloc[0][c]
        if isinstance(v,(list,np.ndarray)):
            a=np.asarray(v)
            print('COL',c,'ARRAY',a.shape,'dtype',a.dtype,'head',a[:5].tolist(),flush=True)
        else:
            print('COL',c,'SCALAR',repr(v),flush=True)
    except Exception as e: print('COLERR',c,repr(e),flush=True)
