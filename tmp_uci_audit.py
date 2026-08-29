from pathlib import Path
p=Path('tmp_entsoe_audit.py')
s=p.read_text()
s=s.replace('entsoe/1H/train-00000-of-00001.parquet','uci_air_quality/1D/train-00000-of-00001.parquet')
s=s.replace('entsoe_audit_results.csv','uci_audit_results.csv').replace('entsoe_1H.parquet','uci_1D.parquet')
s=s.replace('H = 168','H = 28').replace('N_WINDOWS = 20','N_WINDOWS = 11')
s=s.replace('WEATHER = ["temperature", "radiation_direct_horizontal", "radiation_diffuse_horizontal"]','WEATHER = ["T", "RH", "AH"]')
s=s.replace('for c in ["target","solar_generation_actual","wind_onshore_generation_actual"]+WEATHER:', 'for c in ["target"]+WEATHER:')
s=s.replace('[1,24,168,336]', '[1,7,14,28]').replace('lag24', 'lag7').replace('lag168', 'lag14').replace('lag336', 'lag28')
s=s.replace('rolling(24, min_periods=24)', 'rolling(7, min_periods=7)').replace('rolling(168, min_periods=168)', 'rolling(28, min_periods=28)')
s=s.replace('roll24', 'roll7').replace('roll168', 'roll28')
s=s.replace('arr[-24]', 'arr[-7]').replace('arr[-168]', 'arr[-14]').replace('arr[-336]', 'arr[-28]')
s=s.replace('arr[-24:]', 'arr[-7:]').replace('arr[-168:]', 'arr[-28:]')
s=s.replace('scales[sid]=float(np.mean(np.abs(a[24:]-a[:-24])))','scales[sid]=float(np.mean(np.abs(a[7:]-a[:-7])))')
# Daily covariates: proxy from one week ago and 4-week same-weekday climatology.
s=s.replace('gg.shift(168)', 'gg.shift(7)').replace('[168,336,504,672]', '[7,14,21,28]')
# Daily calendar features: hour terms are constant but harmless; retain same model structure for comparability.
p.write_text(s)
exec(compile(s, str(p), 'exec'))
