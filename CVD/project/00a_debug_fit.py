import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
from patsy import bs
import os
warnings.filterwarnings('ignore')

def read_csv_auto(path):
    for enc in ['utf-8-sig','utf-8','gbk']:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.read_csv(path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")
uwt_h = read_csv_auto(os.path.join(DATA_DIR, '204High_Stroke_SDI_zone_all.csv'))
uwt_c = read_csv_auto(os.path.join(DATA_DIR, '204Low_Stroke_SDI_zone_all.csv'))
pwt_d = read_csv_auto(os.path.join(DATA_DIR, 'Mean_pop_Stroke_SDI_zone_all.csv'))

id_cols = ['iso_a3','year','location','sex','age','cause','measure','metric']
eid = [c for c in id_cols if c in uwt_h.columns]
ht = list(set(uwt_h.columns) - set(eid))
ck = [c for c in uwt_c.columns if c in eid or c not in ht]
muwt = pd.merge(uwt_h[eid+ht], uwt_c[ck], on=eid, how='outer')
muwt['year'] = pd.to_numeric(muwt['year'], errors='coerce')
pid = [c for c in eid if c in pwt_d.columns]
pwt_d['year'] = pd.to_numeric(pwt_d['year'], errors='coerce')
pm = [c for c in pwt_d.columns if c.startswith('mean_')]
mcols = eid + ['val']
df_u = muwt[mcols + ht + [c for c in uwt_c.columns if c not in ht and c not in eid]].copy()
df_p = pd.merge(muwt[mcols], pwt_d[pid+pm], on=pid, how='inner')

for col_u, col_p, label in [
    ('Mean_Temp','mean_Mean','年均温'),
    ('Max_Temp','mean_Max','年最高温'),
    ('Min_Temp','mean_Min','年最低温'),
    ('P90','mean_P90','P90'),
    ('P1','mean_P1','P1'),
    ('Heatwave_P90_2d','mean_Heatwave_P90_2d','HW_P90_2d'),
    ('Coldspell_P3_2d','mean_Coldspell_P3_2d','CS_P3_2d'),
]:
    for c, d, n in [(col_u, df_u, 'UWT'), (col_p, df_p, 'PWT')]:
        print(f'{n}[{label}] col={c} in df: {c in d.columns}')
        if c in d.columns:
            sf = c.replace('.','_')
            mdf = d[['val',c,'location','year']].dropna().rename(columns={c:sf})
            print(f'  rows={len(mdf)}, unique vals={mdf[sf].nunique()}')
            
            formula = "Q('val') ~ bs(Q('{}'), df=4, degree=3) + C(Q('location')) + bs(Q('year'), df=3, degree=3)".format(sf)
            try:
                m = smf.ols(formula, data=mdf).fit()
                print(f'  FIT OK, R2={m.rsquared:.4f}')
            except Exception as e:
                print(f'  ERROR: {str(e)[:300]}')
    print()
