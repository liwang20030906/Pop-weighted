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

eid = [c for c in ['iso_a3','year','location','sex','age','cause','measure','metric'] if c in uwt_h.columns]
ht = list(set(uwt_h.columns) - set(eid))
ck = [c for c in uwt_c.columns if c in eid or c not in ht]
muwt = pd.merge(uwt_h[eid+ht], uwt_c[ck], on=eid, how='outer')
muwt['year'] = pd.to_numeric(muwt['year'], errors='coerce')
pid = [c for c in eid if c in pwt_d.columns]
pwt_d['year'] = pd.to_numeric(pwt_d['year'], errors='coerce')
pm = [c for c in pwt_d.columns if c.startswith('mean_')]
cold_extra = [c for c in uwt_c.columns if c not in ht and c not in eid]
_uwt_cols = eid + ['val'] + [c for c in ht if c not in eid] + [c for c in cold_extra if c not in eid]
_uwt_final = list(dict.fromkeys(_uwt_cols))
df_u = muwt[_uwt_final].copy()
pwt_tmp = pwt_d[pid+pm].copy()
df_p = pd.merge(muwt[eid+['val']], pwt_tmp, on=pid, how='inner')

# Test
mdf = df_p[['val','mean_P90','mean_Mean','location','year']].dropna().copy()
print('Cols:', mdf.columns.tolist())
print('Shape:', mdf.shape)

# Try different formula styles
for formula in [
    'val ~ mean_P90 + bs(mean_Mean, df=4, degree=3) + C(location) + bs(year, df=3, degree=3)',
    "Q('val') ~ Q('mean_P90') + bs(Q('mean_Mean'), df=4, degree=3) + C(Q('location')) + bs(Q('year'), df=3, degree=3)",
]:
    print(f'\nFormula: {formula}')
    try:
        m = smf.ols(formula, data=mdf).fit()
        print(f'  OK! coef={m.params["mean_P90"]:.4f}, p={m.pvalues["mean_P90"]:.4f}')
    except Exception as e:
        print(f'  ERROR: {e[:200]}')
