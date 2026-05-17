"""Extract FE-GAM model coefficients for all indicators."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs
import warnings
import os
warnings.filterwarnings("ignore")


def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")

# Merge data
uwt_heat = read_csv_auto(os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"))
uwt_cold = read_csv_auto(os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"))
pwt = read_csv_auto(os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"))

id_cols = ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]
existing_id = [c for c in id_cols if c in uwt_heat.columns]
high_temp = list(set(uwt_heat.columns) - set(existing_id))
cold_keep = [c for c in uwt_cold.columns if c in existing_id or c not in high_temp]
merged_uwt = pd.merge(
    uwt_heat[existing_id + high_temp],
    uwt_cold[cold_keep],
    on=existing_id,
    how="outer",
)
model_cols = existing_id + ["val", "upper", "lower", "SDI", "Zone", "Continent"]
pwt_id = [c for c in existing_id if c in pwt.columns]
pwt_temp = [c for c in pwt.columns if c.startswith("mean_")]
merged_uwt["year"] = pd.to_numeric(merged_uwt["year"], errors="coerce")
pwt["year"] = pd.to_numeric(pwt["year"], errors="coerce")
df = pd.merge(merged_uwt[model_cols], pwt[pwt_id + pwt_temp], on=pwt_id, how="inner")

# Indicators to analyze
indicators = {
    "基础温度_年均温": "mean_Mean",
    "基础温度_年最高温": "mean_Max",
    "基础温度_年最低温": "mean_Min",
    "极端热_P88.9": "mean_P88.9",
    "极端热_P89.9": "mean_P89.9",
    "极端热_P90": "mean_P90",
    "极端热_P92.5": "mean_P92.5",
    "极端热_P93": "mean_P93",
    "极端热_P95": "mean_P95",
    "极端热_P97": "mean_P97",
    "极端热_P98": "mean_P98",
    "极端热_P99": "mean_P99",
    "极端冷_P1": "mean_P1",
    "极端冷_P2.5": "mean_P2.5",
    "极端冷_P3": "mean_P3",
    "极端冷_P5": "mean_P5",
    "极端冷_P7.5": "mean_P7.5",
    "极端冷_P10": "mean_P10",
}

print("=" * 70)
print("FE-GAM 各模型温度样条系数")
print("=" * 70)

for name, col in indicators.items():
    safe = col.replace(".", "_")
    mdf = df[["val", col, "location", "year"]].dropna().rename(columns={col: safe}).copy()

    outcome_q = "Q('val')"
    indicator_q = "Q('" + safe + "')"
    location_q = "Q('location')"
    year_q = "Q('year')"

    formula = (
        outcome_q + " ~ bs(" + indicator_q + ", df=4, degree=3) + "
        "C(" + location_q + ") + bs(" + year_q + ", df=3, degree=3)"
    )
    formula0 = "Q('val') ~ C(Q('location')) + bs(Q('year'), df=3, degree=3)"

    m = smf.ols(formula, data=mdf).fit()
    m0 = smf.ols(formula0, data=mdf).fit()
    anova = sm.stats.anova_lm(m0, m)
    p_val = anova.iloc[1]["Pr(>F)"] if len(anova) >= 2 else None
    f_val = anova.iloc[1]["F"] if len(anova) >= 2 else None

    bs_coefs = {k: v for k, v in m.params.items() if safe in k and "bs" in k}

    print(f"\n[{name}]  均值={mdf[safe].mean():.4f}  R2={m.rsquared:.4f}  F({len(bs_coefs)},{m.df_resid})={f_val:.4f}  p={p_val:.6f}")
    print("  样条系数 (B-spline, df=4, degree=3):")
    for k, v in bs_coefs.items():
        se = m.bse[k]
        t = v / se
        pv = m.pvalues[k]
        stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"    {k:<50s} = {v:+.4f}  (SE={se:.4f}, t={t:+.3f}, p={pv:.4f}) {stars}")

print()
print("=" * 70)
print("模型结构说明")
print("=" * 70)
print("""
Formula:
  val ~ bs(temperature, df=4, degree=3) + C(location) + bs(year, df=3, degree=3)

val     : GBD 2019 脑卒中年龄标化死亡率 (每10万人)
bs(t,4) : 三次B样条, 4自由度 → 4个基函数系数
C(loc)  : 191个国家虚拟变量 (固定效应)
bs(yr,3): 时间趋势三次B样条, 3自由度

F检验  : 完整模型 vs 简化模型(无温度项) 的方差分析
         评估温度指标的联合显著性

数据来源:
  温度   : ERA5-Land 人口加权 (PWT)
  健康   : GBD 2019 Stroke
  样本   : 4011 obs, 191 countries, 2000-2020
""")
