"""
极端温度对卒中死亡率的影响评估 —— UWT vs PWT 对比
方法: 模型中加入年均温作为控制变量，分离极端暴露的独立效应
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs
import warnings
import os
warnings.filterwarnings("ignore")


# =========================
# 1. 数据准备
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")

uwt_heat = read_csv_auto(os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"))
uwt_cold = read_csv_auto(os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"))
pwt_data = read_csv_auto(os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"))

id_cols = ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]
eid = [c for c in id_cols if c in uwt_heat.columns]
ht = list(set(uwt_heat.columns) - set(eid))
ck = [c for c in uwt_cold.columns if c in eid or c not in ht]
muwt = pd.merge(uwt_heat[eid + ht], uwt_cold[ck], on=eid, how="outer")
muwt["year"] = pd.to_numeric(muwt["year"], errors="coerce")
pid = [c for c in eid if c in pwt_data.columns]
pwt_data["year"] = pd.to_numeric(pwt_data["year"], errors="coerce")
pm = [c for c in pwt_data.columns if c.startswith("mean_")]

# UWT 数据
cold_extra = [c for c in uwt_cold.columns if c not in ht and c not in eid]
_uwt_cols = eid + ["val"] + [c for c in ht if c not in eid] + [c for c in cold_extra if c not in eid]
_uwt_final = list(dict.fromkeys(_uwt_cols))
df_uwt = muwt[_uwt_final].copy()

# PWT 数据 (使用 UWT 的 val)
pwt_tmp = pwt_data[pid + pm].copy()
df_pwt = pd.merge(muwt[eid + ["val"]], pwt_tmp, on=pid, how="inner")
_pwt_final = list(dict.fromkeys(df_pwt.columns))
df_pwt = df_pwt[_pwt_final].copy()

print(f"UWT: {df_uwt.shape}, PWT: {df_pwt.shape}")

# =========================
# 2. 指标定义
# =========================
# 关联: (类别, 极端指标_UWT, 极端指标_PWT, 控制变量_UWT, 控制变量_PWT, 简称)
ANALYSIS_PAIRS = [
    # ---- 极端热 vs 年均温 ----
    ("极端热_P90",  "P90",  "mean_P90",  "Mean_Temp", "mean_Mean", "P90"),
    ("极端热_P93",  "P93",  "mean_P93",  "Mean_Temp", "mean_Mean", "P93"),
    ("极端热_P95",  "P95",  "mean_P95",  "Mean_Temp", "mean_Mean", "P95"),
    ("极端热_P97",  "P97",  "mean_P97",  "Mean_Temp", "mean_Mean", "P97"),
    ("极端热_P98",  "P98",  "mean_P98",  "Mean_Temp", "mean_Mean", "P98"),
    ("极端热_P99",  "P99",  "mean_P99",  "Mean_Temp", "mean_Mean", "P99"),

    # ---- 极端冷 vs 年均温 ----
    ("极端冷_P1",   "P1",   "mean_P1",   "Mean_Temp", "mean_Mean", "P1"),
    ("极端冷_P3",   "P3",   "mean_P3",   "Mean_Temp", "mean_Mean", "P3"),
    ("极端冷_P5",   "P5",   "mean_P5",   "Mean_Temp", "mean_Mean", "P5"),
    ("极端冷_P10",  "P10",  "mean_P10",  "Mean_Temp", "mean_Mean", "P10"),

    # ---- 热浪天数 vs 年均温+年最高温(控制一般热水平) ----
    ("热浪_P90_2d", "Heatwave_P90_2d", "mean_Heatwave_P90_2d", None, None, "HW_P90_2d"),
    ("热浪_P93_2d", "Heatwave_P93_2d", "mean_Heatwave_P93_2d", None, None, "HW_P93_2d"),
    ("热浪_P97_2d", "Heatwave_P97_2d", "mean_Heatwave_P97_2d", None, None, "HW_P97_2d"),
    ("热浪_P93_3d", "Heatwave_P93_3d", "mean_Heatwave_P93_3d", None, None, "HW_P93_3d"),
    ("热浪_P90_3d", "Heatwave_P90_3d", "mean_Heatwave_P90_3d", None, None, "HW_P90_3d"),

    # ---- 冷事件天数 vs 年均温+年最低温 ----
    ("冷事件_P3_2d",  "Coldspell_P3_2d",  "mean_Coldspell_P3_2d",  None, None, "CS_P3_2d"),
    ("冷事件_P5_2d",  "Coldspell_P5_2d",  "mean_Coldspell_P5_2d",  None, None, "CS_P5_2d"),
    ("冷事件_P10_2d", "Coldspell_P10_2d", "mean_ColdSpell_P10_2d", None, None, "CS_P10_2d"),
    ("冷事件_P3_3d",  "Coldspell_P3_3d",  "mean_Coldspell_P3_3d",  None, None, "CS_P3_3d"),
    ("冷事件_P5_3d",  "Coldspell_P5_3d",  "mean_Coldspell_P5_3d",  None, None, "CS_P5_3d"),
]

# 默认控制变量组 (heatwave 用 mean+max, coldspell 用 mean+min)
DEFAULT_CTRL_HEAT = [("Mean_Temp", "mean_Mean"), ("Max_Temp", "mean_Max")]
DEFAULT_CTRL_COLD = [("Mean_Temp", "mean_Mean"), ("Min_Temp", "mean_Min")]


def safe_name(c):
    return c.replace(".", "_").replace(" ", "_")


def fit_model(df, extreme_col, ctrl_cols, label):
    """val ~ extreme + bs(ctrl1) + bs(ctrl2) + C(location) + bs(year)"""
    cols_needed = ["val", extreme_col, "location", "year"] + [c for c in ctrl_cols if c is not None]
    cols_available = [c for c in cols_needed if c in df.columns]
    missing = set(cols_needed) - set(cols_available)
    if missing:
        return {"error": f"Missing cols: {missing}"}

    mdf = df[cols_available].dropna().copy()
    rename = {}
    for c in mdf.columns:
        sf = safe_name(c)
        if sf != c:
            rename[c] = sf
    mdf = mdf.rename(columns=rename)

    safe_extreme = safe_name(extreme_col)
    safe_ctrls = [safe_name(c) for c in ctrl_cols if c is not None]

    if len(mdf) < 100:
        return {"error": "Too few obs"}

    # 构建公式 — 使用 safe name (无特殊字符) 直接嵌入
    formula_parts = ["val ~ {}".format(safe_extreme)]
    for sc in safe_ctrls:
        formula_parts.append("bs({}, df=4, degree=3)".format(sc))
    formula_parts.append("C(location)")
    formula_parts.append("bs(year, df=3, degree=3)")

    formula = " + ".join(formula_parts)
    formula0 = "val ~ " + " + ".join(
        ["bs({}, df=4, degree=3)".format(sc) for sc in safe_ctrls] +
        ["C(location)", "bs(year, df=3, degree=3)"]
    )

    try:
        model = smf.ols(formula, data=mdf).fit()
    except Exception as e:
        return {"error": str(e)[:200]}

    # 极端指标的系数 (线性项) — 直接用 safe name
    coef = model.params.get(safe_extreme, np.nan)
    se = model.bse.get(safe_extreme, np.nan)
    t_val = coef / se if se != 0 and not np.isnan(se) else np.nan
    p_val = model.pvalues.get(safe_extreme, np.nan)
    ci = [np.nan, np.nan]
    try:
        ci_df = model.conf_int()
        if safe_extreme in ci_df.index:
            ci = [ci_df.loc[safe_extreme, 0], ci_df.loc[safe_extreme, 1]]
    except Exception:
        pass

    # 模型比较: 完整模型 vs 不含极端指标的模型
    try:
        model0 = smf.ols(formula0, data=mdf).fit()
        anova = sm.stats.anova_lm(model0, model)
        f_val = anova.iloc[1]["F"] if len(anova) >= 2 else np.nan
        p_anova = anova.iloc[1]["Pr(>F)"] if len(anova) >= 2 else np.nan
    except Exception:
        f_val, p_anova = np.nan, np.nan

    # 极端暴露每IQR变化对应的死亡率变化
    iqr = mdf[safe_extreme].quantile(0.75) - mdf[safe_extreme].quantile(0.25)
    effect_per_iqr = coef * iqr

    return {
        "extreme_col": extreme_col,
        "coef": coef, "se": se, "t": t_val, "p": p_val,
        "ci_low": ci[0], "ci_high": ci[1],
        "f_anova": f_val, "p_anova": p_anova,
        "iqr": iqr, "effect_per_iqr": effect_per_iqr,
        "n_obs": int(model.nobs),
        "rsquared": model.rsquared,
        "df_resid": model.df_resid,
    }


# =========================
# 3. 运行分析
# =========================
print("\n" + "=" * 100)
print("  极端温度对卒中死亡率的独立效应 —— UWT vs PWT 对比")
print("  方法: 控制年均温(热浪加控年最高温, 冷事件加控年最低温) + 国家FE + 年份平滑")
print("=" * 100)

results = []

for cat, col_u, col_p, ctrl_u, ctrl_p, label in ANALYSIS_PAIRS:
    # 确定控制变量
    if cat.startswith("热浪"):
        ctrls_u = [(c, c) for c in [cu for cu, cp in DEFAULT_CTRL_HEAT] if (ctrl_u is None or ctrl_u == cu)]
        ctrls_p = [(c, c) for c in [cp for cu, cp in DEFAULT_CTRL_HEAT] if (ctrl_p is None or ctrl_p == cp)]
        ctrls_u_raw = [cu for cu, cp in DEFAULT_CTRL_HEAT]
        ctrls_p_raw = [cp for cu, cp in DEFAULT_CTRL_HEAT]
    elif cat.startswith("冷事件"):
        ctrls_u_raw = [cu for cu, cp in DEFAULT_CTRL_COLD]
        ctrls_p_raw = [cp for cu, cp in DEFAULT_CTRL_COLD]
    else:
        ctrls_u_raw = [ctrl_u] if ctrl_u else ["Mean_Temp"]
        ctrls_p_raw = [ctrl_p] if ctrl_p else ["mean_Mean"]

    # UWT
    res_u = fit_model(df_uwt, col_u, ctrls_u_raw, f"UWT_{label}")
    # PWT
    res_p = fit_model(df_pwt, col_p, ctrls_p_raw, f"PWT_{label}")

    for res, src in [(res_u, "UWT"), (res_p, "PWT")]:
        if "error" in res:
            print(f"  {src} {label:>15s}: SKIP ({res['error']})")
            continue
        res["source"] = src
        res["category"] = cat
        res["label"] = label
        results.append(res)

# =========================
# 4. 输出对比表
# =========================
# 先按类别分组
cats_order = ["极端热_P90", "极端热_P93", "极端热_P95", "极端热_P97", "极端热_P98", "极端热_P99",
              "极端冷_P1", "极端冷_P3", "极端冷_P5", "极端冷_P10",
              "热浪_P90_2d", "热浪_P93_2d", "热浪_P97_2d", "热浪_P93_3d", "热浪_P90_3d",
              "冷事件_P3_2d", "冷事件_P5_2d", "冷事件_P10_2d", "冷事件_P3_3d", "冷事件_P5_3d"]

print(f"\n{'='*100}")
print(f"  {'指标':<18s} {'来源':>4s} {'Coef(per 1 unit)':>16s} {'SE':>8s} {'p值':>8s} {'per IQR效应':>14s} {'F_anova':>10s} {'p_anova':>10s} {'R^2':>8s}")
print(f"  {'-'*18} {'-'*4} {'-'*16} {'-'*8} {'-'*8} {'-'*14} {'-'*10} {'-'*10} {'-'*8}")

for cat in cats_order:
    grp = [r for r in results if r["category"] == cat]
    if not grp:
        continue
    print(f"\n  [{cat}]")
    for r in sorted(grp, key=lambda x: x["source"]):
        p_str = f'{r["p"]:<8.4f}' if not np.isnan(r["p"]) else f'{"N/A":<8s}'
        stars = " ***" if r["p"] < 0.001 else " **" if r["p"] < 0.01 else " *" if r["p"] < 0.05 else ""
        pa_str = f'{r["p_anova"]:<10.4f}' if not np.isnan(r["p_anova"]) else f'{"N/A":<10s}'
        coef_str = f'{r["coef"]:+.4f}'
        print(f"  {cat:<18s} {r['source']:>4s} {coef_str:>16s} {r['se']:<8.4f} {p_str:<8s}{stars} {r['effect_per_iqr']:+14.4f} {r['f_anova']:<10.3f} {pa_str:<10s} {r['rsquared']:<8.4f}")

# =========================
# 5. UWT vs PWT 系数对比总结
# =========================
print(f"\n{'='*100}")
print(f"  UWT vs PWT 系数对比 (每1单位指标变化对应死亡率变化)")
print(f"{'='*100}")
print(f"  {'指标':<20s} {'UWT Coef':>12s} {'PWT Coef':>12s} {'差异':>12s} {'UWT显著':>10s} {'PWT显著':>10s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

comparisons = {}
for r in results:
    k = r["label"]
    if k not in comparisons:
        comparisons[k] = {}
    comparisons[k][r["source"]] = r

for label in sorted(comparisons.keys()):
    u = comparisons[label].get("UWT", {})
    p = comparisons[label].get("PWT", {})
    if not u or not p:
        continue
    cu, pu = u.get("coef", np.nan), u.get("p", np.nan)
    cp, pp = p.get("coef", np.nan), p.get("p", np.nan)
    diff = cp - cu
    usig = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else ""
    psig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
    print(f"  {label:<20s} {cu:>+12.4f} {cp:>+12.4f} {diff:>+12.4f} {usig:>10s} {psig:>10s}")

print()
print("解读: Coef > 0 表示极端指标升高 → 卒中死亡率升高")
print("      IQR效应 = Coef x 四分位距, 表示该指标从P25升到P75时死亡率的变化")
