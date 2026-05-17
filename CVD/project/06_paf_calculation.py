"""
年均温的冷/热可归因分数 (PAF) —— UWT vs PWT 对比
方法: FE-GAM -> MRT -> 人口加权 PAF + Bootstrap CI
参考: R code compute_paf
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs
import warnings
import os

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")
GBD_DIR = os.path.join(BASE_DIR, "..", "data", "0.GBD")


def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


# =========================
# 1. 数据准备
# =========================
uwt_heat = read_csv_auto(os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"))
uwt_cold = read_csv_auto(os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"))
pwt_data = read_csv_auto(os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"))
pop_data = read_csv_auto(os.path.join(GBD_DIR, "1.1GBD_Pop_location_2000-2020_iso_a3.csv"))

# Population data
pop_df = pop_data[["iso_a3", "year", "val"]].copy()
pop_df = pop_df.rename(columns={"val": "popnum"})
pop_df["year"] = pd.to_numeric(pop_df["year"], errors="coerce")

# Merge UWT high + cold
id_cols = ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]
eid = [c for c in id_cols if c in uwt_heat.columns]
ht = list(set(uwt_heat.columns) - set(eid))
ck = [c for c in uwt_cold.columns if c in eid or c not in ht]
muwt = pd.merge(uwt_heat[eid + ht], uwt_cold[ck], on=eid, how="outer")
muwt["year"] = pd.to_numeric(muwt["year"], errors="coerce")
pid = [c for c in eid if c in pwt_data.columns]
pwt_data["year"] = pd.to_numeric(pwt_data["year"], errors="coerce")
pm = [c for c in pwt_data.columns if c.startswith("mean_")]

# UWT: health + UWT temp + population
cold_extra = [c for c in uwt_cold.columns if c not in ht and c not in eid]
_uwt_cols = eid + ["val"] + [c for c in ht if c not in eid] + [c for c in cold_extra if c not in eid]
df_uwt = muwt[list(dict.fromkeys(_uwt_cols))].copy()
df_uwt = pd.merge(df_uwt, pop_df, on=["iso_a3", "year"], how="inner")

# PWT: health from UWT + PWT temp + population
df_pwt_raw = pd.merge(muwt[eid + ["val"]], pwt_data[pid + pm], on=pid, how="inner")
df_pwt = df_pwt_raw[list(dict.fromkeys(df_pwt_raw.columns))].copy()
df_pwt = pd.merge(df_pwt, pop_df, on=["iso_a3", "year"], how="inner")

print(f"UWT data: {df_uwt.shape[0]} rows, {df_uwt['location'].nunique()} locations")
print(f"PWT data: {df_pwt.shape[0]} rows, {df_pwt['location'].nunique()} locations")


# =========================
# 2. FE-GAM 拟合
# =========================
def fit_gam_mean_temp(df, temp_col):
    """Fit FE-GAM: val ~ bs(temp, df=4) + C(location) + bs(year, df=3)"""
    sf = temp_col.replace(".", "_").replace(" ", "_")
    mdf = df[["val", temp_col, "location", "year", "popnum"]].dropna().rename(
        columns={temp_col: sf}).copy()

    formula = (
        "Q('val') ~ bs(Q('" + sf + "'), df=4, degree=3) + "
        "C(Q('location')) + bs(Q('year'), df=3, degree=3)"
    )
    model = smf.ols(formula, data=mdf).fit()
    return model, mdf, sf


# =========================
# 3. 人口加权 PAF 计算 (正确方法)
# =========================
def compute_paf(model, mdf, sf, n_boot=500, seed=2025):
    """
    Population-weighted PAF.
    Formula: PAF = sum(max(pred - pred(MRT), 0) * popnum) / sum(pred * popnum)
    """
    np.random.seed(seed)
    df_in = mdf.reset_index(drop=True).copy()

    # 1. Predictions at observed temps
    df_in["pred"] = model.predict(df_in)

    # 2. Grid search for MRT
    x_min, x_max = df_in[sf].min(), df_in[sf].max()
    grid = pd.DataFrame({
        sf: np.linspace(x_min, x_max, 400),
        "location": df_in["location"].values[0],
        "year": df_in["year"].values[0],
    })
    grid["fit"] = model.predict(grid)
    MRT = grid[sf].values[np.argmin(grid["fit"].values)]

    # 3. Counterfactual prediction at MRT
    df_ref = df_in[["location", "year"]].copy()
    df_ref[sf] = MRT
    df_in["pred_ref"] = model.predict(df_ref)

    # 4. Population-weighted PAF
    delta = np.maximum(df_in["pred"] - df_in["pred_ref"], 0)
    denominator = np.sum(df_in["pred"] * df_in["popnum"])

    cold_mask = df_in[sf] < MRT
    heat_mask = df_in[sf] > MRT

    PAF_total = np.sum(delta * df_in["popnum"]) / denominator
    PAF_cold = np.sum(delta[cold_mask] * df_in["popnum"][cold_mask]) / denominator
    PAF_heat = np.sum(delta[heat_mask] * df_in["popnum"][heat_mask]) / denominator

    # 5. Bootstrap CIs (resample, no model refitting)
    n = len(df_in)
    boot_paf = np.zeros((n_boot, 3))

    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        samp = df_in.iloc[idx].reset_index(drop=True).copy()

        samp["pred"] = model.predict(samp)
        samp_ref = samp[["location", "year"]].copy()
        samp_ref[sf] = MRT
        samp["pred_ref"] = model.predict(samp_ref)

        delta_b = np.maximum(samp["pred"] - samp["pred_ref"], 0)
        denom_b = np.sum(samp["pred"] * samp["popnum"])

        boot_paf[b, 0] = np.sum(delta_b * samp["popnum"]) / denom_b
        c_b = samp[sf] < MRT
        boot_paf[b, 1] = np.sum(delta_b[c_b] * samp["popnum"][c_b]) / denom_b
        h_b = samp[sf] > MRT
        boot_paf[b, 2] = np.sum(delta_b[h_b] * samp["popnum"][h_b]) / denom_b

    LCI = np.percentile(boot_paf, 2.5, axis=0)
    UCI = np.percentile(boot_paf, 97.5, axis=0)

    return {
        "MRT": MRT,
        "PAF_total": PAF_total, "PAF_cold": PAF_cold, "PAF_heat": PAF_heat,
        "PAF_total_LCI": LCI[0], "PAF_cold_LCI": LCI[1], "PAF_heat_LCI": LCI[2],
        "PAF_total_UCI": UCI[0], "PAF_cold_UCI": UCI[1], "PAF_heat_UCI": UCI[2],
        "n_obs": n, "n_cold": int(cold_mask.sum()), "n_heat": int(heat_mask.sum()),
        "mean_temp": df_in[sf].mean(),
        "cold_mean_temp": df_in.loc[cold_mask, sf].mean() if cold_mask.sum() > 0 else np.nan,
        "heat_mean_temp": df_in.loc[heat_mask, sf].mean() if heat_mask.sum() > 0 else np.nan,
    }


# =========================
# 4. 主程序
# =========================
print("=" * 80)
print("  人口加权 PAF: 年均温对卒中死亡率 (UWT vs PWT)")
print("  公式: PAF = sum( max(pred - pred(MRT), 0) * popnum ) / sum( pred * popnum )")
print("=" * 80)

# UWT
print("\n[1/2] Fitting UWT model (Mean_Temp)...")
model_uwt, mdf_uwt, sf_uwt = fit_gam_mean_temp(df_uwt, "Mean_Temp")
print(f"      R^2 = {model_uwt.rsquared:.4f}")
uwt_paf = compute_paf(model_uwt, mdf_uwt, sf_uwt)

# PWT
print("[2/2] Fitting PWT model (mean_Mean)...")
model_pwt, mdf_pwt, sf_pwt = fit_gam_mean_temp(df_pwt, "mean_Mean")
print(f"      R^2 = {model_pwt.rsquared:.4f}")
pwt_paf = compute_paf(model_pwt, mdf_pwt, sf_pwt)

# =========================
# 5. 输出
# =========================
print("\n" + "=" * 80)
print("  RESULTS")
print("=" * 80)

print(f"\n  {'':<35} {'UWT':>20} {'PWT':>20} {'差异':>20}")
print(f"  {'-'*35} {'-'*20} {'-'*20} {'-'*20}")
print(f"  {'R^2':<35} {model_uwt.rsquared:>20.4f} {model_pwt.rsquared:>20.4f}")
print(f"  {'MRT (K)':<35} {uwt_paf['MRT']:>20.2f} {pwt_paf['MRT']:>20.2f} "
      f"{pwt_paf['MRT'] - uwt_paf['MRT']:>+20.2f}")
print(f"  {'样本量':<35} {uwt_paf['n_obs']:>20} {pwt_paf['n_obs']:>20}")
print(f"  {'平均温度 (K)':<35} {uwt_paf['mean_temp']:>20.2f} {pwt_paf['mean_temp']:>20.2f}")

print(f"\n  {'--- 人口加权 PAFs (Bootstrap 95% CI) ---':^85}")

items = [("PAF_total", "总 PAF"), ("PAF_cold", "冷效应 PAF"), ("PAF_heat", "热效应 PAF")]

for key, label in items:
    u = uwt_paf[key]; ul = uwt_paf[f"{key}_LCI"]; uu = uwt_paf[f"{key}_UCI"]
    p = pwt_paf[key]; pl = pwt_paf[f"{key}_LCI"]; pu = pwt_paf[f"{key}_UCI"]

    print(f"\n  [{label}]")
    print(f"    UWT = {u:.4%}  (95% CI: {ul:.4%} - {uu:.4%})")
    print(f"    PWT = {p:.4%}  (95% CI: {pl:.4%} - {pu:.4%})")
    print(f"    Δ   = {p - u:+.4%}")

print(f"\n  {'--- 温度分布 ---':^60}")
print(f"  {'':<35} {'UWT':>20} {'PWT':>20}")
print(f"  {'冷观测数':<35} {uwt_paf['n_cold']:>20} {pwt_paf['n_cold']:>20}")
print(f"  {'热观测数':<35} {uwt_paf['n_heat']:>20} {pwt_paf['n_heat']:>20}")

print("\n" + "=" * 80)
print("  公式说明")
print("=" * 80)
print("""
  1. GAM:   val ~ bs(T_mean, df=4) + C(location) + bs(year, df=3)
  2. MRT:   argmin f(T) on grid of 400 temperature points
  3. delta:  max( f(T_i) - f(MRT), 0 )  — 可避免的超额死亡
  4. PAF:    sum( delta_i * popnum_i ) / sum( f(T_i) * popnum_i )
  5. CI:     500 bootstrap resamples, percentile method

  与旧版 (RR-based) 的区别:
  - 旧: AF_i = (RR_i - 1) / RR_i, RR_i = f(T_i) / f(MMT)
  - 新: 使用人口数直接加权 (popnum), 而非简单地求均值
  - 分母为所有观测的加权预测死亡率总和 (含冷热两端)
""")
