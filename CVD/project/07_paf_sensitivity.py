"""
年均温 PAF 敏感性分析 —— 排除 PWT-UWT 差异最大的国家后重新计算
方法: 人口加权 PAF = sum(max(pred - pred(MRT), 0) * popnum) / sum(pred * popnum)
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

pop_df = pop_data[["iso_a3", "year", "val"]].copy()
pop_df = pop_df.rename(columns={"val": "popnum"})
pop_df["year"] = pd.to_numeric(pop_df["year"], errors="coerce")

eid = [c for c in ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]
       if c in uwt_heat.columns]
ht = list(set(uwt_heat.columns) - set(eid))
ck = [c for c in uwt_cold.columns if c in eid or c not in ht]
muwt = pd.merge(uwt_heat[eid + ht], uwt_cold[ck], on=eid, how="outer")
muwt["year"] = pd.to_numeric(muwt["year"], errors="coerce")
pid = [c for c in eid if c in pwt_data.columns]
pwt_data["year"] = pd.to_numeric(pwt_data["year"], errors="coerce")
pm = [c for c in pwt_data.columns if c.startswith("mean_")]

cold_extra = [c for c in uwt_cold.columns if c not in ht and c not in eid]
_uwt_cols = eid + ["val"] + [c for c in ht if c not in eid] + [c for c in cold_extra if c not in eid]
df_uwt = muwt[list(dict.fromkeys(_uwt_cols))].copy()
df_uwt = pd.merge(df_uwt, pop_df, on=["iso_a3", "year"], how="inner")

df_pwt_raw = pd.merge(muwt[eid + ["val"]], pwt_data[pid + pm], on=pid, how="inner")
df_pwt = df_pwt_raw[list(dict.fromkeys(df_pwt_raw.columns))].copy()
df_pwt = pd.merge(df_pwt, pop_df, on=["iso_a3", "year"], how="inner")


# =========================
# 2. 各国 PWT-UWT 温差
# =========================
merged_temp = pd.merge(
    df_uwt[["iso_a3", "location", "year", "Mean_Temp"]],
    df_pwt[["iso_a3", "location", "year", "mean_Mean"]],
    on=["iso_a3", "location", "year"], how="inner"
)
merged_temp["delta_T"] = merged_temp["mean_Mean"] - merged_temp["Mean_Temp"]

country_delta = (
    merged_temp.groupby("location")
    .agg(mean_delta=("delta_T", "mean"),
         mean_UWT=("Mean_Temp", "mean"),
         mean_PWT=("mean_Mean", "mean"),
         n_years=("year", "nunique"))
    .sort_values("mean_delta", ascending=False)
    .reset_index()
)

print("=" * 80)
print("  各国 PWT-UWT 温度差异 (mean_Mean - Mean_Temp)")
print("=" * 80)
print(f"\n  {'国家':<30s} {'UWT均值':>8s} {'PWT均值':>8s} {'Δ(K)':>8s} {'年份数':>6s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

for _, r in country_delta.iterrows():
    marker = " <<<" if abs(r["mean_delta"]) > 2.0 else ""
    try:
        loc_name = str(r["location"])[:28]
        print("  {:<30s} {:>8.2f} {:>8.2f} {:>+8.2f} {:>6}{}".format(
            loc_name, r["mean_UWT"], r["mean_PWT"], r["mean_delta"], r["n_years"], marker))
    except UnicodeEncodeError:
        continue


# =========================
# 3. 人口加权 PAF 函数
# =========================
def fit_and_paf(df, temp_col, exclude_locations=None, n_boot=500, seed=2025):
    """Fit FE-GAM and compute population-weighted PAF with bootstrap CI"""
    sf = temp_col.replace(".", "_").replace(" ", "_")
    mdf = df[["val", temp_col, "location", "year", "popnum"]].dropna().rename(
        columns={temp_col: sf}).copy()

    if exclude_locations:
        mdf = mdf[~mdf["location"].isin(exclude_locations)]

    n_loc = mdf["location"].nunique()
    if n_loc < 10:
        return None

    formula = (
        "Q('val') ~ bs(Q('" + sf + "'), df=4, degree=3) + "
        "C(Q('location')) + bs(Q('year'), df=3, degree=3)"
    )
    model = smf.ols(formula, data=mdf).fit()

    # MRT grid search
    np.random.seed(seed)
    df_in = mdf.reset_index(drop=True).copy()

    df_in["pred"] = model.predict(df_in)

    x_min, x_max = df_in[sf].min(), df_in[sf].max()
    grid = pd.DataFrame({
        sf: np.linspace(x_min, x_max, 400),
        "location": df_in["location"].values[0],
        "year": df_in["year"].values[0],
    })
    grid["fit"] = model.predict(grid)
    MRT = grid[sf].values[np.argmin(grid["fit"].values)]

    df_ref = df_in[["location", "year"]].copy()
    df_ref[sf] = MRT
    df_in["pred_ref"] = model.predict(df_ref)

    delta = np.maximum(df_in["pred"] - df_in["pred_ref"], 0)
    denominator = np.sum(df_in["pred"] * df_in["popnum"])

    cold_mask = df_in[sf] < MRT
    heat_mask = df_in[sf] > MRT

    PAF_total = np.sum(delta * df_in["popnum"]) / denominator
    PAF_cold = np.sum(delta[cold_mask] * df_in["popnum"][cold_mask]) / denominator
    PAF_heat = np.sum(delta[heat_mask] * df_in["popnum"][heat_mask]) / denominator

    # Bootstrap
    n = len(df_in)
    boot_paf = np.zeros((n_boot, 3))

    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        samp = df_in.iloc[idx].reset_index(drop=True).copy()

        samp["pred"] = model.predict(samp)
        sr = samp[["location", "year"]].copy()
        sr[sf] = MRT
        samp["pred_ref"] = model.predict(sr)

        db = np.maximum(samp["pred"] - samp["pred_ref"], 0)
        denom_b = np.sum(samp["pred"] * samp["popnum"])

        boot_paf[b, 0] = np.sum(db * samp["popnum"]) / denom_b
        cb = samp[sf] < MRT
        boot_paf[b, 1] = np.sum(db[cb] * samp["popnum"][cb]) / denom_b
        hb = samp[sf] > MRT
        boot_paf[b, 2] = np.sum(db[hb] * samp["popnum"][hb]) / denom_b

    LCI = np.percentile(boot_paf, 2.5, axis=0)
    UCI = np.percentile(boot_paf, 97.5, axis=0)

    # Per-country AF (simple, no bootstrap for individual countries)
    country_af = {}
    for loc in mdf["location"].unique():
        mask = df_in["location"] == loc
        c_delta = delta[mask]
        c_pop = df_in["popnum"][mask]
        c_cold = cold_mask[mask]
        c_heat = heat_mask[mask]
        country_af[loc] = {
            "af_cold": np.sum(c_delta[c_cold] * c_pop[c_cold]) / np.sum(df_in["pred"][mask] * c_pop),
            "af_heat": np.sum(c_delta[c_heat] * c_pop[c_heat]) / np.sum(df_in["pred"][mask] * c_pop),
            "af_total": np.sum(c_delta * c_pop) / np.sum(df_in["pred"][mask] * c_pop),
        }

    return {
        "MRT": MRT,
        "PAF_cold": PAF_cold, "PAF_heat": PAF_heat, "PAF_total": PAF_total,
        "PAF_cold_LCI": LCI[1], "PAF_heat_LCI": LCI[2], "PAF_total_LCI": LCI[0],
        "PAF_cold_UCI": UCI[1], "PAF_heat_UCI": UCI[2], "PAF_total_UCI": UCI[0],
        "n_obs": n, "n_locations": n_loc,
        "rsquared": model.rsquared,
        "country_af": country_af,
    }


# =========================
# 4. 逐步排除分析
# =========================
country_abs_delta = country_delta.copy()
country_abs_delta["abs_delta"] = country_abs_delta["mean_delta"].abs()
country_abs_delta = country_abs_delta.sort_values("abs_delta", ascending=False)

exclusion_steps = [0, 1, 3, 5, 10, 15, 20, 30]

print(f"\n{'='*80}")
print(f"  逐步排除分析: 按 |ΔT| 降序移除国家后重新计算 PAF")
print(f"{'='*80}")
print(f"\n  {'排除':>5s} {'UWT_冷':>8s} {'UWT_热':>8s} {'PWT_冷':>8s} {'PWT_热':>8s} "
      f"{'U-PΔ冷':>8s} {'U-PΔ热':>8s} {'N_loc':>5s}")
print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

history = []
for n_exclude in exclusion_steps:
    excluded = set() if n_exclude == 0 else set(country_abs_delta.head(n_exclude)["location"])

    r_u = fit_and_paf(df_uwt, "Mean_Temp", excluded)
    r_p = fit_and_paf(df_pwt, "mean_Mean", excluded)

    if r_u is None or r_p is None:
        continue

    cold_diff = r_u["PAF_cold"] - r_p["PAF_cold"]
    heat_diff = r_u["PAF_heat"] - r_p["PAF_heat"]

    print(f"  {n_exclude:>5} {r_u['PAF_cold']:>8.4%} {r_u['PAF_heat']:>8.4%} "
          f"{r_p['PAF_cold']:>8.4%} {r_p['PAF_heat']:>8.4%} "
          f"{cold_diff:>+8.4%} {heat_diff:>+8.4%} {r_p['n_locations']:>5}")

    history.append({
        "n_exclude": n_exclude,
        "excluded_countries": sorted(excluded),
        "uwt": r_u, "pwt": r_p,
        "cold_diff": cold_diff, "heat_diff": heat_diff,
    })


# =========================
# 5. 排除10国后各国详情
# =========================
print(f"\n{'='*80}")
print(f"  排除前10个 |ΔT| 最大国家后的各国 PAF 贡献")
top10 = sorted(country_abs_delta.head(10)["location"].tolist())
print(f"  排除: {top10}")
print(f"{'='*80}")

if len(history) >= 4:
    h10 = history[4]
    print(f"\n  {'国家':<30s} {'UWT冷':>8s} {'PWT冷':>8s} {'UWT热':>8s} {'PWT热':>8s} "
          f"{'UWT总':>8s} {'PWT总':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    all_locs = set(h10["uwt"]["country_af"].keys()) | set(h10["pwt"]["country_af"].keys())
    for loc in sorted(all_locs):
        u = h10["uwt"]["country_af"].get(loc, {"af_cold": 0, "af_heat": 0, "af_total": 0})
        p = h10["pwt"]["country_af"].get(loc, {"af_cold": 0, "af_heat": 0, "af_total": 0})
        print(f"  {loc:<30s} {u['af_cold']:>8.4f} {p['af_cold']:>8.4f} "
              f"{u['af_heat']:>8.4f} {p['af_heat']:>8.4f} "
              f"{u['af_total']:>8.4f} {p['af_total']:>8.4f}")


# =========================
# 6. 摘要
# =========================
print(f"\n{'='*80}")
print(f"  敏感性分析摘要")
print(f"{'='*80}")

for h in history:
    label = f"排除{h['n_exclude']}国"
    print(f"  {label:<15s} UWT冷={h['uwt']['PAF_cold']:.4%} PWT冷={h['pwt']['PAF_cold']:.4%} "
          f"Δ冷={h['cold_diff']:+.4%} Δ热={h['heat_diff']:+.4%}")

print(f"""
  解释:
  - 排除温差大的国家后, UWT和PWT的冷效应PAF差距是否缩小?
  - 若缩小 -> 温差大的国家是UWT/PWT差异的主要驱动者
  - 公式: PAF = sum(max(pred - pred(MRT), 0) * popnum) / sum(pred * popnum)
  - CI: 500 bootstrap resamples, percentile method
""")
