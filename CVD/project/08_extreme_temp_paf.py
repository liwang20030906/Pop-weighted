"""
Extreme Temperature Population-Attributable Fraction — UWT vs PWT
Uses FIXED REFERENCE (not MRT) as counterfactual:
  Cold indicators: reference = sample MAX (mildest cold extreme)
  Hot indicators:  reference = sample MIN (mildest hot extreme)
  Count indicators: reference = 0 days
PAF = sum(max(pred_obs - pred_ref, 0) * popnum) / sum(pred_obs * popnum)
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
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.read_csv(path)


# =========================
# 1. Data
# =========================
uwt_heat = read_csv_auto(os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"))
uwt_cold = read_csv_auto(os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"))
pwt_data = read_csv_auto(os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"))
pop_data = read_csv_auto(os.path.join(GBD_DIR, "1.1GBD_Pop_location_2000-2020_iso_a3.csv"))

pop_df = pop_data[["iso_a3", "year", "val"]].rename(columns={"val": "popnum"})
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
# 2. Indicator definitions
#   (group, label, uwt_col, pwt_col, type, ref_desc)
#   type: "cold" = harm below ref (ref=MAX), "hot" = harm above ref (ref=MIN)
#         "cold_count" = cold spell days (ref=0), "hot_count" = heatwave days (ref=0)
# =========================
INDICATORS = [
    # Cold percentiles
    ("Cold extreme (percentile)", "P1",   "P1",   "mean_P1",   "cold"),
    ("Cold extreme (percentile)", "P3",   "P3",   "mean_P3",   "cold"),
    ("Cold extreme (percentile)", "P5",   "P5",   "mean_P5",   "cold"),
    ("Cold extreme (percentile)", "P10",  "P10",  "mean_P10",  "cold"),
    # Hot percentiles
    ("Hot extreme (percentile)",  "P90",  "P90",  "mean_P90",  "hot"),
    ("Hot extreme (percentile)",  "P93",  "P93",  "mean_P93",  "hot"),
    ("Hot extreme (percentile)",  "P95",  "P95",  "mean_P95",  "hot"),
    ("Hot extreme (percentile)",  "P97",  "P97",  "mean_P97",  "hot"),
    ("Hot extreme (percentile)",  "P98",  "P98",  "mean_P98",  "hot"),
    ("Hot extreme (percentile)",  "P99",  "P99",  "mean_P99",  "hot"),
    # Cold spell days (count)
    ("Cold spell days",  "CS_P3_2d",  "Coldspell_P3_2d",   "mean_Coldspell_P3_2d",   "cold_count"),
    ("Cold spell days",  "CS_P5_2d",  "Coldspell_P5_2d",   "mean_Coldspell_P5_2d",   "cold_count"),
    ("Cold spell days",  "CS_P10_2d", "Coldspell_P10_2d",  "mean_ColdSpell_P10_2d",  "cold_count"),
    # Heatwave days (count)
    ("Heatwave days",    "HW_P90_2d", "Heatwave_P90_2d",  "mean_Heatwave_P90_2d",  "hot_count"),
    ("Heatwave days",    "HW_P93_2d", "Heatwave_P93_2d",  "mean_Heatwave_P93_2d",  "hot_count"),
    ("Heatwave days",    "HW_P97_2d", "Heatwave_P97_2d",  "mean_Heatwave_P97_2d",  "hot_count"),
]


# =========================
# 3. PAF with FIXED reference
# =========================
def extreme_paf(df, extreme_col, mean_col, ind_type, n_boot=200, seed=2025):
    """
    Compute PAF using a fixed reference counterfactual instead of MRT.

    ind_type:
      "cold"       -> reference = sample MAX (warmest extreme cold)
      "hot"        -> reference = sample MIN (coolest extreme hot)
      "cold_count" -> reference = 0 days (no cold spells)
      "hot_count"  -> reference = 0 days (no heatwaves)

    PAF = sum(max(pred_obs - pred_ref, 0) * popnum) / sum(pred_obs * popnum)
    """
    np.random.seed(seed)
    sf = extreme_col.replace(".", "_").replace(" ", "_")
    mean_sf = mean_col.replace(".", "_").replace(" ", "_")

    cols_need = ["val", extreme_col, mean_col, "location", "year", "popnum"]
    cols_avail = [c for c in cols_need if c in df.columns]
    mdf = df[cols_avail].dropna().copy()
    mdf = mdf.rename(columns={extreme_col: sf})
    if mean_col != mean_sf:
        mdf = mdf.rename(columns={mean_col: mean_sf})

    if mdf["location"].nunique() < 20 or len(mdf) < 100:
        return None

    # Fit with mean temp as control
    formula = (
        "Q('val') ~ bs(Q('" + sf + "'), df=4, degree=3) + "
        "bs(Q('" + mean_sf + "'), df=4, degree=3) + "
        "C(Q('location')) + bs(Q('year'), df=3, degree=3)"
    )
    try:
        model = smf.ols(formula, data=mdf).fit()
    except Exception as e:
        return {"error": str(e)[:200]}

    # Reduced model for F-test
    formula_red = (
        "Q('val') ~ bs(Q('" + mean_sf + "'), df=4, degree=3) + "
        "C(Q('location')) + bs(Q('year'), df=3, degree=3)"
    )
    model_red = smf.ols(formula_red, data=mdf).fit()
    anova = sm.stats.anova_lm(model_red, model)
    f_ext = anova.iloc[1]["F"] if len(anova) >= 2 else np.nan
    p_ext = anova.iloc[1]["Pr(>F)"] if len(anova) >= 2 else np.nan

    # Set reference value (must stay within [x_min, x_max] for B-spline validity)
    x_vals = mdf[sf].values
    x_min_obs, x_max_obs = x_vals.min(), x_vals.max()
    if ind_type == "cold":
        ref_val = x_max_obs  # warmest cold extreme (least harm)
    elif ind_type == "hot":
        ref_val = x_min_obs  # coolest hot extreme (least harm)
    elif ind_type in ("cold_count", "hot_count"):
        ref_val = max(0.0, x_min_obs)  # 0 days, but clip to observed min for bspline
    else:
        return None

    # Predictions
    df_in = mdf.reset_index(drop=True).copy()
    df_in["pred"] = model.predict(df_in)

    # Counterfactual: set extreme to reference, keep mean temp at observed
    df_ref = df_in[["location", "year", mean_sf]].copy()
    df_ref[sf] = ref_val
    try:
        df_in["pred_ref"] = model.predict(df_ref)
    except Exception:
        return {"error": f"prediction failed at ref={ref_val:.2f}, range=[{x_min_obs:.2f},{x_max_obs:.2f}]"}

    # PAF
    delta = np.maximum(df_in["pred"] - df_in["pred_ref"], 0)
    denominator = np.sum(df_in["pred"] * df_in["popnum"])
    PAF = np.sum(delta * df_in["popnum"]) / denominator

    # Count how many obs are on the "harm" side of reference
    if ind_type in ("cold", "cold_count"):
        n_harm = (x_vals < ref_val).sum()
    else:
        n_harm = (x_vals > ref_val).sum()

    # Bootstrap
    n = len(df_in)
    n_eff = min(n_boot, 200)
    boot = np.zeros(n_eff)
    for b in range(n_eff):
        idx = np.random.choice(n, size=n, replace=True)
        samp = df_in.iloc[idx].reset_index(drop=True).copy()
        samp["pred"] = model.predict(samp)
        sr = samp[["location", "year", mean_sf]].copy()
        sr[sf] = ref_val
        samp["pred_ref"] = model.predict(sr)
        db = np.maximum(samp["pred"] - samp["pred_ref"], 0)
        denom_b = np.sum(samp["pred"] * samp["popnum"])
        boot[b] = np.sum(db * samp["popnum"]) / denom_b

    LCI, UCI = np.percentile(boot, [2.5, 97.5])

    return {
        "PAF": PAF, "LCI": LCI, "UCI": UCI,
        "ref_val": ref_val,
        "n_obs": n, "n_locations": mdf["location"].nunique(),
        "n_harm": int(n_harm),
        "rsquared": model.rsquared,
        "f_extreme": f_ext, "p_extreme": p_ext,
        "mean_extreme": x_vals.mean(),
    }


# =========================
# 4. Run
# =========================
print("=" * 95)
print("  EXTREME TEMPERATURE PAF — Fixed Reference (not MRT)")
print("  Counterfactual: all observations at the mildest extreme level")
print("=" * 95)

print(f"\n  {'Indicator':<14s} {'Type':>10s} {'Ref':>12s} "
      f"{'UWT PAF':>14s} {'PWT PAF':>14s} {'Diff':>10s} "
      f"{'p_UWT':>8s} {'p_PWT':>8s} {'Direction':>12s}")
print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

results = []
for group, label, col_u, col_p, itype in INDICATORS:
    if col_u not in df_uwt.columns or col_p not in df_pwt.columns:
        continue

    r_u = extreme_paf(df_uwt, col_u, "Mean_Temp", itype)
    r_p = extreme_paf(df_pwt, col_p, "mean_Mean", itype)

    if r_u is None or r_p is None or "error" in r_u or "error" in r_p:
        err = r_u.get("error", "") if r_u else "" or r_p.get("error", "") if r_p else ""
        print(f"  {label:<14s} FAIL: {err[:60]}")
        continue

    diff = r_u["PAF"] - r_p["PAF"]
    sig_u = "***" if r_u["p_extreme"] < 0.001 else "**" if r_u["p_extreme"] < 0.01 else "*" if r_u["p_extreme"] < 0.05 else ""
    sig_p = "***" if r_p["p_extreme"] < 0.001 else "**" if r_p["p_extreme"] < 0.01 else "*" if r_p["p_extreme"] < 0.05 else ""

    direction = ""
    if itype in ("cold", "cold_count"):
        direction = "(U>P)" if diff > 0 else "(P>U)"
    else:
        direction = "(P>U)" if diff < 0 else "(U>P)"

    print(f"  {label:<14s} {itype:>10s} {r_u['ref_val']:>12.2f} "
          f"{r_u['PAF']:>12.4%} ({r_u['LCI']:.4%}-{r_u['UCI']:.4%}) "
          f"{r_p['PAF']:>12.4%} ({r_p['LCI']:.4%}-{r_p['UCI']:.4%}) "
          f"{diff:>+10.4%} "
          f"{r_u['p_extreme']:>8.4f}{sig_u} {r_p['p_extreme']:>8.4f}{sig_p} {direction:>12s}")

    results.append({
        "group": group, "label": label, "type": itype,
        "u_paf": r_u["PAF"], "p_paf": r_p["PAF"],
        "u_lci": r_u["LCI"], "p_lci": r_p["LCI"],
        "u_uci": r_u["UCI"], "p_uci": r_p["UCI"],
        "diff": diff, "direction": direction,
        "u_F": r_u["f_extreme"], "u_p": r_u["p_extreme"],
        "p_F": r_p["f_extreme"], "p_p": r_p["p_extreme"],
        "u_ref": r_u["ref_val"], "p_ref": r_p["ref_val"],
        "u_mean": r_u["mean_extreme"], "p_mean": r_p["mean_extreme"],
    })

# =========================
# 5. Summary
# =========================
print(f"\n{'='*95}")
print(f"  SUMMARY: Does PWT reduce cold-extreme PAF and increase hot-extreme PAF?")
print(f"{'='*95}")

cold_results = [r for r in results if r["type"] in ("cold", "cold_count")]
hot_results = [r for r in results if r["type"] in ("hot", "hot_count")]

n_cold_drop = sum(1 for r in cold_results if r["diff"] > 0)
n_hot_rise = sum(1 for r in hot_results if r["diff"] < 0)

print(f"\n  Cold extreme indicators ({len(cold_results)} total):")
print(f"    Cold PAF DROPS under PWT: {n_cold_drop}/{len(cold_results)}")
for r in cold_results:
    arrow = "v" if r["diff"] > 0 else "^"
    print(f"    {r['label']:<8s} UWT={r['u_paf']:.4%} -> PWT={r['p_paf']:.4%}  "
          f"(diff={r['diff']:+.4%}) {arrow}")

print(f"\n  Hot extreme indicators ({len(hot_results)} total):")
print(f"    Hot PAF RISES under PWT: {n_hot_rise}/{len(hot_results)}")
for r in hot_results:
    arrow = "^" if r["diff"] < 0 else "v"
    print(f"    {r['label']:<8s} UWT={r['u_paf']:.4%} -> PWT={r['p_paf']:.4%}  "
          f"(diff={r['diff']:+.4%}) {arrow}")

print(f"""\n
  Reference definitions:
    Cold percentile: ref = sample MAX (= warmest cold extreme observed)
    Hot percentile:  ref = sample MIN (= coolest hot extreme observed)
    Count days:      ref = 0 days (no extreme events)

  PAF interpretation:
    "What fraction of predicted CVD mortality is attributable to
     extreme temperatures being more severe than the mildest-observed level?"

  Model: val ~ bs(extreme, df=4) + bs(T_mean, df=4) + C(location) + bs(year, df=3)
""")
