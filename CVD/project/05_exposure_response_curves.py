"""
UWT vs PWT 暴露-响应曲线 + MMT — 纯文本终端输出
"""
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

uwt_heat = read_csv_auto(os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"))
uwt_cold = read_csv_auto(os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"))
pwt_data = read_csv_auto(os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"))

id_cols = ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]
existing_id = [c for c in id_cols if c in uwt_heat.columns]
high_temp = list(set(uwt_heat.columns) - set(existing_id))
cold_keep = [c for c in uwt_cold.columns if c in existing_id or c not in high_temp]
merged_uwt = pd.merge(
    uwt_heat[existing_id + high_temp], uwt_cold[cold_keep],
    on=existing_id, how="outer",
)
merged_uwt["year"] = pd.to_numeric(merged_uwt["year"], errors="coerce")
pwt_id = [c for c in existing_id if c in pwt_data.columns]
pwt_data["year"] = pd.to_numeric(pwt_data["year"], errors="coerce")

model_cols = existing_id + ["val", "upper", "lower"]
cold_extra = [c for c in uwt_cold.columns if c not in high_temp and c not in existing_id]

# 去重 — val 可能在 model_cols 和 high_temp 中重复
_uwt_cols = model_cols + [c for c in high_temp if c not in model_cols] + [c for c in cold_extra if c not in model_cols]
_uwt_cols_final = []
for c in _uwt_cols:
    if c not in _uwt_cols_final:
        _uwt_cols_final.append(c)

df_uwt = merged_uwt[_uwt_cols_final].copy()

pwt_temp_cols = [c for c in pwt_data.columns if c.startswith("mean_")]
_pwt_cols = model_cols + pwt_temp_cols
_pwt_cols_final = []
for c in _pwt_cols:
    if c not in _pwt_cols_final:
        _pwt_cols_final.append(c)

df_pwt = pd.merge(
    merged_uwt[model_cols],
    pwt_data[pwt_id + pwt_temp_cols],
    on=pwt_id, how="inner",
)
# 同样去重 val
existing = set()
_pwt_final = []
for c in df_pwt.columns:
    if c not in existing:
        existing.add(c)
        _pwt_final.append(c)
df_pwt = df_pwt[_pwt_final]


MATCHED_PAIRS = [
    # (类别, UWT字段, PWT字段, 简称)
    ("基础温度", "Mean_Temp", "mean_Mean", "年均温"),
    ("基础温度", "Max_Temp", "mean_Max", "年最高温"),
    ("基础温度", "Min_Temp", "mean_Min", "年最低温"),
    ("极端热", "P90", "mean_P90", "P90"),
    ("极端热", "P93", "mean_P93", "P93"),
    ("极端热", "P95", "mean_P95", "P95"),
    ("极端热", "P97", "mean_P97", "P97"),
    ("极端热", "P98", "mean_P98", "P98"),
    ("极端热", "P99", "mean_P99", "P99"),
    ("极端冷", "P1", "mean_P1", "P1"),
    ("极端冷", "P3", "mean_P3", "P3"),
    ("极端冷", "P5", "mean_P5", "P5"),
    ("极端冷", "P10", "mean_P10", "P10"),
    ("热暴露", "Heatwave_P90_2d", "mean_Heatwave_P90_2d", "HW_P90_2d"),
    ("热暴露", "Heatwave_P93_2d", "mean_Heatwave_P93_2d", "HW_P93_2d"),
    ("热暴露", "Heatwave_P97_2d", "mean_Heatwave_P97_2d", "HW_P97_2d"),
    ("冷暴露", "Coldspell_P3_2d", "mean_Coldspell_P3_2d", "CS_P3_2d"),
    ("冷暴露", "Coldspell_P5_2d", "mean_Coldspell_P5_2d", "CS_P5_2d"),
    ("冷暴露", "Coldspell_P10_2d", "mean_ColdSpell_P10_2d", "CS_P10_2d"),
]


def safe_col(col):
    return col.replace(".", "_").replace(" ", "_")


def fit_and_predict(df, col, n_points=100):
    sf = safe_col(col)
    mdf = df[["val", col, "location", "year"]].dropna().rename(columns={col: sf}).copy()
    if len(mdf) < 100:
        return None, None, None, None

    formula = (
        f"Q('val') ~ bs(Q('{sf}'), df=4, degree=3) + "
        f"C(Q('location')) + bs(Q('year'), df=3, degree=3)"
    )
    formula0 = f"Q('val') ~ C(Q('location')) + bs(Q('year'), df=3, degree=3)"

    try:
        model = smf.ols(formula, data=mdf).fit()
        model0 = smf.ols(formula0, data=mdf).fit()
        anova = sm.stats.anova_lm(model0, model)
        p_val = anova.iloc[1]["Pr(>F)"] if len(anova) >= 2 else None
        f_val = anova.iloc[1]["F"] if len(anova) >= 2 else None
    except Exception as e:
        return None, None, None, str(e)

    x_min, x_max = mdf[sf].min(), mdf[sf].max()
    x_grid = np.linspace(x_min, x_max, n_points)
    loc = mdf["location"].mode().values[0]
    yr = mdf["year"].mean()
    base_row = mdf.iloc[[0]].copy()

    preds = []
    for x in x_grid:
        r = base_row.copy()
        r[sf] = x
        r["location"] = loc
        r["year"] = yr
        try:
            preds.append(model.predict(r).values[0])
        except Exception:
            preds.append(np.nan)

    preds = np.array(preds)
    mmt_idx = np.nanargmin(preds)
    mmt_val = x_grid[mmt_idx]
    mmt_pred = preds[mmt_idx]

    return x_grid, preds, (mmt_val, mmt_pred), (p_val, f_val)


def ascii_plot(ax_uwt, ay_uwt, ax_pwt, ay_pwt, mmt_u, mmt_p, label, width=60, height=15):
    """用 ASCII 字符绘制双曲线对比图"""
    all_x = np.concatenate([ax_uwt, ax_pwt])
    all_y = np.concatenate([ay_uwt, ay_pwt])
    valid = ~np.isnan(all_y)
    x_all, y_all = all_x[valid], all_y[valid]

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    y_pad = (y_max - y_min) * 0.1
    y_min -= y_pad
    y_max += y_pad

    # 创建画布
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    def x2col(x):
        return int((x - x_min) / (x_max - x_min) * (width - 1))

    def y2row(y):
        return int((y_max - y) / (y_max - y_min) * (height - 1))

    # 画 UWT 线 (-)
    for i in range(len(ax_uwt) - 1):
        r1, c1 = y2row(ay_uwt[i]), x2col(ax_uwt[i])
        r2, c2 = y2row(ay_uwt[i + 1]), x2col(ax_uwt[i + 1])
        r1, c1 = max(0, min(height - 1, r1)), max(0, min(width - 1, c1))
        r2, c2 = max(0, min(height - 1, r2)), max(0, min(width - 1, c2))
        steps = max(abs(r2 - r1), abs(c2 - c1), 1)
        for s in range(steps + 1):
            r = int(r1 + (r2 - r1) * s / steps)
            c = int(c1 + (c2 - c1) * s / steps)
            r, c = max(0, min(height - 1, r)), max(0, min(width - 1, c))
            canvas[r][c] = "b"

    # 画 PWT 线 (O)
    for i in range(len(ax_pwt) - 1):
        r1, c1 = y2row(ay_pwt[i]), x2col(ax_pwt[i])
        r2, c2 = y2row(ay_pwt[i + 1]), x2col(ax_pwt[i + 1])
        r1, c1 = max(0, min(height - 1, r1)), max(0, min(width - 1, c1))
        r2, c2 = max(0, min(height - 1, r2)), max(0, min(width - 1, c2))
        steps = max(abs(r2 - r1), abs(c2 - c1), 1)
        for s in range(steps + 1):
            r = int(r1 + (r2 - r1) * s / steps)
            c = int(c1 + (c2 - c1) * s / steps)
            r, c = max(0, min(height - 1, r)), max(0, min(width - 1, c))
            canvas[r][c] = "O" if canvas[r][c] == " " else "@"

    # 画 MMT 线
    for (mmt, ch) in [(mmt_u, "|"), (mmt_p, ":")]:
        if mmt is None:
            continue
        col = x2col(mmt)
        for r in range(height):
            if canvas[r][col] == " ":
                canvas[r][col] = ch

    # 渲染
    lines = ["" .join(row) for row in canvas]

    # 轴标签
    x_label = f"temp: {x_min:.1f} ~ {x_max:.1f}"
    y_label = f"死亡率: {y_min:.1f} ~ {y_max:.1f}"
    mmt_line_u = f"MMT_UWT={mmt_u:.1f}" if mmt_u else ""
    mmt_line_p = f"MMT_PWT={mmt_p:.1f}" if mmt_p else ""

    return f"""
{'='*(width+2)}
{label:^{width+2}}
{'='*(width+2)}
{y_label}
{chr(10).join(lines)}
{x_label:^{width}}
  b = UWT  O = PWT  | = MMT_UWT  : = MMT_PWT  @ = 重合
{mmt_line_u:^{width+2}}
{mmt_line_p:^{width+2}}
"""


# =========================
# 主程序
# =========================
print()
print("=" * 70)
print("  UWT vs PWT 暴露-响应曲线 (ASCII) — 带 MMT 标注")
print("=" * 70)

current_cat = ""
for cat, uwt_col, pwt_col, label in MATCHED_PAIRS:
    if cat != current_cat:
        current_cat = cat
        print(f"\n{'█' * 70}")
        print(f"  [{cat}]")
        print(f"{'█' * 70}")

    # FIT
    xu, yu, mmt_u, info_u = fit_and_predict(df_uwt, uwt_col)
    xp, yp, mmt_p, info_p = fit_and_predict(df_pwt, pwt_col)

    if xu is None or xp is None:
        print(f"\n  [{label}] 模型拟合失败")
        continue

    pu, fu = info_u if info_u else (None, None)
    pp, fp = info_p if info_p else (None, None)

    # ASCII 图
    plot = ascii_plot(
        xu, yu, xp, yp,
        mmt_u[0] if mmt_u else None,
        mmt_p[0] if mmt_p else None,
        f"{label}  ({uwt_col} vs {pwt_col})",
        width=70, height=18,
    )
    print(plot)

    # 数值表格
    print(f"  {'':>12} {'UWT':>15} {'PWT':>15} {'差异':>15}")
    print(f"  {'MMT':>12} {mmt_u[0]:>15.2f} {mmt_p[0]:>15.2f} {mmt_p[0]-mmt_u[0]:>+15.2f}")
    print(f"  {'min死亡率':>12} {mmt_u[1]:>15.2f} {mmt_p[1]:>15.2f} {mmt_p[1]-mmt_u[1]:>+15.2f}")
    print(f"  {'p值':>12} {pu:>15.4f} {pp:>15.4f}")
    print(f"  {'F值':>12} {fu:>15.2f} {fp:>15.2f}")
    print(f"  {'均值':>12} {np.nanmean(yu):>15.2f} {np.nanmean(yp):>15.2f}")

print()
print("=" * 70)
print("  图例: b = UWT(非加权)  O = PWT(人口加权)  | = MMT")
print("=" * 70)
