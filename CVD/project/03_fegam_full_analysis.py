import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import argparse
import warnings
import re
import os
from patsy import bs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FE-GAM: 温度指标与脑卒中死亡率的固定效应广义加性模型分析"
    )
    parser.add_argument(
        "--uwt-heat", "-uh",
        default=os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"),
        help="UWT 高温文件路径"
    )
    parser.add_argument(
        "--uwt-cold", "-uc",
        default=os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"),
        help="UWT 低温文件路径"
    )
    parser.add_argument(
        "--pwt", "-p",
        default=os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"),
        help="PWT 人口加权文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(BASE_DIR, "FE_GAM_results.xlsx"),
        help="输出 Excel 路径"
    )
    parser.add_argument(
        "--spline-df",
        type=int,
        default=4,
        help="温度指标样条自由度 (默认4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    return parser.parse_args()


def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def prepare_data(uwt_heat_path, uwt_cold_path, pwt_path):
    """
    合并 UWT 和 PWT 数据，构建统一的分析数据框。
    使用 PWT 的温度指标作为暴露变量，UWT 的 val 作为健康结局。
    """
    print("读取数据...")
    uwt_heat = read_csv_auto(uwt_heat_path)
    uwt_cold = read_csv_auto(uwt_cold_path)
    pwt = read_csv_auto(pwt_path)

    print(f"  UWT High: {uwt_heat.shape}")
    print(f"  UWT Cold: {uwt_cold.shape}")
    print(f"  PWT:      {pwt.shape}")

    # 标准化列名（统一大小写、去除前缀）
    pwt_renamed = pwt.copy()

    # 将 PWT 中的 mean_* 字段映射为与 UWT 字段同义的名称，方便统一处理
    rename_map = {}
    for col in pwt.columns:
        if col.startswith("mean_"):
            rename_map[col] = col.replace("mean_", "")
    pwt_renamed = pwt_renamed.rename(columns=rename_map)

    # 合并 UWT high + cold
    id_cols_uwt = ["iso_a3", "year", "location", "sex", "age", "cause", "measure", "metric"]

    # 找到实际存在的标识列
    existing_id = [c for c in id_cols_uwt if c in uwt_heat.columns]

    # UWT high 和 cold 按标识列合并
    # cold 中去掉 high 中已有的温度列
    high_temp_cols = set(uwt_heat.columns) - set(existing_id)
    cold_cols_to_keep = [c for c in uwt_cold.columns if c in existing_id or c not in high_temp_cols]

    merged_uwt = pd.merge(
        uwt_heat[existing_id + list(high_temp_cols)],
        uwt_cold[cold_cols_to_keep],
        on=existing_id,
        how="outer"
    )

    # 限制到关键列用于建模
    model_cols = existing_id + ["val", "upper", "lower", "SDI", "Zone", "Continent"]

    # 从 PWT 获取温度指标
    pwt_id_cols = [c for c in existing_id if c in pwt.columns]
    pwt_temp_cols = [c for c in pwt.columns if c.startswith("mean_")]

    # 合并 PWT 温度 + UWT 健康结局
    merged_uwt_key = merged_uwt[model_cols].copy()
    pwt_key = pwt[pwt_id_cols + pwt_temp_cols].copy()

    # 确保 year 类型一致
    merged_uwt_key["year"] = pd.to_numeric(merged_uwt_key["year"], errors="coerce")
    pwt_key["year"] = pd.to_numeric(pwt_key["year"], errors="coerce")

    merged = pd.merge(
        merged_uwt_key,
        pwt_key,
        on=pwt_id_cols,
        how="inner"
    )

    print(f"\n合并后数据: {merged.shape}")
    print(f"  location 数: {merged['location'].nunique()}")
    print(f"  年份范围: {int(merged['year'].min())} - {int(merged['year'].max())}")

    return merged


def classify_indicators(df):
    """
    将 PWT 温度指标字段分类为：年均温、年最高温、年最低温、极端热、极端冷。
    """
    # PWT 字段都以 mean_ 开头
    temp_cols = [c for c in df.columns if c.startswith("Mean_")
                 or (c.startswith("mean_") and c not in ["mean_Mean", "mean_Max", "mean_Min"])]

    # 重新分类所有温度字段
    all_temp = [c for c in df.columns if c.startswith("mean_")]

    categories = {
        "基础温度_年均温": [],
        "基础温度_年最高温": [],
        "基础温度_年最低温": [],
        "极端热": [],
        "极端冷": [],
        "热暴露": [],
        "冷暴露": [],
    }

    for col in all_temp:
        col_lower = col.lower()

        if "mean_mean" in col_lower or col == "mean_Mean":
            categories["基础温度_年均温"].append(col)
            continue
        if "mean_max" in col_lower or col == "mean_Max":
            categories["基础温度_年最高温"].append(col)
            continue
        if "mean_min" in col_lower or col == "mean_Min":
            categories["基础温度_年最低温"].append(col)
            continue

        # 提取 P 值
        p_match = re.search(r"[pP]_?(\d+(?:\.\d+)?)", col)
        p_val = float(p_match.group(1)) if p_match else None

        is_hot = any(kw in col_lower for kw in ["heatwave", "hot"])
        is_cold = any(kw in col_lower for kw in ["cold", "spell"])

        if is_hot:
            categories["热暴露"].append(col)
        elif is_cold:
            categories["冷暴露"].append(col)
        elif p_val is not None:
            if p_val >= 80:
                categories["极端热"].append(col)
            elif p_val <= 20:
                categories["极端冷"].append(col)

    return categories


def run_fe_gam(df, indicator_col, outcome_col="val", id_col="location", time_col="year", spline_df=4):
    """
    运行固定效应 GAM 模型:
    val ~ bs(indicator, df) + C(location) + bs(year, df=3)
    """
    model_df = df[[outcome_col, indicator_col, id_col, time_col]].dropna().copy()
    # 将 indicator 列名中的点替换为下划线，避免 patsy 解析问题
    safe_indicator = indicator_col.replace(".", "_").replace(" ", "_")
    model_df = model_df.rename(columns={indicator_col: safe_indicator})

    n_locations = model_df[id_col].nunique()
    if n_locations < 2 or len(model_df) < 10:
        return None

    # 构建公式 - 对含特殊字符的列名用 Q() 引用
    formula = (
        f"Q('{outcome_col}') ~ "
        f"bs(Q('{safe_indicator}'), df={spline_df}, degree=3) + "
        f"C(Q('{id_col}')) + "
        f"bs(Q('{time_col}'), df=3, degree=3)"
    )

    try:
        model = smf.ols(formula=formula, data=model_df).fit()
    except Exception as e:
        return {"error": str(e)}

    # 提取指标相关的统计
    # 找到 spline 项对应的参数
    spline_params = [p for p in model.params.index if safe_indicator in p and "bs" in p]

    # 使用模型比较 F-test: 完整模型 vs 不含温度指标的简化模型
    if len(spline_params) > 0:
        try:
            formula_reduced = (
                f"Q('{outcome_col}') ~ "
                f"C(Q('{id_col}')) + "
                f"bs(Q('{time_col}'), df=3, degree=3)"
            )
            model_reduced = smf.ols(formula=formula_reduced, data=model_df).fit()
            anova_result = sm.stats.anova_lm(model_reduced, model)
            if len(anova_result) >= 2:
                f_stat_spline = anova_result.iloc[1]["F"]
                p_val_spline = anova_result.iloc[1]["Pr(>F)"]
            else:
                f_stat_spline = None
                p_val_spline = None
        except Exception:
            f_stat_spline = None
            p_val_spline = None
    else:
        f_stat_spline = None
        p_val_spline = None

    # 计算指标对结局的边际效应（在指示变量均值处的导数近似）
    indicator_mean = model_df[safe_indicator].mean()

    # 计算模型 R-squared
    rsquared = model.rsquared
    rsquared_adj = model.rsquared_adj
    aic = model.aic
    bic = model.bic
    n_obs = int(model.nobs)

    return {
        "indicator": indicator_col,
        "safe_indicator": safe_indicator,
        "n_obs": n_obs,
        "n_locations": n_locations,
        "rsquared": round(rsquared, 4),
        "rsquared_adj": round(rsquared_adj, 4),
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "f_stat_spline": round(f_stat_spline, 4) if f_stat_spline is not None else None,
        "p_val_spline": round(p_val_spline, 6) if p_val_spline is not None else None,
        "indicator_mean": round(indicator_mean, 4),
        "model": model,
        "model_df": model_df,
        "spline_params": spline_params,
    }


def run_indicator_group(df, indicators, group_name, spline_df):
    """
    对一组指标分别运行 FE-GAM。
    """
    results = []

    for i, indicator in enumerate(indicators):
        print(f"  [{i+1}/{len(indicators)}] {indicator} ...", end=" ")
        result = run_fe_gam(df, indicator, spline_df=spline_df)
        if result is None:
            print("数据不足，跳过")
            continue
        if "error" in result:
            print(f"模型失败: {result['error']}")
            continue
        result["group"] = group_name
        results.append(result)
        print(f"   R^2={result['rsquared']}, p_spline={result['p_val_spline']}")

    return results


def compute_partial_dependence(result, n_points=100):
    """
    计算温度指标的偏依赖图 (Partial Dependence Plot)。
    控制 location 为最常见值，year 为均值。
    """
    model = result["model"]
    model_df = result["model_df"]
    indicator = result["indicator"]
    safe_indicator = result.get("safe_indicator", indicator)

    # 创建评估网格
    indicator_range = np.linspace(
        model_df[safe_indicator].min(),
        model_df[safe_indicator].max(),
        n_points
    )

    # 基线预测：固定其他协变量
    # 取最常见 location
    most_common_loc = model_df["location"].mode().values[0]
    mean_year = model_df["year"].mean()
    base = model_df[model_df["location"] == most_common_loc].iloc[[0]].copy()
    if len(base) == 0:
        base = model_df.iloc[[0]].copy()

    predictions = []
    for val in indicator_range:
        pred_df = base.copy()
        pred_df[safe_indicator] = val
        pred_df["location"] = most_common_loc
        pred_df["year"] = mean_year
        try:
            pred = model.predict(pred_df).values[0]
            predictions.append(pred)
        except Exception:
            predictions.append(np.nan)

    return pd.DataFrame({
        safe_indicator: indicator_range,
        "predicted_val": predictions,
    })


def main():
    args = parse_args()

    print("=" * 60)
    print("FE-GAM: 温度指标与脑卒中死亡率分析")
    print("=" * 60)

    # 1. 数据准备
    df = prepare_data(args.uwt_heat, args.uwt_cold, args.pwt)

    # 2. 指标分类
    categories = classify_indicators(df)
    print("\n指标分类:")
    for cat, cols in categories.items():
        if len(cols) > 0:
            print(f"  {cat}: {len(cols)} 个指标")

    # 3. 运行 FE-GAM
    all_results = []
    pd_data = {}

    # 定义要分析的组（用户指定的5类）
    run_groups = [
        ("基础温度_年均温", "年均温"),
        ("基础温度_年最高温", "年最高温"),
        ("基础温度_年最低温", "年最低温"),
        ("极端热", "极端热"),
        ("极端冷", "极端冷"),
    ]

    for cat_key, display_name in run_groups:
        indicators = categories.get(cat_key, [])
        if len(indicators) == 0:
            print(f"\n[{display_name}] 无指标，跳过")
            continue

        print(f"\n{'='*40}")
        print(f"[{display_name}] 共 {len(indicators)} 个指标")
        print("=" * 40)

        group_results = run_indicator_group(df, indicators, display_name, args.spline_df)
        all_results.extend(group_results)

        # 存储偏依赖数据
        for r in group_results:
            pd_data[r["indicator"]] = compute_partial_dependence(r)

    # 4. 汇总结果
    print(f"\n\n{'='*60}")
    print(f"分析完成！共运行 {len(all_results)} 个模型")
    print("=" * 60)

    if len(all_results) == 0:
        print("无有效模型结果。")
        return

    # 构建汇总表
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "指标组": r["group"],
            "指标字段": r["indicator"],
            "样本量": r["n_obs"],
            "地点数": r["n_locations"],
            "R_squared": r["rsquared"],
            "R_squared_adj": r["rsquared_adj"],
            "AIC": r["aic"],
            "BIC": r["bic"],
            "F_stat_样条": r["f_stat_spline"],
            "p_val_样条": r["p_val_spline"],
            "指标均值": r["indicator_mean"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # 显著性标记
    def significance(p):
        if p is None:
            return ""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        return ""

    summary_df["显著性"] = summary_df["p_val_样条"].apply(significance)

    # 5. 导出 Excel
    print(f"\n导出结果到 {args.output}...")
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        # Sheet 1: 模型汇总
        summary_df.to_excel(writer, sheet_name="FE_GAM模型汇总", index=False)

        # Sheets 2-6: 各指标组详细结果
        for group_name in summary_df["指标组"].unique():
            group_df = summary_df[summary_df["指标组"] == group_name].copy()
            sheet_name = group_name[:31]  # Excel sheet name limit
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sheet: 偏依赖数据
        if pd_data:
            pd_list = []
            for k, v in pd_data.items():
                v_copy = v.copy()
                v_copy.insert(0, "indicator", k)
                pd_list.append(v_copy)
            pd_combined = pd.concat(pd_list, axis=0, ignore_index=True)
            pd_combined.to_excel(writer, sheet_name="偏依赖数据", index=False)

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("模型结果摘要")
    print("=" * 60)
    for group_name in summary_df["指标组"].unique():
        gdf = summary_df[summary_df["指标组"] == group_name]
        sig_count = (gdf["p_val_样条"] < 0.05).sum()
        print(f"\n[{group_name}] {len(gdf)} 个模型, {sig_count} 个显著 (p<0.05)")
        for _, row in gdf.iterrows():
            sig = row["显著性"]
            print(f"  {row['指标字段']:<45s} R^2={row['R_squared']:<8.4f} p={row['p_val_样条']} {sig}")

    print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
