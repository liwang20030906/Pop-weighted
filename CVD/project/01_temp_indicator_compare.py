import pandas as pd
import numpy as np
import re
import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "1.Temp")


# =========================
# 1. CLI 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="UWT vs PWT 温度指标对比分析工具 —— 计算人口加权与非人口加权的均值差异并生成汇总 Excel"
    )
    parser.add_argument(
        "--uwt-heat", "-uh",
        default=os.path.join(DATA_DIR, "204High_Stroke_SDI_zone_all.csv"),
        help="UWT 高温文件路径 (默认: 204High_Stroke_SDI_zone_all.csv)"
    )
    parser.add_argument(
        "--uwt-cold", "-uc",
        default=os.path.join(DATA_DIR, "204Low_Stroke_SDI_zone_all.csv"),
        help="UWT 低温文件路径 (默认: 204Low_Stroke_SDI_zone_all.csv)"
    )
    parser.add_argument(
        "--pwt", "-p",
        default=os.path.join(DATA_DIR, "Mean_pop_Stroke_SDI_zone_all.csv"),
        help="PWT 人口加权文件路径 (默认: Mean_pop_Stroke_SDI_zone_all.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(BASE_DIR, "UWT_PWT_indicator_summary.xlsx"),
        help="输出 Excel 文件路径 (默认: UWT_PWT_indicator_summary.xlsx)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细输出"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    uwt_heat_path = args.uwt_heat
    uwt_cold_path = args.uwt_cold
    pwt_path = args.pwt
    output_path = args.output


    # =========================
    # 2. 读取 CSV
    # =========================
    def read_csv_auto(path):
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)


    uwt_heat_df = read_csv_auto(uwt_heat_path)
    uwt_cold_df = read_csv_auto(uwt_cold_path)
    pwt_df = read_csv_auto(pwt_path)


    # =========================
    # 3. 合并 UWT 高温 + 低温文件
    # =========================
    def normalize_text(s):
        s = str(s).strip().lower()
        s = s.replace("℃", "c")
        s = s.replace("-", "_")
        s = s.replace(" ", "_")
        s = re.sub(r"_+", "_", s)
        return s.strip("_")


    def get_id_columns(df_list):
        """
        自动寻找多个文件共有的标识字段。
        这些字段用于横向合并 high 和 low。
        """
        candidate_cols = [
            "measure",
            "location",
            "sex",
            "age",
            "cause",
            "metric",
            "year",
            "iso_a3",
            "country",
            "sdi",
            "zone",
            "continent",
        ]

        common_cols = set(df_list[0].columns)

        for df in df_list[1:]:
            common_cols = common_cols & set(df.columns)

        id_cols = [c for c in candidate_cols if c in common_cols]

        # 如果没有识别到，至少使用 year
        if len(id_cols) == 0 and all("year" in df.columns for df in df_list):
            id_cols = ["year"]

        return id_cols


    def merge_uwt_heat_cold(heat_df, cold_df):
        """
        将 UWT 高温文件和低温文件横向合并。
        若两个文件中存在同名温度字段，会保留 high 版本并丢弃 low 中重复字段。
        """

        id_cols = get_id_columns([heat_df, cold_df])

        print("UWT high/low 合并使用的标识字段：", id_cols)

        if len(id_cols) == 0:
            print("警告：没有找到可用于横向合并的共同标识字段，将采用纵向拼接方式。")
            return pd.concat([heat_df, cold_df], axis=0, ignore_index=True)

        heat_df = heat_df.copy()
        cold_df = cold_df.copy()

        # 避免 year 类型不一致导致 merge 失败
        if "year" in id_cols:
            heat_df["year"] = pd.to_numeric(heat_df["year"], errors="coerce")
            cold_df["year"] = pd.to_numeric(cold_df["year"], errors="coerce")

        heat_df = heat_df.drop_duplicates(subset=id_cols)
        cold_df = cold_df.drop_duplicates(subset=id_cols)

        heat_cols = set(heat_df.columns)
        cold_extra_cols = [
            c for c in cold_df.columns
            if c in id_cols or c not in heat_cols
        ]

        cold_df = cold_df[cold_extra_cols]

        merged_df = pd.merge(
            heat_df,
            cold_df,
            on=id_cols,
            how="outer"
        )

        return merged_df


    uwt_df = merge_uwt_heat_cold(uwt_heat_df, uwt_cold_df)

    print("UWT heat 行列数：", uwt_heat_df.shape)
    print("UWT cold 行列数：", uwt_cold_df.shape)
    print("合并后 UWT 行列数：", uwt_df.shape)
    print("PWT 行列数：", pwt_df.shape)


    # =========================
    # 4. 字段语义解析工具
    # =========================
    def p_to_str(x):
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return "%g" % x


    def extract_percentile(text):
        t = normalize_text(text)

        patterns = [
            r"(?:^|_)p_?(\d+(?:\.\d+)?)($|_)",
            r"(?:^|_)(\d+(?:\.\d+)?)_?(?:st|nd|rd|th)?_?percentile($|_)",
            r"percentile_?(\d+(?:\.\d+)?)",
        ]

        for pat in patterns:
            m = re.search(pat, t)
            if m:
                return float(m.group(1))

        return None


    def extract_duration(text):
        t = normalize_text(text)

        patterns = [
            r"(?:^|_)(\d+)_?d($|_)",
            r"(?:^|_)(\d+)_?day(?:s)?($|_)",
        ]

        matches_all = []

        for pat in patterns:
            matches = re.findall(pat, t)
            if matches:
                matches_all.extend(matches)

        if len(matches_all) == 0:
            return None

        last = matches_all[-1]

        if isinstance(last, tuple):
            return int(last[0])

        return int(last)


    def percentile_label(p):
        p_str = p_to_str(p)

        if p_str == "1":
            return "1st percentile"
        elif p_str == "2":
            return "2nd percentile"
        elif p_str == "3":
            return "3rd percentile"
        else:
            return f"{p_str}th percentile"


    def parse_indicator_signature(col):
        t = normalize_text(col)
        compact = t.replace("_", "")

        p = extract_percentile(t)
        d = extract_duration(t)

        # 热浪 / 热暴露
        is_heat = any(k in compact for k in [
            "heatwave",
            "heatwaves",
            "hotday",
            "hotdays",
            "hwd"
        ])

        if is_heat:
            if p is None and d is None:
                return None

            p_key = p_to_str(p) if p is not None else "none"
            d_key = str(d) if d is not None else "none"

            semantic_key = f"heatwave|p={p_key}|d={d_key}"

            if p is not None and d is not None:
                indicator_name = f"heatwave days, P{p_to_str(p)}, {d}d"
            elif p is not None:
                indicator_name = f"heatwave days, P{p_to_str(p)}"
            elif d is not None:
                indicator_name = f"heatwave days, {d}d"
            else:
                indicator_name = str(col)

            sort_value = 4000 + (p if p is not None else 0) * 10 + (d if d is not None else 0)

            return {
                "指标类型": "热暴露",
                "指标名称": indicator_name,
                "语义键": semantic_key,
                "P阈值": p,
                "持续天数": d,
                "排序值": sort_value,
            }

        # 冷事件 / 冷暴露
        is_cold = any(k in compact for k in [
            "coldspell",
            "coldspells",
            "coldwave",
            "coldwaves",
            "coldday",
            "colddays",
            "csd"
        ])

        if is_cold:
            if p is None and d is None:
                return None

            p_key = p_to_str(p) if p is not None else "none"
            d_key = str(d) if d is not None else "none"

            semantic_key = f"coldspell|p={p_key}|d={d_key}"

            if p is not None and d is not None:
                indicator_name = f"cold spell days, P{p_to_str(p)}, {d}d"
            elif p is not None:
                indicator_name = f"cold spell days, P{p_to_str(p)}"
            elif d is not None:
                indicator_name = f"cold spell days, {d}d"
            else:
                indicator_name = str(col)

            sort_value = 5000 + (p if p is not None else 0) * 10 + (d if d is not None else 0)

            return {
                "指标类型": "冷暴露",
                "指标名称": indicator_name,
                "语义键": semantic_key,
                "P阈值": p,
                "持续天数": d,
                "排序值": sort_value,
            }

        # 分位数指标
        if p is not None:
            if p >= 80:
                metric_type = "极端热"
                sort_value = 1000 + p
            elif p <= 20:
                metric_type = "极端冷"
                sort_value = 2000 + p
            else:
                metric_type = "温度分位数"
                sort_value = 3000 + p

            semantic_key = f"percentile|p={p_to_str(p)}"

            return {
                "指标类型": metric_type,
                "指标名称": percentile_label(p),
                "语义键": semantic_key,
                "P阈值": p,
                "持续天数": np.nan,
                "排序值": sort_value,
            }

        # 基础温度
        base_patterns = [
            (
                "base_mean",
                "年均温",
                [
                    "meanmean",
                    "annualmean",
                    "annualaveragetemp",
                    "averagetemperature",
                    "averagetemp",
                    "avgtemp",
                    "tmean",
                    "meantemp",
                    "meantemperature",
                    "temperaturemean",
                ],
            ),
            (
                "base_max",
                "年最高温",
                [
                    "meanmax",
                    "annualmax",
                    "annualmaximumtemp",
                    "maxtemp",
                    "tmax",
                    "maximumtemperature",
                    "temperaturemax",
                ],
            ),
            (
                "base_min",
                "年最低温",
                [
                    "meanmin",
                    "annualmin",
                    "annualminimumtemp",
                    "mintemp",
                    "tmin",
                    "minimumtemperature",
                    "temperaturemin",
                ],
            ),
        ]

        for semantic_key, indicator_name, patterns in base_patterns:
            if compact in patterns or any(compact.endswith(pat) for pat in patterns):
                sort_value = {
                    "base_mean": 1,
                    "base_max": 2,
                    "base_min": 3,
                }[semantic_key]

                return {
                    "指标类型": "基础温度",
                    "指标名称": indicator_name,
                    "语义键": semantic_key,
                    "P阈值": np.nan,
                    "持续天数": np.nan,
                    "排序值": sort_value,
                }

        return None


    # =========================
    # 5. 提取温度指标字段
    # =========================
    EXCLUDE_COLUMNS = {
        "measure",
        "location",
        "sex",
        "age",
        "cause",
        "metric",
        "year",
        "val",
        "upper",
        "lower",
        "iso_a3",
        "country",
        "sdi",
        "zone",
        "continent",
    }


    def is_numeric_like(series, min_ratio=0.5):
        x = pd.to_numeric(series, errors="coerce")
        return x.notna().mean() >= min_ratio


    def extract_temperature_indicators(df, source_name):
        rows = []

        for col in df.columns:
            col_norm = normalize_text(col)

            if col_norm in EXCLUDE_COLUMNS:
                continue

            if not is_numeric_like(df[col]):
                continue

            info = parse_indicator_signature(col)

            if info is None:
                continue

            rows.append({
                "数据源": source_name,
                "原始字段名": col,
                "指标类型": info["指标类型"],
                "指标名称": info["指标名称"],
                "语义键": info["语义键"],
                "P阈值": info["P阈值"],
                "持续天数": info["持续天数"],
                "排序值": info["排序值"],
            })

        result = pd.DataFrame(
            rows,
            columns=[
                "数据源",
                "原始字段名",
                "指标类型",
                "指标名称",
                "语义键",
                "P阈值",
                "持续天数",
                "排序值",
            ]
        )

        if len(result) > 0:
            result = result.sort_values(["排序值", "语义键", "原始字段名"]).reset_index(drop=True)

        return result


    # =========================
    # 6. 按语义键匹配 UWT 和 PWT
    # =========================
    def match_indicators_by_semantics(uwt_indicators, pwt_indicators):
        output_cols = [
            "指标类型",
            "指标名称",
            "语义键",
            "P阈值",
            "持续天数",
            "UWT字段",
            "PWT字段",
            "排序值",
        ]

        if len(uwt_indicators) == 0 or len(pwt_indicators) == 0:
            return pd.DataFrame(columns=output_cols)

        uwt_first = (
            uwt_indicators
            .sort_values(["排序值", "原始字段名"])
            .drop_duplicates(subset=["语义键"], keep="first")
        )

        pwt_first = (
            pwt_indicators
            .sort_values(["排序值", "原始字段名"])
            .drop_duplicates(subset=["语义键"], keep="first")
        )

        merged = pd.merge(
            uwt_first,
            pwt_first,
            on="语义键",
            how="inner",
            suffixes=("_UWT", "_PWT")
        )

        if len(merged) == 0:
            return pd.DataFrame(columns=output_cols)

        result = pd.DataFrame({
            "指标类型": merged["指标类型_PWT"],
            "指标名称": merged["指标名称_PWT"],
            "语义键": merged["语义键"],
            "P阈值": merged["P阈值_PWT"],
            "持续天数": merged["持续天数_PWT"],
            "UWT字段": merged["原始字段名_UWT"],
            "PWT字段": merged["原始字段名_PWT"],
            "排序值": merged["排序值_PWT"],
        })

        result = result.sort_values(["排序值", "语义键"]).reset_index(drop=True)

        return result


    # =========================
    # 7. 选择国家-年份匹配字段
    # =========================
    def choose_pair_id_cols(uwt_df, pwt_df):
        candidates = [
            ["iso_a3", "year"],
            ["location", "year"],
            ["country", "year"],
            ["Country", "year"],
            ["year"],
        ]

        for cols in candidates:
            if all(c in uwt_df.columns for c in cols) and all(c in pwt_df.columns for c in cols):
                return cols

        return []


    id_cols = choose_pair_id_cols(uwt_df, pwt_df)

    print("UWT 与 PWT 计算均值时使用的国家-年份匹配字段：", id_cols)


    # =========================
    # 8. 计算同一批国家-年份下的均值
    # =========================
    def calculate_pairwise_mean(uwt_df, pwt_df, uwt_col, pwt_col, id_cols):
        if len(id_cols) > 0:
            uwt_tmp = uwt_df[id_cols + [uwt_col]].copy()
            pwt_tmp = pwt_df[id_cols + [pwt_col]].copy()

            if "year" in id_cols:
                uwt_tmp["year"] = pd.to_numeric(uwt_tmp["year"], errors="coerce")
                pwt_tmp["year"] = pd.to_numeric(pwt_tmp["year"], errors="coerce")

            uwt_tmp[uwt_col] = pd.to_numeric(uwt_tmp[uwt_col], errors="coerce")
            pwt_tmp[pwt_col] = pd.to_numeric(pwt_tmp[pwt_col], errors="coerce")

            uwt_tmp = uwt_tmp.drop_duplicates(subset=id_cols)
            pwt_tmp = pwt_tmp.drop_duplicates(subset=id_cols)

            merged = pd.merge(
                uwt_tmp,
                pwt_tmp,
                on=id_cols,
                how="inner"
            )

            uwt_mean = merged[uwt_col].mean(skipna=True)
            pwt_mean = merged[pwt_col].mean(skipna=True)
            n_pairs = len(merged)

        else:
            uwt_mean = pd.to_numeric(uwt_df[uwt_col], errors="coerce").mean(skipna=True)
            pwt_mean = pd.to_numeric(pwt_df[pwt_col], errors="coerce").mean(skipna=True)
            n_pairs = min(len(uwt_df), len(pwt_df))

        return uwt_mean, pwt_mean, n_pairs


    def judge_direction(diff, threshold=1e-6):
        if pd.isna(diff):
            return ""

        if diff > threshold:
            return "升高"
        elif diff < -threshold:
            return "降低"
        else:
            return "基本不变"


    def make_explanation(metric_type, direction):
        if direction == "":
            return ""

        if direction == "基本不变":
            return "人口加权前后该指标差异较小，说明人口空间分布对该温度指标的影响有限。"

        if metric_type == "基础温度":
            if direction == "升高":
                return "人口加权后温度升高，表明人口更多分布在相对较暖区域。"
            else:
                return "人口加权后温度降低，表明人口更多分布在相对较冷区域。"

        if metric_type == "极端热":
            if direction == "升高":
                return "人口加权后高温分位数升高，表明人口暴露于更高温环境的程度增强。"
            else:
                return "人口加权后高温分位数降低，表明人口暴露于极端高温环境的程度减弱。"

        if metric_type == "极端冷":
            if direction == "升高":
                return "人口加权后低温分位数升高，表明人口所在区域的极端低温相对较弱。"
            else:
                return "人口加权后低温分位数降低，表明人口暴露于更冷环境的程度增强。"

        if metric_type == "热暴露":
            if direction == "升高":
                return "人口加权后热暴露天数增加，表明人口更集中于热浪或高温事件更频繁的区域。"
            else:
                return "人口加权后热暴露天数减少，表明人口更少分布在热浪或高温事件频繁区域。"

        if metric_type == "冷暴露":
            if direction == "升高":
                return "人口加权后冷暴露天数增加，表明人口更集中于冷事件更频繁的区域。"
            else:
                return "人口加权后冷暴露天数减少，表明人口更少分布在冷事件频繁区域。"

        if direction == "升高":
            return "人口加权后该温度指标升高，说明人口空间分布提高了该指标对应的暴露水平。"
        else:
            return "人口加权后该温度指标降低，说明人口空间分布降低了该指标对应的暴露水平。"


    # =========================
    # 9. 主程序
    # =========================
    uwt_indicators = extract_temperature_indicators(uwt_df, "UWT")
    pwt_indicators = extract_temperature_indicators(pwt_df, "PWT")

    matched_metrics = match_indicators_by_semantics(uwt_indicators, pwt_indicators)

    summary_rows = []

    for _, row in matched_metrics.iterrows():
        metric_type = row["指标类型"]
        metric_name = row["指标名称"]
        uwt_col = row["UWT字段"]
        pwt_col = row["PWT字段"]

        uwt_mean, pwt_mean, n_pairs = calculate_pairwise_mean(
            uwt_df=uwt_df,
            pwt_df=pwt_df,
            uwt_col=uwt_col,
            pwt_col=pwt_col,
            id_cols=id_cols
        )

        diff = pwt_mean - uwt_mean
        direction = judge_direction(diff)
        explanation = make_explanation(metric_type, direction)

        summary_rows.append({
            "指标类型": metric_type,
            "指标名称": metric_name,
            "UWT 均值": uwt_mean,
            "PWT 均值": pwt_mean,
            "PWT-UWT": diff,
            "方向": direction,
            "解释": explanation,
            "有效国家-年份数": n_pairs,
            "UWT字段": uwt_col,
            "PWT字段": pwt_col,
            "语义键": row["语义键"],
        })

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) > 0:
        for col in ["UWT 均值", "PWT 均值", "PWT-UWT"]:
            summary_df[col] = summary_df[col].round(3)


    # =========================
    # 10. 未匹配字段检查
    # =========================
    uwt_keys = set(uwt_indicators["语义键"]) if len(uwt_indicators) > 0 else set()
    pwt_keys = set(pwt_indicators["语义键"]) if len(pwt_indicators) > 0 else set()

    only_uwt_df = uwt_indicators[uwt_indicators["语义键"].isin(uwt_keys - pwt_keys)].copy()
    only_pwt_df = pwt_indicators[pwt_indicators["语义键"].isin(pwt_keys - uwt_keys)].copy()

    duplicate_uwt = uwt_indicators[uwt_indicators.duplicated(subset=["语义键"], keep=False)].copy()
    duplicate_pwt = pwt_indicators[pwt_indicators.duplicated(subset=["语义键"], keep=False)].copy()

    duplicate_df = pd.concat([duplicate_uwt, duplicate_pwt], axis=0)


    # =========================
    # 11. 导出 Excel
    # =========================
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="全部共同指标汇总", index=False)
        matched_metrics.to_excel(writer, sheet_name="语义匹配字段", index=False)

        uwt_indicators.to_excel(writer, sheet_name="UWT合并后识别字段", index=False)
        pwt_indicators.to_excel(writer, sheet_name="PWT识别字段", index=False)

        only_uwt_df.to_excel(writer, sheet_name="仅UWT未匹配", index=False)
        only_pwt_df.to_excel(writer, sheet_name="仅PWT未匹配", index=False)
        duplicate_df.to_excel(writer, sheet_name="重复语义字段", index=False)


    # =========================
    # 12. 打印检查信息
    # =========================
    print("\n统计完成！")
    print(f"结果已保存至：{output_path}")

    print("\nUWT 合并后识别到的温度指标数量：", len(uwt_indicators))
    print("PWT 识别到的温度指标数量：", len(pwt_indicators))
    print("按语义成功匹配的指标数量：", len(matched_metrics))

    print("\n仅 UWT 存在、PWT 没有匹配上的指标数量：", len(only_uwt_df))
    print("仅 PWT 存在、UWT 没有匹配上的指标数量：", len(only_pwt_df))

    print("\n前 20 行统计结果：")
    print(summary_df.head(20))


if __name__ == "__main__":
    main()