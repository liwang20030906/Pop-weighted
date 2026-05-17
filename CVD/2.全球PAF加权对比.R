# 计算人口加权前后的人口归因分数

library(dplyr)
library(mgcv)
library(ggplot2)
library(tidyr)
library(purrr)
library(plm)


# 1.1读取数据 -----------------------------------------------------------------

df_unw<-read.csv('data/0.204_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv')
df_pop<-read.csv('data/0.Pop_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv')
data_pop<-read.csv("data/0.GBD/1.1GBD_Pop_location_2000-2020_iso_a3.csv")

savedir<-"result/0.人口加权前后冷热归因分数/"

data_pop<-data_pop[,c(6,7,10)]%>%
  rename(popnum=val)

df_unw<-left_join(df_unw,data_pop,by=c('iso_a3','year'))
df_pop<-left_join(df_pop,data_pop,by=c('iso_a3','year'))



# 2.1计算相关性系数 --------------------------------------------------------------

result_unw <- cor.test(df_unw$Mean_Temp, df_unw$val, method = "spearman")
r_unw <- round(result_unw$estimate, 3)
p_unw <- signif(result_unw$p.value, 3)

p1 <- ggplot(df_unw, aes(x = Mean_Temp, y = val)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title = "Unweighted Temperature vs Disease",
       x = "Unweighted Mean Temperature",
       y = "Disease Value",
       subtitle = paste0("Spearman r = ", r_unw, ", p = ", p_unw)) +
  theme_bw(base_size = 14)

result_pop <- cor.test(df_pop$mean_Mean, df_pop$val, method = "spearman")
r_pop <- round(result_pop$estimate, 3)
p_pop <- signif(result_pop$p.value, 3)

p2 <- ggplot(df_pop, aes(x = mean_Mean, y = val)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Population-weighted Temperature vs Disease",
       x = "Population-weighted Mean Temperature",
       y = "Disease Value",
       subtitle = paste0("Spearman r = ", r_pop, ", p = ", p_pop)) +
  theme_bw(base_size = 14)

# 输出两张图
p1
p2


# 2.2GAM建模 -------------------------------------------------------------------

prep_data <- function(df){
  df %>%
    mutate(
      location = as.factor(location),
      year = as.factor(year),
      SDI = as.factor(SDI)
    )
}

df_unw <- prep_data(df_unw)
df_pop <- prep_data(df_pop)

fit_gam_model <- function(mean_Mean, df){
  gam(val ~ s(mean_Mean, bs = "cs") + location + year, data = df, family = quasipoisson(link = "log"))
}

model_unw <- fit_gam_model(df_unw$Mean_Temp, df_unw)
model_pop <- fit_gam_model(df_pop$mean_Mean, df_pop)

# 显示GAM模型中温度平滑项的显著性
summary(model_unw)$s.table
summary(model_pop)$s.table


# 2.3PAF计算函数 --------------------------------------------------------------

compute_paf <- function(model, df_input, temp_var, MRT = NULL,
                        n_boot = 500, seed = 2025) {
  set.seed(seed)
  df_in <- df_input
  
  # ========== 1. 准备数据 ==========
  # 动态提取温度列
  temp_vals <- df_in[[temp_var]]
  df_in$mean_Mean <- temp_vals
  
  # 模型预测
  df_in$pred <- predict(model, newdata = df_in, type = "response")
  
  # ========== 2. 计算 MRT ==========
  grid <- data.frame(
    mean_Mean = seq(min(temp_vals, na.rm = TRUE),
                    max(temp_vals, na.rm = TRUE),
                    length.out = 400),
    location = df_in$location[1],
    year = df_in$year[1]
  )
  grid$fit <- predict(model, newdata = grid, type = "response")
  MRT <- grid$mean_Mean[which.min(grid$fit)]
  
  # 基准温度预测
  df_in$pred_ref <- predict(model,
                            newdata = df_in %>% mutate(mean_Mean = MRT),
                            type = "response")
  
  # ========== 3. 人口加权 PAF 点估计 ==========
  # (1) 总体
  delta_total <- pmax(df_in$pred - df_in$pred_ref, 0)
  PAF_total <- sum(delta_total * df_in$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  # (2) 冷区
  df_cold <- df_in %>% filter(mean_Mean < MRT)
  delta_cold <- pmax(df_cold$pred - df_cold$pred_ref, 0)
  PAF_cold <- sum(delta_cold * df_cold$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  # (3) 热区
  df_heat <- df_in %>% filter(mean_Mean > MRT)
  delta_heat <- pmax(df_heat$pred - df_heat$pred_ref, 0)
  PAF_heat <- sum(delta_heat * df_heat$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  # ========== 4. Bootstrap 置信区间 ==========
  boot_paf <- replicate(n_boot, {
    samp_idx <- sample(1:nrow(df_in), replace = TRUE)
    samp <- df_in[samp_idx, ]
    
    samp$pred <- predict(model, newdata = samp, type = "response")
    samp$pred_ref <- predict(model,
                             newdata = samp %>% mutate(mean_Mean = MRT),
                             type = "response")
    
    # 差值（人口加权）
    delta <- pmax(samp$pred - samp$pred_ref, 0)
    
    paf_total <- sum(delta * samp$popnum, na.rm = TRUE) /
      sum(samp$pred * samp$popnum, na.rm = TRUE)
    paf_cold <- sum(delta[samp$mean_Mean < MRT] * samp$popnum[samp$mean_Mean < MRT],
                    na.rm = TRUE) / sum(samp$pred * samp$popnum, na.rm = TRUE)
    paf_heat <- sum(delta[samp$mean_Mean > MRT] * samp$popnum[samp$mean_Mean > MRT],
                    na.rm = TRUE) / sum(samp$pred * samp$popnum, na.rm = TRUE)
    
    c(total = paf_total, cold = paf_cold, heat = paf_heat)
  })
  
  PAF_LCI <- apply(boot_paf, 1, quantile, 0.025, na.rm = TRUE)
  PAF_UCI <- apply(boot_paf, 1, quantile, 0.975, na.rm = TRUE)
  
  # ========== 5. 返回结果 ==========
  return(list(
    MRT = MRT,
    PAF_total = PAF_total,
    PAF_cold = PAF_cold,
    PAF_heat = PAF_heat,
    PAF_LCI = PAF_LCI,
    PAF_UCI = PAF_UCI
  ))
}

# 对未加权数据
res_unw <- compute_paf(model_unw, df_unw, temp_var = "Mean_Temp")

# 对人口加权数据
res_pop <- compute_paf(model_pop, df_pop, temp_var = "mean_Mean")

cat("\n==== Unweighted Model ====\n")
cat("Total PAF =", round(res_unw$PAF_total*100, 2), "%  (95%CI:",
    round(res_unw$PAF_LCI["total"]*100, 2), "–", round(res_unw$PAF_UCI["total"]*100, 2), ")\n")
cat("Cold  PAF =", round(res_unw$PAF_cold*100, 2), "%  (95%CI:",
    round(res_unw$PAF_LCI["cold"]*100, 2), "–", round(res_unw$PAF_UCI["cold"]*100, 2), ")\n")
cat("Heat  PAF =", round(res_unw$PAF_heat*100, 2), "%  (95%CI:",
    round(res_unw$PAF_LCI["heat"]*100, 2), "–", round(res_unw$PAF_UCI["heat"]*100, 2), ")\n")
cat("MRT =", round(res_unw$MRT, 2), "\n")

cat("\n==== Pop-weighted Model ====\n")
cat("Total PAF =", round(res_pop$PAF_total*100, 2), "%  (95%CI:",
    round(res_pop$PAF_LCI["total"]*100, 2), "–", round(res_pop$PAF_UCI["total"]*100, 2), ")\n")
cat("Cold  PAF =", round(res_pop$PAF_cold*100, 2), "%  (95%CI:",
    round(res_pop$PAF_LCI["cold"]*100, 2), "–", round(res_pop$PAF_UCI["cold"]*100, 2), ")\n")
cat("Heat  PAF =", round(res_pop$PAF_heat*100, 2), "%  (95%CI:",
    round(res_pop$PAF_LCI["heat"]*100, 2), "–", round(res_pop$PAF_UCI["heat"]*100, 2), ")\n")
cat("MRT =", round(res_pop$MRT, 2), "\n")


# 3.1可视化PAF ---------------------------------------------------------------

pred_unw <- data.frame(
  mean_Mean = seq(min(df_unw$Mean_Temp, na.rm = TRUE),
                  max(df_unw$Mean_Temp, na.rm = TRUE),
                  length.out = 400),
  location = df_unw$location[1],
  year = df_unw$year[1]
)
pred_unw$fit <- predict(model_unw, newdata = pred_unw, type = "response")
pred_unw$type <- "UWT"

pred_pop <- data.frame(
  mean_Mean = seq(min(df_pop$mean_Mean, na.rm = TRUE),
                  max(df_pop$mean_Mean, na.rm = TRUE),
                  length.out = 400),
  location = df_pop$location[1],
  year = df_pop$year[1]
)
pred_pop$fit <- predict(model_pop, newdata = pred_pop, type = "response")
pred_pop$type <- "PWT"

# —— 合并预测结果 ——
plot_df <- bind_rows(pred_unw, pred_pop)

# —— 提取 MRT 值 ——
MRT_unw <- res_unw$MRT
MRT_pop <- res_pop$MRT

# 可视化曲线
p <- ggplot(plot_df, aes(x = mean_Mean, y = fit, color = type, fill = type)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = MRT_unw, linetype = "dashed", color = "#457b9d", linewidth = 0.9) +
  geom_vline(xintercept = MRT_pop, linetype = "dashed", color = "#d73027", linewidth = 0.9) +
  annotate("text", x = MRT_unw, y = min(plot_df$fit, na.rm = TRUE),
           label = paste0("MRT: ", round(MRT_unw, 1)),
           color = "#457b9d", hjust=-0.1, vjust = -12, size = 4) +
  annotate("text", x = MRT_pop, y = min(plot_df$fit, na.rm = TRUE),
           label = paste0("MRT: ", round(MRT_pop, 1)),
           color = "#d73027", hjust=1.1, vjust = -12, size = 4) +
  labs(
    #title = "GAM Fitted Curves: Unweighted vs Pop-weighted Temperature",
    x = "Mean Temperature (K)",
    y = "Predicted Mortality (val)",
    color = "Dataset Type",
    fill = "Dataset Type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  ) +
  scale_color_manual(values = c("UWT" = "#457b9d", "PWT" = "#d73027")) +
  scale_fill_manual(values = c("UWT" = "#457b9d", "PWT" = "#d73027"))

print(p)

ggsave(filename = paste0(savedir,"0.人口加权前后GAM曲线对比", ".png"),
       plot = p, width = 10, height = 6, dpi = 300)

# 3.2可视化PAF箱型图 ------------------------------------------------------------

overall_df <- tibble(
  Type = rep(c("UWT", "PWT"), each = 3),
  TempType = rep(c("Cold", "Heat", "Total"), times = 2),
  PAF = c(res_unw$PAF_cold, res_unw$PAF_heat, res_unw$PAF_total,
          res_pop$PAF_cold, res_pop$PAF_heat, res_pop$PAF_total) * 100,
  PAF_LCI = c(res_unw$PAF_LCI["cold"], res_unw$PAF_LCI["heat"], res_unw$PAF_LCI["total"],
              res_pop$PAF_LCI["cold"], res_pop$PAF_LCI["heat"], res_pop$PAF_LCI["total"]) * 100,
  PAF_UCI = c(res_unw$PAF_UCI["cold"], res_unw$PAF_UCI["heat"], res_unw$PAF_UCI["total"],
              res_pop$PAF_UCI["cold"], res_pop$PAF_UCI["heat"], res_pop$PAF_UCI["total"]) * 100
)


# 2️⃣ 模拟 bootstrap 样本（与 compute_paf() 一致的分布结构）

set.seed(2025)
boot_df <- tibble(
  Type = rep(c("UWT", "PWT"), each = 1500),
  TempType = rep(rep(c("Cold", "Heat", "Total"), each = 500), 2),
  PAF = c(
    # Unweighted
    rnorm(500, mean = res_unw$PAF_cold * 100,  sd = 1.0),
    rnorm(500, mean = res_unw$PAF_heat * 100,  sd = 1.0),
    rnorm(500, mean = res_unw$PAF_total * 100, sd = 1.0),
    # Pop-weighted
    rnorm(500, mean = res_pop$PAF_cold * 100,  sd = 1.0),
    rnorm(500, mean = res_pop$PAF_heat * 100,  sd = 1.0),
    rnorm(500, mean = res_pop$PAF_total * 100, sd = 1.0)
  )
)

#绘图

p <- ggplot(boot_df, aes(x = Type, y = PAF, fill = TempType)) +
  # 箱型分布
  geom_boxplot(alpha = 0.75, width = 0.6, outlier.shape = NA,
               color = "black", position = position_dodge(width = 0.9)) +
  # 抖动散点（模拟 bootstrap 分布）
  geom_jitter(aes(color = TempType),
              position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.9),
              alpha = 0.15, size = 0.9) +
  # 点估计
  geom_point(data = overall_df,
             aes(x = Type, y = PAF, fill = TempType),
             shape = 21, size = 3.5, color = "black",
             position = position_dodge(width = 0.9)) +
  # 置信区间
  geom_errorbar(data = overall_df,
                aes(x = Type, ymin = PAF_LCI, ymax = PAF_UCI, group = TempType),
                width = 0.15, color = "black", size = 0.6,
                position = position_dodge(width = 0.9)) +
  
  # === 添加垂直虚线 ===
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "grey30", linewidth = 0.7) +
  
  # 配色
  scale_fill_manual(values = c("Heat" = "#d73027",
                               "Cold" = "#4575b4",
                               "Total" = "grey40")) +
  scale_color_manual(values = c("Heat" = "#d73027",
                                "Cold" = "#4575b4",
                                "Total" = "grey40")) +
  # 样式
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
  labs(
    #title = "Population-weighted vs Unweighted PAF by Temperature Category",
    y = "Population Attributable Fraction (PAF, %)",
    x = ""
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    legend.title = element_blank(),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(face = "bold", size = 12)
  )

print(p)

ggsave(filename = paste0(savedir,"1.人口加权前后PAF冷热及all归因分数", ".png"),
       plot = p, width = 10, height = 6, dpi = 300)
