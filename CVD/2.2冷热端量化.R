# 冷端和热端RR量化

library(dplyr)
library(mgcv)
library(ggplot2)
library(tidyr)
library(purrr)
library(plm)


# 1.1读取数据 -----------------------------------------------------------------

df_unw<-read.csv('data/0.204_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv',na.strings = c())
df_pop<-read.csv('data/0.Pop_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv',na.strings = c())
data_pop<-read.csv("data/0.GBD/1.1GBD_Pop_location_2000-2020_iso_a3.csv")

savedir<-"result/0.人口加权前后冷热归因分数/"

data_pop<-data_pop[,c(6,7,10)]%>%
  rename(popnum=val)

df_unw<-left_join(df_unw,data_pop,by=c('iso_a3','year'))
df_pop<-left_join(df_pop,data_pop,by=c('iso_a3','year'))


# 2.1数据预处理 ----------------------------------------------------------------

prep_data <- function(df, temp_col){
  df %>%
    filter(year >= 2000 & year <= 2020) %>%
    # 基本缺失值过滤
    filter(!is.na(val),
           !is.na(.data[[temp_col]]),
           val > 0) %>%               # 避免 log(0)
    mutate(
      location = as.factor(location),
      year     = as.factor(year),
      SDI      = as.factor(SDI),
      # log(死亡率)，更适合高斯模型
      log_val  = log(val)
    )
}

df_unw <- prep_data(df_unw, "Mean_Temp")     # 未加权温度列：Mean_Temp
df_pop <- prep_data(df_pop, "mean_Mean")     # 人口加权温度列：mean_Mean

# -------------------------------------------------------------
# 2. 简单相关性（散点 + Spearman）
# -------------------------------------------------------------
# 未加权
cor_unw <- cor.test(df_unw$Mean_Temp, df_unw$val, method = "spearman")
r_unw   <- round(cor_unw$estimate, 3)
p_unw   <- signif(cor_unw$p.value, 3)
cor_unw
p1 <- ggplot(df_unw, aes(x = Mean_Temp, y = val)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title    = "Unweighted temperature vs stroke mortality",
       x        = "Unweighted annual mean temperature",
       y        = "Age-standardized stroke mortality",
       subtitle = paste0("Spearman r = ", r_unw, ", p = ", p_unw)) +
  theme_bw(base_size = 14)
print(p1)
# 人口加权
cor_pop <- cor.test(df_pop$mean_Mean, df_pop$val, method = "spearman")
r_pop   <- round(cor_pop$estimate, 3)
p_pop   <- signif(cor_pop$p.value, 3)
cor_pop
p2 <- ggplot(df_pop, aes(x = mean_Mean, y = val)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title    = "Population-weighted temperature vs stroke mortality",
       x        = "Population-weighted annual mean temperature",
       y        = "Age-standardized stroke mortality",
       subtitle = paste0("Spearman r = ", r_pop, ", p = ", p_pop)) +
  theme_bw(base_size = 14)
print(p2)

# ggsave(file.path(savedir, "Scatter_unweighted_temp_vs_mortality.png"),
#        p1, width = 6, height = 5)
# ggsave(file.path(savedir, "Scatter_popweighted_temp_vs_mortality.png"),
#        p2, width = 6, height = 5)

# -------------------------------------------------------------
# 3. 建模：固定效应 + 年份 + 国家（GAM, Gaussian on log_val）
# -------------------------------------------------------------
model_unw <- gam(
  log_val ~ s(Mean_Temp, bs = "cs") + location + year,
  data   = df_unw,
  family = gaussian()
)

model_pop <- gam(
  log_val ~ s(mean_Mean, bs = "cs") + location + year,
  data   = df_pop,
  family = gaussian()
)

# 查看温度平滑项显著性
summary(model_unw)$s.table
summary(model_pop)$s.table

# -------------------------------------------------------------
# 4. 暴露–反应曲线：UWT vs PWT 对比
# -------------------------------------------------------------
make_temp_grid <- function(df, temp_col){
  tibble(
    temp = seq(
      quantile(df[[temp_col]], 0.01, na.rm = TRUE),
      quantile(df[[temp_col]], 0.99, na.rm = TRUE),
      length.out = 200
    ),
    location = df$location[1],   # 固定一个国家与年份，仅用于可视化基准
    year     = df$year[1]
  )
}

grid_unw <- make_temp_grid(df_unw, "Mean_Temp")
colnames(grid_unw)[1]<-'Mean_Temp'
grid_pop <- make_temp_grid(df_pop, "mean_Mean")
colnames(grid_pop)[1]<-'mean_Mean'

# 预测 log(死亡率) + 95%CI，并还原到原始尺度
pred_unw <- predict(model_unw, newdata = grid_unw, se.fit = TRUE)
grid_unw <- grid_unw %>%
  mutate(
    fit_log  = pred_unw$fit,
    se_log   = pred_unw$se.fit,
    fit_val  = exp(fit_log),
    low_val  = exp(fit_log - 1.96 * se_log),
    high_val = exp(fit_log + 1.96 * se_log),
    type     = "Unweighted"
  )

pred_pop <- predict(model_pop, newdata = grid_pop, se.fit = TRUE)
grid_pop <- grid_pop %>%
  mutate(
    fit_log  = pred_pop$fit,
    se_log   = pred_pop$se.fit,
    fit_val  = exp(fit_log),
    low_val  = exp(fit_log - 1.96 * se_log),
    high_val = exp(fit_log + 1.96 * se_log),
    type     = "Population-weighted"
  )

curve_df <- bind_rows(
  grid_unw %>% rename(temp_metric = Mean_Temp),
  grid_pop %>% rename(temp_metric = mean_Mean)
)

# 提取 UWT 数据
curve_unw <- curve_df %>% filter(type == "Unweighted")
curve_pop <- curve_df %>% filter(type == "Population-weighted")

# 最小死亡点的温度（MMT）
MMT_unw <- curve_unw$temp_metric[which.min(curve_unw$fit_val)]
MMT_pop <- curve_pop$temp_metric[which.min(curve_pop$fit_val)]

# 偏移量
MMT_shift <- MMT_pop - MMT_unw

MMT_unw; MMT_pop; MMT_shift

p_curve <- ggplot(curve_df,
                  aes(x = temp_metric, y = fit_val,
                      color = type, fill = type)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = low_val, ymax = high_val),
              alpha = 0.2, color = NA) +
  theme_bw(base_size = 14) +
  labs(
    title = "Exposure–response curves under different temperature metrics",
    x     = "Annual mean temperature",
    y     = "Fitted age-standardized stroke mortality",
    color = "Temperature metric",
    fill  = "Temperature metric"
  )
print(p_curve)


ggsave(file.path(savedir, "GAM_exposure_response_UWT_vs_PWT.png"),
       p_curve, width = 7, height = 5)

# -------------------------------------------------------------
# 5. 模型预测敏感性分析：PWT vs UWT
#    比较同一国家–年份下两个模型预测的死亡率差异
# -------------------------------------------------------------
# 假定 df_unw 与 df_pop 在 (location, year) 上是一一对应的
df_pred <- df_unw %>%
  select(iso_a3, location, year, SDI, Mean_Temp, val) %>%
  mutate(
    mean_Mean = df_pop$mean_Mean
  )

# 对同一组点用两个模型分别预测
df_pred <- df_pred %>%
  mutate(
    log_pred_unw = predict(model_unw, newdata = data.frame(
      log_val   = log(val),
      Mean_Temp = Mean_Temp,
      location  = location,
      year      = year
    )),
    log_pred_pop = predict(model_pop, newdata = data.frame(
      log_val   = log(val),
      mean_Mean = mean_Mean,
      location  = location,
      year      = year
    )),
    pred_unw  = exp(log_pred_unw),
    pred_pop  = exp(log_pred_pop),
    diff_abs  = pred_pop - pred_unw,
    diff_rel  = (pred_pop - pred_unw) / pred_unw * 100
  )

# 5.1 按 SDI 分组的平均相对差异
sens_sdi <- df_pred %>%
  group_by(SDI) %>%
  summarise(
    mean_abs_diff = mean(diff_abs, na.rm = TRUE),
    mean_rel_diff = mean(diff_rel, na.rm = TRUE),
    .groups = "drop"
  )
sens_sdi
write.csv(sens_sdi,
          file.path(savedir, "Sensitivity_SDIGroups_PWT_vs_UWT.csv"),
          row.names = FALSE)

p_sdi <- ggplot(sens_sdi, aes(x = SDI, y = mean_rel_diff)) +
  geom_col(fill = "steelblue") +
  theme_bw(base_size = 14) +
  labs(
    title = "Sensitivity of model estimates to temperature metric by SDI",
    x     = "SDI group",
    y     = "Mean relative difference in fitted mortality (%)\n(Pop-weighted vs Unweighted)"
  )
print(p_sdi)
ggsave(file.path(savedir, "Sensitivity_by_SDI.png"),
       p_sdi, width = 6, height = 5)

# 5.2 按国家的敏感性（可用于做地图或排序）
sens_country <- df_pred %>%
  group_by(location, iso_a3) %>%
  summarise(
    mean_rel_diff = mean(diff_rel, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_rel_diff))

write.csv(sens_country,
          file.path(savedir, "Sensitivity_By_Country_PWT_vs_UWT.csv"),
          row.names = FALSE)

# 5.3 散点图：pred_pop vs pred_unw（直观看偏离 1:1 线）
p_scatter <- ggplot(df_pred, aes(x = pred_unw, y = pred_pop)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_bw(base_size = 14) +
  labs(
    title = "Comparison of fitted mortality under UWT vs PWT",
    x     = "Fitted mortality (Unweighted temperature)",
    y     = "Fitted mortality (Population-weighted temperature)"
  )
print(p_scatter)
ggsave(file.path(savedir, "Scatter_pred_PWT_vs_UWT.png"),
       p_scatter, width = 6, height = 5)

# 提取共同温度范围
t_low  <- quantile(curve_df$temp_metric, 0.05)
t_high <- quantile(curve_df$temp_metric, 0.95)

# 定义一个函数从曲线中查指定温度的风险
get_pred <- function(df, temp){
  df$fit_val[which.min(abs(df$temp_metric - temp))]
}

# Unweighted
risk_MMT_unw  <- get_pred(curve_unw, MMT_unw)
risk_cold_unw <- get_pred(curve_unw, t_low)
risk_hot_unw  <- get_pred(curve_unw, t_high)

cold_effect_unw <- risk_cold_unw - risk_MMT_unw
hot_effect_unw  <- risk_hot_unw  - risk_MMT_unw

# Population-weighted
risk_MMT_pop  <- get_pred(curve_pop, MMT_pop)
risk_cold_pop <- get_pred(curve_pop, t_low)
risk_hot_pop  <- get_pred(curve_pop, t_high)

cold_effect_pop <- risk_cold_pop - risk_MMT_pop
hot_effect_pop  <- risk_hot_pop  - risk_MMT_pop

# 变化值
delta_cold <- cold_effect_pop - cold_effect_unw
delta_hot  <- hot_effect_pop  - hot_effect_unw

delta_cold; delta_hot

rel_cold_change <- delta_cold / cold_effect_unw * 100
rel_hot_change  <- delta_hot  / hot_effect_unw  * 100

rel_cold_change; rel_hot_change

