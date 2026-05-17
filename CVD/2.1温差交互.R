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
diff<-read.csv('data/1.Temp/pop_diff_mean_temp.csv')
data_pop<-read.csv("data/0.GBD/1.1GBD_Pop_location_2000-2020_iso_a3.csv")

savedir<-"result/0.人口加权前后冷热归因分数/"

data_pop <- data_pop[, c(6,7,10)] %>%
  rename(popnum = val)

df_unw <- left_join(df_unw, data_pop, by = c('iso_a3','year'))
df_pop <- left_join(df_pop, data_pop, by = c('iso_a3','year'))

diff<-diff[,c(2,3,8)]
df_unw<-left_join(df_unw,diff,by=c('iso_a3','year'))%>%
  filter(year < 2021)
df_pop<-left_join(df_pop,diff,by=c('iso_a3','year'))

# 1.2对国家温差数据进行分层 ----------------------------------------------------------

country_mean_diff<-df_unw%>%
  group_by(iso_a3) %>%
  summarise(
    mean_diff = mean(temp_diff, na.rm = TRUE),
    .groups = "drop"
  )

country_q4 <- country_mean_diff %>%
  mutate(
    abs_diff = abs(mean_diff),
    diff_q4  = ntile(abs_diff, 4),
    diff_q4  = factor(diff_q4, levels = 1:4,
                      labels = c("Q1 (Low |ΔT|)", "Q2", "Q3", "Q4 (High |ΔT|)")),
    diff_sign = ifelse(mean_diff >= 0, "Positive (PWT>UWT)", "Negative (PWT<UWT)")
  )


df_unw<-left_join(df_unw,country_q4,by=c('iso_a3'))

df_pop<-left_join(df_pop,country_q4,by=c('iso_a3'))

table(country_q4$diff_q4)

# 2.1建模 -------------------------------------------------------------------

fit_gam_model <- function(mean_Mean, df){
  gam(log(val) ~ s(mean_Mean, bs = "cs") + s(abs_diff, bs="cs") +  ti(mean_Mean, abs_diff, bs = c("cs", "cs")) + location + year, data = df ,family=quasipoisson(link='log')
  )}

model_pop<-fit_gam_model(df_pop$mean_Mean,df_pop)
model_unw<-fit_gam_model(df_unw$Mean_Temp,df_unw)

summary(model_pop)$s.table
summary(model_unw)$s.table

# 2.2嵌套模型做交互项贡献检验 ---------------------------------------------------------

mod_no_ti<-function(mean_Mean, df){
  gam(
    log(val) ~ s(mean_Mean, bs = "cs") + s(abs_diff, bs="cs") + location + year, data = df ,family=quasipoisson(link='log')
  )}

model_pop_noti<-mod_no_ti(df_pop$mean_Mean,df_pop)
model_unw_noti<-mod_no_ti(df_unw$Mean_Temp,df_unw)

anova(model_pop_noti, model_pop, test = "F")
anova(model_unw_noti, model_unw, test = "F")

# 3.可视化效应修饰 ---------------------------------------------------------------


# 选择 abs_diff 的三个代表水平（10%、50%、90%）
abs_levels <- quantile(df_pop$abs_diff, probs = c(0.10, 0.50, 0.90), na.rm = TRUE)
names(abs_levels) <- c("Low |ΔT| (P10)", "Median |ΔT| (P50)", "High |ΔT| (P90)")

# 温度网格（用 PWT 的 1%~99% 范围，避免极端点）
temp_grid <- seq(
  quantile(df_pop$mean_Mean, 0.01, na.rm = TRUE),
  quantile(df_pop$mean_Mean, 0.99, na.rm = TRUE),
  length.out = 300
)

# newdata：需要包含 mean_Mean, abs_diff, location, year
nd <- expand.grid(
  mean_Mean = temp_grid,
  abs_diff  = as.numeric(abs_levels),
  location  = df_pop$location[1],
  year      = df_pop$year[1]
)

# 预测 log(val) 及其标准误
pr <- predict(model_pop, newdata = nd, se.fit = TRUE)
nd$fit_log <- pr$fit
nd$se_log  <- pr$se.fit

# 在每个 abs_diff 水平内，以该水平的 MMT（fit_log 最小）为参照，构造 RR
rr_df <- as_tibble(nd) %>%
  group_by(abs_diff) %>%
  mutate(
    mmt_log = fit_log[which.min(fit_log)],
    logRR   = fit_log - mmt_log,
    RR      = exp(logRR),
    RR_lo   = exp(logRR - 1.96 * se_log),
    RR_hi   = exp(logRR + 1.96 * se_log),
    abs_group = factor(abs_diff,
                       levels = as.numeric(abs_levels),
                       labels = names(abs_levels))
  ) %>%
  ungroup()

p_em_logrr <- ggplot(rr_df, aes(x = mean_Mean, y = logRR,
                                color = abs_group, fill = abs_group)) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  geom_line(linewidth = 1.1) +
  geom_ribbon(aes(ymin = logRR - 1.96 * se_log,
                  ymax = logRR + 1.96 * se_log),
              alpha = 0.18, color = NA) +
  theme_bw(base_size = 14) +
  labs(
    title = "Effect modification by population–climate mismatch intensity",
    subtitle = "logRR relative to MMT within each |ΔT| level",
    x = "Population-weighted annual mean temperature (PWT)",
    y = "log(RR) vs MMT",
    color = "|ΔT| level",
    fill  = "|ΔT| level"
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

print(p_em_logrr)


# 3.2二维交互风险面 --------------------------------------------------------------


# 2D 网格
temp_grid2 <- seq(
  quantile(df_pop$mean_Mean, 0.01, na.rm = TRUE),
  quantile(df_pop$mean_Mean, 0.99, na.rm = TRUE),
  length.out = 120
)

abs_grid2 <- seq(
  quantile(df_pop$abs_diff, 0.01, na.rm = TRUE),
  quantile(df_pop$abs_diff, 0.99, na.rm = TRUE),
  length.out = 80
)

nd2 <- expand.grid(
  mean_Mean = temp_grid2,
  abs_diff  = abs_grid2,
  location  = df_pop$location[1],
  year      = df_pop$year[1]
)

pr2 <- predict(model_pop, newdata = nd2, se.fit = FALSE)
nd2$fit_log <- as.numeric(pr2)

# 为了可解释：把 risk 标准化成相对最小值（同一张面里）
nd2$logRR <- nd2$fit_log - min(nd2$fit_log, na.rm = TRUE)
nd2$RR    <- exp(nd2$logRR)

p_surface <- ggplot(as_tibble(nd2), aes(x = mean_Mean, y = abs_diff, fill = RR)) +
  geom_raster(interpolate = TRUE) +
  theme_bw(base_size = 14) +
  labs(
    title = "Joint surface of temperature and mismatch intensity",
    subtitle = "RR scaled to the global minimum on the surface",
    x = "Population-weighted annual mean temperature (PWT)",
    y = "|ΔT| = |PWT − UWT|",
    fill = "RR"
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_surface)

