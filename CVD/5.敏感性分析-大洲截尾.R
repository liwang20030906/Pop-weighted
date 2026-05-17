# 对不同大洲截尾验证敏感性

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

savedir<-"result/1.敏感性分析/"

data_pop<-data_pop[,c(6,7,10)]%>%
  rename(popnum=val)

df_unw<-left_join(df_unw,data_pop,by=c('iso_a3','year'))
df_pop<-left_join(df_pop,data_pop,by=c('iso_a3','year'))

prep_data <- function(df){
  df %>%
    dplyr::filter(year >= 2000 & year <= 2020) %>%
    mutate(
      location = as.factor(location),
      year = as.factor(year),
      Continent = as.factor(Continent)
    )
}
df_unw <- prep_data(df_unw)
df_pop <- prep_data(df_pop)

# ============================================================
# 1. 建模
# ============================================================
# ============ 1. 建模（固定效应 + 年份 + 国家）===========
fit_gam_model <- function(mean_Mean, df){
  gam(val ~ s(mean_Mean, bs = "cs") + location + year, data = df , family=quasipoisson(link='log'))
}
model_unw <- fit_gam_model(df_unw$Mean_Temp, df_unw)
model_pop <- fit_gam_model(df_pop$mean_Mean, df_pop)

# ============================================================
# 2. PAF 计算函数（原样保留）
# ============================================================
# ============================================================
compute_paf <- function(model, df_input, temp_var, MRT = NULL,
                        n_boot = 500, seed = 2025) {
  set.seed(seed)
  df_in <- df_input
  
  temp_vals_all<-df_in[[temp_var]]
  t1 <- quantile(temp_vals_all, 0.05, na.rm = TRUE)
  t99 <- quantile(temp_vals_all, 0.95, na.rm = TRUE)
  
  df_in <- df_in %>%
    mutate(mean_Mean = pmin(pmax(.[[temp_var]], t1), t99))
  
  temp_vals <- df_in$mean_Mean
  
  df_in$pred <- predict(model, newdata = df_in, type = "response")
  
  
  grid <- data.frame(
    mean_Mean = seq(t1,t99, length.out = 400),
    location = df_in$location[1],
    year = df_in$year[1]
  )
  
  grid$fit <- predict(model, newdata = grid, type = "response")
  MRT <- grid$mean_Mean[which.min(grid$fit)]
  MRT<-min(max(MRT,t1),t99)
  
  print(MRT)
  
  df_in$pred_ref <- predict(model,
                            newdata = df_in %>% mutate(mean_Mean = MRT),
                            type = "response")
  
  delta_total <- pmax(df_in$pred - df_in$pred_ref, 0)
  PAF_total <- sum(delta_total * df_in$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  df_cold <- df_in %>% filter(mean_Mean < MRT)
  df_heat <- df_in %>% filter(mean_Mean > MRT)
  
  PAF_cold <- sum(pmax(df_cold$pred - df_cold$pred_ref, 0) * df_cold$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  PAF_heat <- sum(pmax(df_heat$pred - df_heat$pred_ref, 0) * df_heat$popnum, na.rm = TRUE) /
    sum(df_in$pred * df_in$popnum, na.rm = TRUE)
  
  boot_mat <- replicate(n_boot, {
    samp <- df_in[sample(1:nrow(df_in), replace = TRUE), ]
    
    samp$pred <- predict(model, newdata = samp, type = "response")
    
    samp$pred_ref <- predict(model,
                             newdata = samp %>% mutate(mean_Mean = MRT),
                             type = "response")
    
    delta <- pmax(samp$pred - samp$pred_ref, 0)
    
    paf_total <- sum(delta * samp$popnum, na.rm = TRUE) / sum(samp$pred * samp$popnum, na.rm = TRUE)
    paf_cold <- sum(delta[samp$mean_Mean < MRT] * samp$popnum[samp$mean_Mean < MRT], na.rm = TRUE) / sum(samp$pred * samp$popnum, na.rm = TRUE)
    paf_heat <- sum(delta[samp$mean_Mean > MRT] * samp$popnum[samp$mean_Mean > MRT], na.rm = TRUE) / sum(samp$pred * samp$popnum, na.rm = TRUE)
    c(total = paf_total, cold = paf_cold, heat = paf_heat)
  })
  
  PAF_LCI <- apply(boot_mat, 1, quantile, 0.025, na.rm = TRUE)
  PAF_UCI <- apply(boot_mat, 1, quantile, 0.975, na.rm = TRUE)
  
  list(
    MRT = MRT,
    PAF_total = PAF_total,
    PAF_cold = PAF_cold,
    PAF_heat = PAF_heat,
    boot_mat = boot_mat,
    PAF_LCI = PAF_LCI,
    PAF_UCI = PAF_UCI
  )
}


# ============================================================
# 3. 分 Continent 计算并生成 boot_df + overall_df（带置信区间）
# ============================================================
continent_levels <- unique(df_pop$Continent)
boot_df_list <- list()
overall_df_list <- list()

for (cont in continent_levels) {
  cat("\n====", cont, "====\n")
  
  df_cont_unw <- df_unw %>% filter(Continent == cont)
  df_cont_pop <- df_pop %>% filter(Continent == cont)
  
  model_cont_unw <- fit_gam_model(df_cont_unw$Mean_Temp, df_cont_unw)
  model_cont_pop <- fit_gam_model(df_cont_pop$mean_Mean, df_cont_pop)
  plot(model_cont_unw)
  plot(model_cont_pop)
  res_unw <- compute_paf(model_cont_unw, df_cont_unw, temp_var = "Mean_Temp")
  res_pop <- compute_paf(model_cont_pop, df_cont_pop, temp_var = "mean_Mean")
  
  n_boot_unw <- ncol(res_unw$boot_mat)
  
  # === 新增置信区间 ===
  ci_unw <- apply(res_unw$boot_mat, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE) * 100
  ci_pop <- apply(res_pop$boot_mat, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE) * 100
  
  boot_df <- tibble(
    Continent = cont,
    Type = rep(rep(c("Unweighted", "Pop-weighted"), each = 3 * n_boot_unw), 1),
    TempType = rep(rep(c("Cold", "Heat", "Total"), each = n_boot_unw), times = 2),
    PAF = c(
      res_unw$boot_mat["cold", ] * 100,
      res_unw$boot_mat["heat", ] * 100,
      res_unw$boot_mat["total", ] * 100,
      res_pop$boot_mat["cold", ] * 100,
      res_pop$boot_mat["heat", ] * 100,
      res_pop$boot_mat["total", ] * 100
    )
  )
  
  overall_df <- tibble(
    Continent = cont,
    Type = rep(c("Unweighted", "Pop-weighted"), each = 3),
    TempType = rep(c("Cold", "Heat", "Total"), times = 2),
    PAF = c(res_unw$PAF_cold, res_unw$PAF_heat, res_unw$PAF_total,
            res_pop$PAF_cold, res_pop$PAF_heat, res_pop$PAF_total) * 100,
    PAF_LCI = c(ci_unw[1, "cold"], ci_unw[1, "heat"], ci_unw[1, "total"],
                ci_pop[1, "cold"], ci_pop[1, "heat"], ci_pop[1, "total"]),
    PAF_UCI = c(ci_unw[2, "cold"], ci_unw[2, "heat"], ci_unw[2, "total"],
                ci_pop[2, "cold"], ci_pop[2, "heat"], ci_pop[2, "total"])
  )
  
  boot_df_list[[cont]] <- boot_df
  overall_df_list[[cont]] <- overall_df
}

boot_df_all <- bind_rows(boot_df_list)
overall_df_all <- bind_rows(overall_df_list)

# ============================================================
# 4. 绘图（添加置信区间误差线）
# ============================================================
continent_levels <- c("SA", "NA", "AS", "AF", "EU", "OC")
boot_df_all$Continent <- factor(boot_df_all$Continent, levels = continent_levels)
overall_df_all$Continent <- factor(overall_df_all$Continent, levels = continent_levels)

cols <- c("Cold" = "#4575b4", "Heat" = "#d73027", "Total" = "grey40")

p <- ggplot(boot_df_all, aes(x = TempType, y = PAF, fill = TempType)) +
  geom_boxplot(alpha = 0.75, width = 0.55, outlier.shape = NA,
               color = "black", position = position_dodge(width = 0.8)) +
  geom_jitter(aes(color = TempType),
              position = position_jitter(width = 0.15, height = 0),
              alpha = 0.25, size = 0.9) +
  geom_point(data = overall_df_all,
             aes(x = TempType, y = PAF, fill = TempType),
             shape = 21, size = 3, color = "black") +
  geom_errorbar(data = overall_df_all,
                aes(x = TempType, ymin = PAF_LCI, ymax = PAF_UCI, group = TempType),
                width = 0.2, color = "black", size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  facet_grid(rows = vars(Type), cols = vars(Continent), scales = "fixed") +
  coord_cartesian(ylim = c(0, 65)) +
  scale_fill_manual(values = cols) +
  scale_color_manual(values = cols) +
  labs(y = "Population Attributable Fraction (PAF, %)", x = "", fill = "", color = "") +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(size = 10, face = "bold"),
    axis.text.y = element_text(size = 11),
    strip.text.x = element_text(size = 12, face = "bold"),
    strip.text.y = element_text(size = 12, face = "bold"),
    plot.margin = margin(10, 10, 10, 10),
    panel.border = element_rect(color = "black", size = 0.8)
  )

print(p)
ggsave(filename = paste0(savedir, "Continent_人口加权前后PAF冷热及all归因分数5-95.png"),
       plot = p, width = 10, height = 6, dpi = 300)