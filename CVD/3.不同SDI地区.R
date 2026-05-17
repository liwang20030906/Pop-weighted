#对不同SDI地区的冷热归因分数进行分层分析

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


# 2.1建模 -------------------------------------------------------------------

prep_data <- function(df){
  df %>%
    filter(year >= 2000 & year <= 2020) %>%
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


# 2.2PAF计算 ----------------------------------------------------------------

compute_paf <- function(model, df_input, temp_var, MRT = NULL,
                        n_boot = 500, seed = 2025) {
  set.seed(seed)
  df_in <- df_input
  temp_vals <- df_in[[temp_var]]
  df_in$mean_Mean <- temp_vals
  df_in$pred <- predict(model, newdata = df_in, type = "response")
  
  
  grid <- data.frame(
    mean_Mean = seq(min(temp_vals, na.rm = TRUE),
                    max(temp_vals, na.rm = TRUE), length.out = 400),
    location = df_in$location[1],
    year = df_in$year[1]
  )
  
  grid$fit <- predict(model, newdata = grid, type = "response")
  MRT <- grid$mean_Mean[which.min(grid$fit)]
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


# 2.3亚组分析 -----------------------------------------------------------------

sdi_levels <- unique(df_pop$SDI)
boot_df_list <- list()
overall_df_list <- list()

for (sdi in sdi_levels) {
  cat("\n====", sdi, "====\n")
  
  df_sdi_unw <- df_unw %>% filter(SDI == sdi)
  df_sdi_pop <- df_pop %>% filter(SDI == sdi)
  
  model_sdi_unw <- fit_gam_model(df_sdi_unw$Mean_Temp, df_sdi_unw)
  model_sdi_pop <- fit_gam_model(df_sdi_pop$mean_Mean, df_sdi_pop)
  
  # plot(model_sdi_unw)
  # plot(model_sdi_pop)
  
  res_unw <- compute_paf(model_sdi_unw, df_sdi_unw, temp_var = "Mean_Temp")
  res_pop <- compute_paf(model_sdi_pop, df_sdi_pop, temp_var = "mean_Mean")
  
  n_boot_unw <- ncol(res_unw$boot_mat)
  
  # === 新增：置信区间计算 ===
  ci_unw <- apply(res_unw$boot_mat, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE) * 100
  ci_pop <- apply(res_pop$boot_mat, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE) * 100
  
  boot_df <- tibble(
    SDI = sdi,
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
  
  # === 新增：合并点估计 + 置信区间 ===
  overall_df <- tibble(
    SDI = sdi,
    Type = rep(c("Unweighted", "Pop-weighted"), each = 3),
    TempType = rep(c("Cold", "Heat", "Total"), times = 2),
    PAF = c(res_unw$PAF_cold, res_unw$PAF_heat, res_unw$PAF_total,
            res_pop$PAF_cold, res_pop$PAF_heat, res_pop$PAF_total) * 100,
    PAF_LCI = c(ci_unw[1, "cold"], ci_unw[1, "heat"], ci_unw[1, "total"],
                ci_pop[1, "cold"], ci_pop[1, "heat"], ci_pop[1, "total"]),
    PAF_UCI = c(ci_unw[2, "cold"], ci_unw[2, "heat"], ci_unw[2, "total"],
                ci_pop[2, "cold"], ci_pop[2, "heat"], ci_pop[2, "total"])
  )
  
  boot_df_list[[sdi]] <- boot_df
  overall_df_list[[sdi]] <- overall_df
}

boot_df_all <- bind_rows(boot_df_list)
overall_df_all <- bind_rows(overall_df_list)

path=paste0(savedir,'2.SDI分层冷热归因结果.csv')
write.csv(overall_df_all,path)

# 3.1可视化PAF ---------------------------------------------------------------

sdi_levels <- c("Low SDI", "Low-middle SDI", "Middle SDI",
                "High-middle SDI", "High SDI")
boot_df_all$SDI <- factor(boot_df_all$SDI, levels = sdi_levels)
overall_df_all$SDI <- factor(overall_df_all$SDI, levels = sdi_levels)

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
  geom_errorbar(data = overall_df_all,   # ← 新增
                aes(x = TempType, ymin = PAF_LCI, ymax = PAF_UCI, group = TempType),
                width = 0.2, color = "black", size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  facet_grid(rows = vars(Type), cols = vars(SDI), scales = "fixed") +
  coord_cartesian(ylim = c(0, 60)) +
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

ggsave(filename = paste0(savedir,"2.SDI_人口加权前后PAF", ".png"),
       plot = p, width = 10, height = 6, dpi = 300)
