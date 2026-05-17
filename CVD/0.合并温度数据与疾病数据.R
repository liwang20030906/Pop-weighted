#合并所有的温度数据和疾病数据
library(tidyr)
library(dplyr)
library(stringr)
library(readxl)
library(corrplot)
library(openxlsx)
library(ggplot2)
library(countrycode)


# 1.1读取疾病数据和信息数据 ----------------------------------------------------------

Dis_data<-"data/0.GBD/0.GBD_CVD_ASR_00-21_ENG.csv"
Info_data<-"data/2.Info/00.GBD204_SDI_纬度_continent_update.csv"
outputfile<-"data/0.GBD/1.GBD_CVD_ASR_00-21_ENG_with_iso.csv"
outputfile1<-"data/0.GBD/2.GBD_CVD_ASR_00-21_ENG_with_iso191.csv"
outputfile2<-"data/0.Pop_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv"
outputfile3<-"data/0.204_GBD_Temp_CVD_ASR_00-21_ENG_with_iso191.csv"

Dis_data<-read.csv(Dis_data)
Info_data<-read.csv(Info_data,na.strings = c())


# 1.2疾病数据匹配信息数据 -----------------------------------------------------------

country<-Dis_data$location
Dis_data$iso_a3<-countrycode(country, origin = "country.name", destination = "iso3c")

Dis_data<-left_join(Dis_data,Info_data,by='iso_a3')

write.csv(file=outputfile,Dis_data)


# 2.1筛选191个国家 -------------------------------------------------------------

Temp_data<-read.csv("data/1.Temp/Mean_pop_Stroke_SDI_zone_all.csv")
Temp_data204<-read.csv("data/1.Temp/204High_Stroke_SDI_zone_all.csv")

country191<-Temp_data%>%
  dplyr::select('Country','iso_a3')
country191<-as.character(country191$iso_a3)

Dis_data191<-Dis_data%>%
  filter(iso_a3 %in% country191)%>%
  filter(year>=2000&year<=2020)

write.csv(file=outputfile1,Dis_data191)

# 3.1合并疾病数据和温度数据 ----------------------------------------------------------

join_keys <- c("iso_a3", "year")

# 只保留 Temp_data 中“Dis_data191 没有的列” + 两个键
temp_cols_to_add <- setdiff(names(Temp_data), names(Dis_data191))

Temp_data2 <- Temp_data %>%
  select(all_of(join_keys), all_of(temp_cols_to_add))

Dis_Temp_data <- left_join(Dis_data191, Temp_data2, by = join_keys)

write.csv(file=outputfile2,Dis_Temp_data)

# 只保留 Temp_data 中“Dis_data191 没有的列” + 两个键
temp_cols_to_add <- setdiff(names(Temp_data204), names(Dis_data191))

Temp_data2204 <- Temp_data204 %>%
  select(all_of(join_keys), all_of(temp_cols_to_add))

Dis_Temp_data204 <- left_join(Dis_data191, Temp_data2204, by = join_keys)

write.csv(file=outputfile3,Dis_Temp_data204)

