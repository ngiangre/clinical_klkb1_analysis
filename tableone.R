
# PURPOSE -----------------------------------------------------------------

#' Create tableone for pgd paper v2
#' 



# load libraries ----------------------------------------------------------

library(tidyverse)
library(tableone)

# load data ---------------------------------------------------------------

data_ = read_csv("../../data/integrated_sample_groups_imputed_data_discretized.csv") %>% 
  rename(Group = X1) %>%
  gather(Sample,Value,-Group) %>% 
  spread(Group,Value) %>% 
  select(-NL,-cohort)


# TableOne ----------------------------------------------------------------

dems <- c("Age","Blood_Type","BMI","Donor_Age","Sex","History_Of_Tobacco_Use","Diabetes")
card <- c("Cardiomyopathy","PGD")
tfactors <- c("Ischemic_Time","Mechanical_Support")
hemo <- c("PA_Diastolic","PA_Systolic","PA_Mean","CVP","PCWP")
labs <- c("Creatinine","INR","TBILI","Sodium")
meds <- c("Antiarrhythmic_Use","Beta_Blocker","Prior_Inotrope")
comps <- c("CVP/PCWP","MELD","Radial_Score")
study <- c('Sample',"Cohort","tmt_tag","set")
allVars <- c(study,dems,card,tfactors,hemo,labs,meds,comps)
myVars <- allVars[!allVars %in% study]

catVars <- c("Sex",
             "Blood_Type",
             "History_Of_Tobacco_Use",
             "Diabetes","Cardiomyopathy",
             "PGD","Mechanical_Support",
             "Antiarrhythmic_Use","Beta_Blocker",
             "Prior_Inotrope",'set','tmt_tag',
             'Cohort')
for(c in catVars){
  if(nrow(unique(data_[c]))==2 & c!="Sex"){
    new <- plyr::mapvalues(unlist(data_[c]),from=c("0","1"),to=c("N","Y"))
    data_[c] <- unname(new)
  }
  if(nrow(unique(data_[c]))==2 & c=="Sex"){
    new <- plyr::mapvalues(unlist(data_[c]),from=c("0","1"),to=c("F","M"))
    data_[c] <- unname(new)
  }
  if(c=="tmt_tag"){
    new <- plyr::mapvalues(unlist(data_[c]),from=c("0","1","2","3","4","5","6","7","8","9"),to=c("126","127C","127N","128C","128N","129C","129N","130C","130N","131N"))
    data_[c] <- unname(new)
  }
  if(c=="Blood_Type"){
    new <- plyr::mapvalues(unlist(data_[c]),from=c("0","1","2","3"),to=c("A","AB","B","O"))
    data_[c] <- unname(new)
  }
  if(c=="Cardiomyopathy"){
    cs = sort(unique(readxl::read_xlsx("../../data/Unified.xlsx")$Cardiomyopathy))
    map = list()
    for(i in seq_along(cs)){map[[as.character(i-1)]] = cs[i]}
    mapped <- sapply(data_[c],function(x){map[as.character(x)]})[,1]
    data_[c] <- unname(unlist(mapped))
  }
}

data_$Ischemic <- as.integer(data_$Cardiomyopathy=="Ischemic")
data_$NonIschemic <- as.integer(data_$Cardiomyopathy!="Ischemic")
card <- c(card,"Ischemic","NonIschemic")
allVars <- c(allVars,"Ischemic","NonIschemic")
myVars <- c(myVars,"Ischemic","NonIschemic")
catVars <- c(catVars,"Ischemic","NonIschemic")

contVars <- setdiff(myVars,catVars)
for(c in contVars){data_[c] <- as.numeric(unlist(data_[c]))}

catVars <- c(catVars,'Cohort_Type')

data_[data_["Cohort"]==1,"Cohort_Type"] <- "Prospective"
data_[data_["Cohort"]==0,"Cohort_Type"] <- "Retrospective"
data_[data_["Cohort"]==2,"Cohort_Type"] <- "Retrospective"

data_[data_["Cohort"]==1,"Cohort"] <- "Columbia"
data_[data_["Cohort"]==0,"Cohort"] <- "Cedar"
data_[data_["Cohort"]==2,"Cohort"] <- "Paris"

dtype_map =
  data.frame(
    "var" = colnames(data_),
    "dtype" = ifelse(colnames(data_) %in% catVars, "string","float")
  )
write.csv(dtype_map,"../../data/sample_groups_dtype_map.csv")

ssplit=strsplit(data_$Sample,"-")
data_$Sample_no_tmt_tag <- sapply(ssplit,function(x){if(length(x)>3){paste(x[1:2],collapse = "-")}else{paste(x,collapse = "-")}})

write.csv(data_ %>% 
          select(-Ischemic,-NonIschemic),"../../data/integrated_sample_groups_imputed_data_raw.csv")

data_upatients = data_ %>% 
  select(-Sample,-tmt_tag,-set) %>% 
  unique() %>% 
  arrange(Sample_no_tmt_tag) %>% 
  unique() 

data_upatients %>% 
  select(-Ischemic,-NonIschemic) %>% 
  write.csv("../../data/integrated_sample_groups_imputed_data_raw_unique_patients.csv")

tab <- CreateTableOne(vars=myVars,test=F,data=data_upatients,factorVars = catVars,strata="Cohort")

tab2 <- print(tab, exact = "cohort", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
write.csv(tab2,"../../data/TableOne.csv")


tab <- CreateTableOne(vars=myVars,test=T,data=data_upatients,factorVars = catVars,strata="PGD")

tab2 <- print(tab, exact = "cohort", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
write.csv(tab2,"../../data/TableOne_PGD.csv")

pgd <- as.integer(as.factor(unlist(data_upatients[,c("PGD")])))-1;
pi <- as.integer(as.factor(unlist(data_upatients[,c("Prior_Inotrope")])))-1
cohorts <- as.integer(as.factor(unlist(data_upatients[,c("Cohort")])))-1
tmp=data.frame("PGD"=pgd,"Prior_Inotrope"=pi,"Cohort"=cohorts)
summary(aov(PGD~Prior_Inotrope,tmp))
summary(aov(PGD~Cohort,tmp))
summary(aov(PGD~Prior_Inotrope + Cohort,tmp))


