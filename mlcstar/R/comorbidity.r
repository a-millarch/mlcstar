install.packages('comorbidity', lib = '.')
.libPaths("./")
library(comorbidity)

dataset_path <- "data/interim/pre_comorbidity_df.csv"
patients <- read.csv(dataset_path)

# --- Elixhauser ---
elix <- comorbidity(x = patients, id = "PID", code = "Diagnosekode",
                    map = "elixhauser_icd10_quan", assign0 = TRUE)
elix$ASMT_ELIX <- score(elix, weights = "vw", assign0 = TRUE)
# Prefix condition columns
cond_cols_elix <- setdiff(names(elix), c("PID", "ASMT_ELIX"))
names(elix)[names(elix) %in% cond_cols_elix] <- paste0("ELIX_", cond_cols_elix)

# --- Charlson ---
charl <- comorbidity(x = patients, id = "PID", code = "Diagnosekode",
                     map = "charlson_icd10_quan", assign0 = TRUE)
charl$ASMT_CHARLSON <- score(charl, weights = "quan", assign0 = TRUE)
cond_cols_charl <- setdiff(names(charl), c("PID", "ASMT_CHARLSON"))
names(charl)[names(charl) %in% cond_cols_charl] <- paste0("CHAR_", cond_cols_charl)

# --- Merge and write ---
result <- merge(elix, charl, by = "PID")
write.csv(result, "data/interim/computed_comorbidity_df.csv", row.names = FALSE)
