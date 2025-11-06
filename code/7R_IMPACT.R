# Install pROC if not installed
# install.packages("pROC")

library(pROC)

# Read the predictions CSV
# data_true <- read.csv("../IMPACT/y_true_composite.csv")
# data_pred <- read.csv("../IMPACT/y_pred_composite.csv")
data_true <- read.csv("../IMPACT/y_true.csv")
data_pred <- read.csv("../IMPACT/y_pred.csv")
# data_true <- read.csv("../IMPACT/y_true_prob.csv")
# data_pred <- read.csv("../IMPACT/y_pred_prob.csv")

# Check the data
# head(data)

model <- c("core", "ext")
model_outcome <- c("mortal", "unfavor")
biomarkers <- c("GFAP", "NFL", "Tau", "UCH.L1", "S100B", "NSE")
composite <- c("composite")
results <- list()
for (m in model) {
  for (o in model_outcome) {
    for (bio in biomarkers) {
      y_true <- paste(m, o, bio, sep = "_")
      y_pred1 <- paste(m, o, bio, "IMPACT", sep = "_")
      y_pred2 <- paste(m, o, bio, "COMBINE", sep = "_")
      # y_pred2 <- paste(m, o, bio, "Cluster", sep = "_")
      print(y_pred1)
      roc1 <- roc(data_true[[y_true]], data_pred[[y_pred1]], quiet = TRUE) # IMPACT model
      roc2 <- roc(data_true[[y_true]], data_pred[[y_pred2]], quiet = TRUE) # IMPACT + cluster model
      test_result <- roc.test(roc2, roc1, method = "delong")

      # Extract AUCs
      auc1 <- auc(roc1)
      auc2 <- auc(roc2)

      # Extract p-value and 95% CI for AUC difference
      pval <- test_result$p.value
      ci <- test_result$conf.int # a 2-element vector

      # Store results in a data frame row
      result_row <- data.frame(
        biomarker = bio,
        model_type = m,
        outcome = o,
        AUC_IMPACT = round(auc1, 2),
        AUC_COMBINE = round(auc2, 2),
        AUC_diff = sprintf("%.2f (%.2f, %.2f)", (auc2 - auc1), ci[1], ci[2]),
        p_value = round(pval, 2),
        AUC_diff_CI_lower = ci[1],
        AUC_diff_CI_upper = ci[2],
        p_value = pval
      )

      # Append to results list
      results[[paste(m, o, bio, sep = "_")]] <- result_row

      # Optionally print result nicely
      cat(sprintf(
        "AUC_IMPACT=%.4f, AUC_COMBINE=%.4f, p=%.4f, 95%%CI=[%.4f, %.4f]\n",
        auc1, auc2, pval, ci[1], ci[2]
      ))

      # plot(roc1, col = "blue", main = paste("ROC -", m, o, bio))
      # plot(roc2, col = "red", add = TRUE)
      # legend("bottomright", legend = c("IMPACT", "Cluster"), col = c("blue", "red"), lwd = 2)
    }
  }
}

# Combine list of data frames into a single data frame
final_results <- do.call(rbind, results)
# write.csv(final_results, "../IMPACT/DeLong_biomarker.csv", row.names = FALSE)
# # Compute ROC objects

# y_true  <- "ext_unfavor_NFL"
# y_pred1 <- "ext_unfavor_NFL_IMPACT"
# y_pred2 <- "core_unfavor_NFL_COMBINE"

# roc1 <- roc(data_true[[y_true]], data_pred[[y_pred1]], quiet=TRUE)  # IMPACT model
# roc2 <- roc(data_true[[y_true]], data_pred[[y_pred2]], quiet=TRUE)  # IMPACT + cluster model

# # Print AUCs
# cat("AUC for IMPACT model:", auc(roc1), "\n")
# cat("AUC for IMPACT + Cluster model:", auc(roc2), "\n")

# # Perform DeLong test to compare AUCs
# test_result <- roc.test(roc1, roc2, method = "delong")

# # Print test result
# print(test_result)
