library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(cowplot)
library(readxl)
# Assume results_auc and results_r2, model_labels, model_colors, model_order, and outcome_labels are preloaded

results_auc <- read_excel("./Partial_R2/partial_r2_auc_results.xlsx", sheet = "partial_auc_results")
results_r2 <- read_excel("./Partial_R2/partial_r2_auc_results.xlsx", sheet = "partial_r2_results")

# Define model order and base features
model_order <- c("Univariate", "Model_A", "Model_B", "Model_C", "Model_D")
# core_features <- c("Age", "GCS", "Pupils")
results_auc$Model <- factor(
  results_auc$Model,
  levels = c("Univariate", "Model_A", "Model_B", "Model_C", "Model_D"),
  ordered = TRUE
)
results_r2$Model <- factor(
  results_r2$Model,
  levels = c("Univariate", "Model_A", "Model_B", "Model_C", "Model_D"),
  ordered = TRUE
)
outcome_labels <- c(
  "GOSE_mort" = "Mortality",
  "GOSE_unfav" = "Poor recovery"
)
biomarker_labels <- c(
  "GFAP_cluster" = "GFAP",
  "NFL_cluster" = "NfL",
  "Tau_cluster" = "Tau",
  "UCH.L1_cluster" = "UCH_L1",
  "S100B_cluster" = "S100B",
  "NSE_cluster" = "NSE",
  "composite" = "CB"
)
model_labels <- c(
  "Univariate" = "Univariate ",
  "Model_A" = "IMPACT Core",
  "Model_B" = "IMPACT Core + biomarker ",
  "Model_C" = "IMPACT Extended",
  "Model_D" = "IMPACT Ext + biomarker "
)
model_colors <- c(
  "Univariate" = "#E69F00",
  "Model_A"    = "#E69F00",
  "Model_B"    = "#56B4E9",
  "Model_C"    = "#009E73",
  "Model_D"    = "#CC79A7"
)

# Function to create a plot for a given dataset and outcome
create_metric_plot <- function(results_df, metric_name, outcome, y_limit, show_x_axis = TRUE) {
  df_biomarkers <- results_df %>%
    filter(Outcome == outcome, Variable == Biomarker)

  df_biomarkers$Variable <- biomarker_labels[df_biomarkers$Biomarker]
  df_biomarkers$PlotValue <- as.numeric(df_biomarkers[[metric_name]]) * 100
  df_biomarkers$Variable <- factor(df_biomarkers$Variable, levels = biomarker_labels[unique(df_biomarkers$Biomarker)])
  df_biomarkers$Model <- factor(df_biomarkers$Model, levels = model_order, ordered = TRUE)
  used_models <- intersect(model_order, unique(df_biomarkers$Model))

  p <- ggplot(df_biomarkers, aes(x = Variable, y = PlotValue, fill = Model)) +
    geom_bar(stat = "identity", position = position_dodge(preserve = "single"), width = 0.85) +
    scale_fill_manual(
      values = model_colors[used_models],
      breaks = used_models,
      labels = model_labels[used_models]
    ) +
    scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, y_limit)) +
    labs(
      title = outcome_labels[[outcome]],
      # title = paste(ifelse(metric_name == "Partial_AUC", "Delta AUC", "Delta R²"), "-", outcome_labels[[outcome]]),
      x = if (show_x_axis) "Biomarker" else "",
      y = ifelse(metric_name == "Partial_AUC", "Delta AUC", "R²/ ΔR²"),
      fill = "Model: "
    ) +
    theme_classic() +
    theme(
      axis.title.x = element_text(size = ifelse(show_x_axis, 15, 0)),
      axis.text.x = element_text(size = 13, angle = 45, hjust = 1),
      axis.title.y = element_text(size = 15),
      axis.text.y = element_text(size = 13),
      plot.title = element_text(size = 15),
      legend.position = "none"
    )

  return(p)
}


# Set y limits
max_auc <- max(results_auc %>% filter(Variable == Biomarker) %>% pull(Partial_AUC), na.rm = TRUE) * 100
y_limit_auc <- ceiling(max_auc / 5) * 5
max_r2 <- max(as.numeric(results_r2 %>% filter(Variable == Biomarker) %>% pull(Partial_R2)), na.rm = TRUE) * 100
y_limit_r2 <- ceiling(max_r2 / 5) * 5

# Create all 4 plots
p1 <- create_metric_plot(results_auc, "Partial_AUC", "GOSE_mort", y_limit_auc, show_x_axis = FALSE)
p2 <- create_metric_plot(results_auc, "Partial_AUC", "GOSE_unfav", y_limit_auc, show_x_axis = FALSE)
p3 <- create_metric_plot(results_r2, "Partial_R2", "GOSE_mort", y_limit_r2, show_x_axis = FALSE)
p4 <- create_metric_plot(results_r2, "Partial_R2", "GOSE_unfav", y_limit_r2, show_x_axis = FALSE)


# Extract legend
legend <- get_legend(
  p1 + theme(legend.position = "bottom", legend.title = element_text(size = 14), legend.text = element_text(size = 13), plot.margin = margin(0, 0, 0, 0))
)

# # Combine plots
# final_grid <- plot_grid(p1, p2, p3, p4, nrow = 2) # , labels = c("A", "B", "C", "D"))
# final_plot <- plot_grid(legend, final_grid, ncol = 1, rel_heights = c(0.05, 1))

# # Save plot
# ggsave("combined_auc_r2_grid.png", final_plot, width = 16, height = 10, dpi = 300)

# --- Graph 1: AUC only ---
auc_grid <- plot_grid(p1, p2, nrow = 1) # or ncol = 1 for vertical layout
auc_plot <- plot_grid(auc_grid, legend, ncol = 1, rel_heights = c(1, 0.05))
ggsave("auc_plots.png", auc_plot, width = 14, height = 6, dpi = 300)

# --- Graph 2: R² only ---
r2_grid <- plot_grid(p3, p4, nrow = 1)
r2_plot <- plot_grid(r2_grid, legend, ncol = 1, rel_heights = c(1, 0.05)) + theme(plot.background = element_rect(fill = "white", color = NA))
ggsave("r2_plots.png", r2_plot, width = 14, height = 6, dpi = 300)
