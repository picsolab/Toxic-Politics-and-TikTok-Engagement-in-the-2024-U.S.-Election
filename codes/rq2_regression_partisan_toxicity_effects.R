# -*- coding: utf-8 -*-


# ---------------------------------------------------------------
# RQ2 Model: Interaction Between Toxicity and Partisan Leaning
# ---------------------------------------------------------------
# Description:
# This model tests whether the relationship between toxicity and user
# engagement differs by partisan leaning (D-leaning, R-leaning, Neither).
#
# The outcome variable is the number of interactions (likes + comments + shares),
# and the model includes both main effects and interaction effects for toxicity
# and party affiliation. Random effects account for music, author, and posting day.

# Load required libraries
library(lme4)
library(lmerTest)
library(dplyr)
library(bestNormalize)
library(car)
library(rsq)

# -----------------------
# Data Loading & Filtering
# -----------------------
data <- read.csv("../data.csv", header = TRUE)
data <- data %>%
  filter(words >= 10)

# -----------------------
# Normalize & Scale Inputs
# -----------------------
cols <- c('Interaction', 'RedHue', 'Duration', 'Toxicity', 'Views', 'Age')
for (col in cols) {
  cat("Normalizing:", col, "\n")
  norm_result <- bestNormalize(data[[col]], allow_orderNorm = FALSE)
  data[[col]] <- scale(norm_result$x.t)
}

# -----------------------
# Model Specification: Toxicity × Partisan
# -----------------------
# Includes main effects and interaction term for toxicity × partisan
# Controls for visual, demographic, and linguistic features
tox_model <- lmer(Interaction ~ Toxicity * partisan_leaning +
                     RedHue + Duration + Hedges + Anger + Views + Age +
                     (1 | music_id) + (1 | username) + (1 | days)
                  data = data, na.action = na.exclude)

# -----------------------
# Model Output and Diagnostics
# -----------------------
summary(tox_model)                                  # Coefficients and significance
r2(tox_model)                                       # R² (marginal + conditional)
AIC(tox_model); BIC(tox_model)                      # Model fit
confint(tox_model, level = 0.95, method = "Wald")   # Confidence intervals