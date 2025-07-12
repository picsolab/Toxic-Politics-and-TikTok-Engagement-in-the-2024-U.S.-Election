# -*- coding: utf-8 -*-


# ---------------------------------------------------------------
# Baseline Covariate Model: Feature Selection for Engagement Prediction
# ---------------------------------------------------------------
# Description:
# This script performs baseline modeling to identify key non-topical,
# non-toxic features that predict interaction counts on political TikTok videos.
# The model uses linear mixed-effects (LME) regression and applies stepwise
# backward elimination to identify the most predictive covariates.
#
# Random effects account for variability across user, music, and day.

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
# Normalize & Scale Numeric Covariates
# -----------------------
# Normalize each variable and scale to mean=0, sd=1 for comparability
cols <- c('Interaction', 'RedHue', 'Duration', 'Toxicity', 'Views', 'Age')
for (col in cols) {
  cat("Normalizing:", col, "\n")
  norm_result <- bestNormalize(data[[col]], allow_orderNorm = FALSE)
  data[[col]] <- scale(norm_result$x.t)
}

# -----------------------
# Binarize Psycholinguistic Features
# -----------------------
# Convert continuous linguistic signals into binary indicators
data <- data %>%
  mutate(
    cause = ifelse(Cause > 0, 1, 0),
    posemo = ifelse(Pos_Emotion > 0, 1, 0),
    anger = ifelse(Anger > 0, 1, 0),
    anx   = ifelse(Anxiety > 0, 1, 0)
  )

# -----------------------
# Full Covariate Model
# -----------------------
# Includes all visual, textual, demographic, and linguistic features
full_model <- lmer(Interaction ~ Keyframes_num + RedHue + Duration + FPS + Cause +
                     Generalization + Hedges + Subjectivity + Pos_Emotion + Anxiety + Anger +
                     Views + Proportion_White + Age + Proportion_Angry + Proportion_Sad +
                     Proportion_Happy + Proportion_Fear + Proportion_Man +
                     (1 | music_id) + (1 | username) + (1 | days) +
                     partisan_leaning,
                   data = data, na.action = na.exclude, REML = FALSE)

r2(full_model)

# -----------------------
# Stepwise Backward Elimination (Fixed Effects Only)
# -----------------------
# Iteratively remove covariates that do not significantly contribute
# to predicting interaction (based on likelihood ratio tests)
final_model <- step(full_model,
                    scope = ~ Views + (1 | music_id) + (1 | username) + (1 | days),
                    direction = "backward", test = "Chisq")

# -----------------------
# Final Covariate Model Summary
# -----------------------
# Resulting model includes only covariates that improve model fit
summary(final_model)

# Optionally: manually specify best-performing final model
baseline_model <- lmer(Interaction ~ RedHue + Duration + Hedges + Anger +
                         Views + Age + partisan_leaning +
                         (1 | music_id) + (1 | username) + (1 | days),
                       data = data, na.action = na.exclude)

summary(baseline_model)
r2(baseline_model)
AIC(baseline_model); BIC(baseline_model)
