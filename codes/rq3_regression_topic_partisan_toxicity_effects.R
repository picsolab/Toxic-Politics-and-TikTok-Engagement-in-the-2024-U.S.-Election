# -*- coding: utf-8 -*-


# ------------------------------------------------------------------
# RQ3: Linear Mixed Effects Model - Topic × Toxicity × Party
# ------------------------------------------------------------------
# Description:
# This model tests how interaction varies across political topics,
# toxicity levels, and party alignment (D-leaning vs. R-leaning) using three-way interactions.
# Nonpartisan posts are excluded due to sparse topical coverage.
# Topic keywords are grouped into broader categories (Table C1, Appendix C).

# Load required packages
library(lme4)
library(lmerTest)
library(dplyr)
library(bestNormalize)
library(car)
library(performance)
library(rsq)

# -----------------------
# Load and prepare data
# -----------------------
data <- read.csv('../data.csv', header = TRUE)


# Filter out short transcripts
data <- data %>%
  filter(words >= 10)

# -----------------------
# Normalize & scale predictors
# -----------------------
cols_to_normalize <- c('Interaction', 'RedHue', 'Duration', 'Toxicity', 'Views', 'Age')

for (col in cols_to_normalize) {
  cat("Normalizing:", col, "\n")
  norm_result <- bestNormalize(data[[col]], allow_orderNorm = FALSE)
  data[[col]] <- scale(norm_result$x.t)
}

# -----------------------
# Define topic groupings
# -----------------------
data$PoliticalFiguresEvents <- rowSums(data[, c(
  "Impeachment", "January 6 Riots", "Project 2025", "MAGA", "Obama",
  "Hunter Biden", "Trump's Assassination Attempt"
)], na.rm = TRUE) > 0

data$SocioCulturalIssues <- rowSums(data[, c(
  "Racism", "Abortion", "Religion", "Socialism", "Gun", "Nazi"
)], na.rm = TRUE) > 0

data$Elections <- rowSums(data[, c("Election 2024", "Election Fraud")], na.rm = TRUE) > 0
data$GeopoliticalConflicts <- rowSums(data[, c("Israel War", "Ukraine War", "Afghanistan")], na.rm = TRUE) > 0
data$Economy <- rowSums(data[, c("Economic issues", "Cyclone Helene")], na.rm = TRUE) > 0

topics <- c("Elections", "Economy", "SocioCulturalIssues", "PoliticalFiguresEvents", "GeopoliticalConflicts", "Immigration", "Labor")
data <- data %>%
  mutate(across(all_of(topics), as.integer))

# -----------------------
# Prepare final analysis subset
# -----------------------
# Keep only partisan-aligned content (exclude 'Neither')
data <- data %>% filter(partisan_leaning %in% c('D-leaning', 'R-leaning'))

# -----------------------
# Fit linear mixed effects model:
# interaction ~ Toxicity * Party * TopicGroup + Covariates + (1|RandomEffects)
# Captures:
# α_G: topic main effects
# α_L: topic × toxicity interaction
# α_N: topic × party interaction
# α_H: topic × toxicity × party interaction

# -----------------------
model_formula <- as.formula(
  paste0(
    "Interaction ~ ",
    paste(paste0("Toxicity * partisan_leaning * ", topics), collapse = " + "),
    " + RedHue + Duration + Hedges + Anger + Views + Age",
    " + (1 | music_id) + (1 | username) + (1 | days)"
  )
)

d_model <- lmer(model_formula, data = data, na.action = na.exclude)

# -----------------------
# Model Diagnostics and Output
# -----------------------
summary(d_model)                                # View fixed effects
r2(d_model)                                     # R² values (marginal, conditional)
AIC(d_model); BIC(d_model)                      # Fit indices
vif(d_model)                                    # Check multicollinearity
confint(d_model, level = 0.95, method = "Wald") # Confidence intervals