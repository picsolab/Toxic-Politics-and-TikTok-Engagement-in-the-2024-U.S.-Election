# -*- coding: utf-8 -*-


# ------------------------------------------------------------------
# RQ4: LME Models - Toxicity Effects Pre/Post Significant Events
# ------------------------------------------------------------------
# This script evaluates how toxicity (severe, sexual) influences interaction
# before vs. after a significant event, e.g.,  Trump’s conviction (May 30, 2024).
# We test whether these forms of toxicity show heightened engagement post-event,
# controlling for party leaning and video-level covariates.

# Load required libraries
library(lme4)
library(lmerTest)
library(dplyr)
library(bestNormalize)
library(car)
library(performance)
library(rsq)
library(lubridate)

# -----------------------
# Load and preprocess data
# -----------------------
data <- read.csv("../data.csv", header = TRUE)

# Filter: keep only posts with enough text
data <- data %>% filter(words >= 10)

# -----------------------
# Normalize and scale numeric predictors
# -----------------------
cols <- c('Interaction', 'RedHue', 'Duration', 'Views', 'Age', 'SevereToxicity', 'SexuallyExplicit')

# Normalize using bestNormalize and standardize
for (col in cols) {
  cat("Normalizing:", col, "\n")
  norm_result <- bestNormalize(data[[col]], allow_orderNorm = FALSE)
  data[[col]] <- scale(norm_result$x.t)
}

# -----------------------
# Topic grouping for control variables
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
# Event setup: Trump conviction
# -----------------------
trump_conviction <- as.Date("2024-05-30")

# Function to subset and mark pre/post event period
create_event_subset <- function(df, event_date) {
  df %>%
    filter(created_date >= (event_date - 7) & created_date <= (event_date + 7)) %>%
    mutate(post_event = ifelse(created_date < event_date, 0, 1))
}


# Set baseline party (reference: Neither)
data$partisan_leaning <- factor(data$partisan_leaning)
data$partisan_leaning <- relevel(data$partisan_leaning, ref = "Neither")

# Subset data to 7 days before and after Trump’s conviction
data_conviction <- create_event_subset(data, trump_conviction)

# -----------------------
# Model: Severe Toxicity × Event
# -----------------------
model_severe <- lmer(Interaction ~
                       SevereToxicity * post_event +
                       partisan_leaning +
                       RedHue + Duration + Hedges + Anger + Views + Age +
                       (1 | music_id) + (1 | username),
                     data = data_conviction, na.action = na.exclude)

summary(model_severe)
r2(model_severe)
AIC(model_severe); BIC(model_severe)
vif(model_severe)
confint(model_severe, level = 0.95, method = "Wald")

# -----------------------
# Model: Sexually Explicit × Event
# -----------------------
model_explicit <- lmer(Interaction ~
                       SexuallyExplicit * post_event +
                       partisan_leaning +
                       RedHue + Duration + Hedges + Anger + Views + Age +
                       (1 | music_id) + (1 | username),
                     data = data_conviction, na.action = na.exclude)

summary(model_explicit)
r2(model_explicit)
AIC(model_explicit); BIC(model_explicit)
vif(model_explicit)
confint(model_explicit, level = 0.95, method = "Wald")