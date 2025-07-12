
# Toxic Politics and TikTok Engagement in the 2024 U.S. Election: Code & Replication Materials

This repository contains all analysis and feature extraction code used in the study **“Toxic Politics and TikTok Engagement in the 2024 U.S. Election”**, published in the *Harvard Kennedy School Misinformation Review*. The project investigates how political partisanship, toxicity, and topical content influence user engagement with TikTok videos during the 2024 U.S. presidential election cycle.

---

## 📂 Repository Contents

The repository includes scripts for data processing, statistical modeling, and topic labeling:

### 🧠 Feature Extraction

- `video_feature_extraction.py`  
  Extracts visual and audio features from TikTok videos (e.g., RGB tone, facial expressions, demographics, duration).

- `keyframe_feature_summary.py`  
  Aggregates keyframe-level features (e.g., average emotion, gender, race distributions) into video-level statistics.

### 🗂 Topic Detection

- `topic_assignment.py`  
  Assigns topics to TikTok transcripts using a keyword-matching strategy based on the manually curated keyword list.

- `topic-keywords.csv`  
  Keyword definitions for 22 political topics, based on iterative validation and refinement.

### 📊 Statistical Analyses

- `rq1_mwu_analysis.py`  
  Compares engagement (views and interactions) between partisan and nonpartisan videos using Mann–Whitney U tests with FDR correction.

- `rq2_regression_partisan_toxicity_effects.R`  
  Linear mixed-effects model examining the interaction between toxicity and partisan leaning on user engagement.

- `rq3_regression_topic_partisan_toxicity_effects.R`  
  Mixed-effects model with three-way interactions between toxicity, topic groups, and party alignment.

- `rq4_regression_event_effects.R`  
  Evaluates how toxicity levels influence engagement before and after major political events (e.g., Trump’s conviction).

- `baseline_regression_model_selection.R`  
  Stepwise feature selection (covariate-only) to identify baseline predictors of engagement.

---

## 📁 Data Availability

We collected data using the [TikTok Research API](https://developers.tiktok.com/products/research-api/), following TikTok's Terms of Service. Due to platform restrictions, we are unable to share raw video transcripts or metadata.

However, we have released a dataset containing **TikTok post IDs** to enable replication by researchers with API access. These can be used to retrieve videos and metadata in compliance with TikTok’s data-sharing policies.

---

## 🔧 How to Use

1. Clone the repository:
   ```bash
   gh repo clone picsolab/Toxic-Politics-and-TikTok-Engagement-in-the-2024-U.S.-Election
   cd Toxic-Politics-and-TikTok-Engagement-in-the-2024-U.S.-Election
   ```

2. Install dependencies:
   - Python: See `requirements.txt` (`whisper`, `deepface`, etc.)
   - R: Scripts require `lme4`, `MuMIn`, `bestNormalize`, etc.

3. Run analyses:
   - Use the Python scripts for feature extraction and topic assignment.
   - Use the R scripts to reproduce the regression models in the paper.

---

## 📬 Contact

If you have any questions about the code, data, or analysis, feel free to contact the authors at:

📧 **ahana.biswas@pitt.edu**

---

<!-- ## 📜 Citation

If you use this code or dataset, please cite:

[TBA] -->
