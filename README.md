**Name:** Ritik Agrawal  
**Roll No:** DA25M026
# 🤖 Conversational AI Evaluation System (DA5401 Data Challenge)

## 🌟 Overview

This project implements an automated, hybrid evaluation system designed to score the quality of conversational AI agent responses (0-10). The primary technical challenge addressed was the **severe class imbalance** (91% high scores) in the training data, which caused standard models to fail.

Our solution is a **Cascaded Expert Architecture** that successfully captures the rare low-quality responses.

## 🎯 Problem Statement

Given a pair of inputs—a **Metric Definition Embedding** (hidden goal) and a **Prompt-Response Embedding** (the chat)—the system must predict a fitness score (0-10).

### Key Challenges Faced:

* **Extreme Class Imbalance:** 91% of training data was scores 9 or 10.
* **Hidden Semantics:** Metric definitions were provided only as embeddings, requiring strong feature engineering.
* **Multilingual Data:** The system needed to generalize across mixed Indian languages (Tamil, Hindi, etc.) and English.

## 🛠️ Methodology and Architecture

We used a three-stage, expert-based strategy built on LightGBM and Isolation Forest.

### 1. Feature Engineering & PCA

To create a powerful input, we combined raw embeddings with custom **interaction features** (element-wise difference and product) to measure the semantic alignment between the metric goal and the response.

* **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** to the combined feature vector (approx. 3000 dimensions) to retain 95% of the variance, removing noise and speeding up training.

### 2. Cascaded Expert System (The "Team of Experts")

The core of the solution is a routing mechanism that sends data to specialized models based on expected quality. 

| Model Component | Role | Training Data | Key Technique |
| :--- | :--- | :--- | :--- |
| **Gatekeeper** | **Anomaly Detection** | Trained *only* on High Scores (> 7). | **Isolation Forest** |
| **Low Score Expert (LSE)** | Predicts **Anomalies** (0-7). | Data <= 7. | **Inverse Frequency Sample Weighting** |
| **High Score Expert (HSE)** | Predicts **Normal** Scores (> 7). | Data > 7. | **Inverse Frequency Sample Weighting** |

### 3. The Low Score Breakthrough (Sample Weighting)

To force the LSE to predict rare scores (like 0, 1, or 2), we used **Inverse Frequency Sample Weighting** during training. This meant the model paid significantly more attention to a single score of '0' than to a frequent score of '7', effectively overcoming the dataset imbalance for the crucial low-quality range.

## 🧪 Experiments and Results

The cascaded approach successfully stabilized the prediction system against the highly skewed data distribution.

* **Gatekeeper Tuning:** An **Anomaly Threshold**  was tuned to route the most suspicious samples to the LSE.
* **Final Performance:** The combined system achieved an **RMSE of 2.921**, significantly outperforming single-model regression baselines which failed to predict any scores below 8.

## ⚙️ Dependencies

* `Python 3.x`
* `lightgbm`
* `scikit-learn` (for Isolation Forest, PCA, StandardScaler)
* `numpy`
* `pandas`
