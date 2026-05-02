# 🏎️ Can Qualifying Predict Race Outcomes in Formula 1?

## 📊 Project Overview
This repository contains the end-to-end data science pipeline for my **Senior Capstone Project**. The goal of this study is to determine the statistical significance of qualifying positions on final race results in Formula 1, utilizing historical data and machine learning classifiers.

By analyzing track characteristics, driver performance, and session variables, this project predicts whether a driver will finish on the podium based on their starting grid position.

## 🚀 Key Features
* **Machine Learning Models:** Implementation of **XGBoost** and **Random Forest** to classify race outcomes.
* **Interactive Deployment:** A custom **Streamlit** application (`app.py`) for real-time predictions and data exploration.
* **Historical Dashboard:** Integration of the initial project dashboard (developed last semester) providing a retrospective look at seasonal trends.
* **Complete Datasets:** All raw and processed F1 datasets are included in the repository for full reproducibility.

## 🛠️ Handling Class Imbalance
During the modeling phase, we identified a significant class imbalance (podium finishes vs. field results). To ensure the model didn't just "guess" the majority class, we implemented a two-tier strategy:

Cascade Classifier Architecture: We utilized a multi-stage approach to progressively filter out majority-class instances, allowing the final models to focus on the nuances of high-probability podium finishers.

Cost-Sensitive Learning: We adjusted the internal loss functions of the XGBoost and Random Forest models. By assigning a higher penalty to misclassified podium results, we forced the algorithms to prioritize the minority class, significantly improving the Macro-F1 Score and Precision-Recall trade-off.

## 🏁 Final Results
Optimal Model: XGBoost with cost-sensitive tuning.

Key Finding: Qualifying position remains the strongest predictor, but team reliability and track-specific "overtake difficulty" coefficients were essential for high-accuracy predictions.

Deployment: The final model is fully serialized and served via the Streamlit dashboard.
