# MLflow Experiment Tracking - AI Model Comparison

## Overview
This project demonstrates experiment tracking using MLflow by training multiple classification models and logging performance metrics, parameters, models, and evaluation artifacts.

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Dataset
Breast Cancer Wisconsin Dataset (from sklearn)

## Tools Used
- Python
- MLflow
- Scikit-Learn
- Seaborn
- Matplotlib

## Features
- Experiment tracking
- Model comparison
- Accuracy logging
- Confusion matrix visualization
- Artifact management
- Model versioning

## How to Run

```bash
pip install -r requirements.txt
mlflow ui
cd src
python train.py
