# MLflow Experiment Tracking - AI Model Comparison

## Architecture Overview

This repository represents the **foundational MLOps layer** of an AI system.

Workflow:
1. Data ingestion using sklearn dataset
2. Training multiple ML models
3. Experiment tracking using MLflow
4. Logging metrics and artifacts (confusion matrices)
5. Storing trained models for comparison

The purpose of this project is to demonstrate **experiment reproducibility and model comparison**, which is a core requirement in production ML systems.
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

## MLflow UI

Below are screenshots from the MLflow UI showing experiment tracking and model comparison.

![Experiments](docs/mlflow-experiments.png)
![Run Details](docs/mlflow-run-details.png)


## Roadmap

Planned enhancements for this project:
- Register best-performing model using MLflow Model Registry
- Promote model to Production stage
- Deploy model as a REST API using FastAPI
- Containerize inference service using Docker
- Add monitoring and alerting

These enhancements will evolve this repository into a **production-grade MLOps system**.


## How to Run

```bash
pip install -r requirements.txt
mlflow ui
cd src
python train.py
