import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import plot_confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

mlflow.set_experiment("Cancer_Classification_MLflow_Demo")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = (predictions == y_test).mean()

        # Log params & metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # Confusion matrix
        image_path = plot_confusion_matrix(y_test, predictions, model_name)
        mlflow.log_artifact(image_path)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} Accuracy: {accuracy}")
