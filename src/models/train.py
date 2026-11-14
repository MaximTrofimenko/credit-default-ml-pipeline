"""Подбор параметров, оценка на тесте и запись в MLflow."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.data.validation import TARGET
from src.models.pipeline import make_pipeline

plt.switch_backend("Agg")


def calculate_metrics(y_true, y_pred, probabilities):
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def save_plots(model, x_test, y_test, reports_dir):
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.savefig(reports_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.savefig(reports_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    estimator = model.named_steps["model"]
    names = model.named_steps["preprocessing"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = abs(estimator.coef_[0])
    top = pd.Series(importance, index=names).nlargest(12)
    top.sort_values().plot.barh(figsize=(8, 5))
    plt.xlabel("Важность признака")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png")
    plt.close()


def train():
    train_data = pd.read_csv("data/processed/train.csv")
    test_data = pd.read_csv("data/processed/test.csv")
    x_train = train_data.drop(columns=[TARGET])
    x_test = test_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    y_test = test_data[TARGET]

    pipeline = make_pipeline(x_train.columns, LogisticRegression(max_iter=1000))
    grid = [
        {"model": [LogisticRegression(max_iter=1000)], "model__C": [0.1, 1, 10]},
        {
            "model": [RandomForestClassifier(n_estimators=50, random_state=42)],
            "model__max_depth": [6, 10],
        },
    ]
    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=3, n_jobs=2)
    search.fit(x_train, y_train)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    model = search.best_estimator_
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, probabilities)
    metrics["best_cv_roc_auc"] = search.best_score_
    with (reports_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    joblib.dump(model, "models/model.joblib")
    save_plots(model, x_test, y_test, reports_dir)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit_default")
    for index, score in enumerate(search.cv_results_["mean_test_score"]):
        params = search.cv_results_["params"][index]
        with mlflow.start_run(run_name=f"experiment_{index + 1}"):
            mlflow.log_param("model", type(params["model"]).__name__)
            for name, value in params.items():
                if name != "model":
                    mlflow.log_param(name, value)
            mlflow.log_metric("cv_roc_auc", float(score))

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("model", type(model.named_steps["model"]).__name__)
        mlflow.log_params(
            {key: value for key, value in search.best_params_.items() if key != "model"}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts("reports", artifact_path="plots_and_metrics")
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_test.head(2),
            serialization_format="cloudpickle",
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
