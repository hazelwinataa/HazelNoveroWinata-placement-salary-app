import mlflow
import mlflow.sklearn


def set_mlflow_experiment(name):
    mlflow.set_experiment(name)


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, float(value))


def log_model(model, artifact_path="model"):
    mlflow.sklearn.log_model(model, artifact_path)