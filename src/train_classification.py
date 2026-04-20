import os
import joblib
import mlflow
import optuna
import pandas as pd

from optuna.samplers import TPESampler

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.config import (
    DATA_PATH,
    CLASSIFICATION_TARGET,
    CLASSIFICATION_NUMERIC_MODEL_1,
    CLASSIFICATION_NUMERIC_MODEL_2,
    CLASSIFICATION_CATEGORICAL,
    CLASSIFICATION_ALL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    CLASSIFICATION_RESULTS_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_EXPERIMENT
)

from src.data_ingestion import load_data
from src.preprocessing import build_scaled_preprocessor, build_tree_preprocessor
from src.evaluate import evaluate_classification
from src.mlflow_utils import (
    set_mlflow_experiment,
    log_params,
    log_metrics,
    log_model
)


f1_scorer = make_scorer(f1_score, pos_label=1)


def create_classification_model(model_name, params=None):
    params = params or {}

    if model_name == "logistic_regression":
        default_params = {
            "random_state": RANDOM_STATE
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    if model_name == "random_forest":
        default_params = {
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    if model_name == "lightgbm":
        default_params = {
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1
        }
        default_params.update(params)
        return LGBMClassifier(**default_params)

    raise ValueError(f"Unknown classification model: {model_name}")


def get_preprocessor_for_model(model_name, numeric_features, categorical_features):
    if model_name == "logistic_regression":
        return build_scaled_preprocessor(numeric_features, categorical_features)

    return build_tree_preprocessor(numeric_features, categorical_features)


def get_grid_param_spaces():
    return {
        "logistic_regression": [
            {
                "model__solver": ["liblinear"],
                "model__penalty": ["l1", "l2"],
                "model__C": [0.001, 0.01, 0.1, 1],
                "model__class_weight": [None, "balanced"],
                "model__max_iter": [1000, 2000],
                "model__tol": [1e-4, 1e-3]
            }
        ],
        "random_forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt"],
            "model__class_weight": [None, "balanced"]
        },
        "lightgbm": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [-1, 8],
            "model__num_leaves": [31, 63],
            "model__min_child_samples": [20, 40],
            "model__class_weight": [None, "balanced"]
        }
    }


def create_logistic_model_by_trial(trial, fixed_params):
    return LogisticRegression(
        solver=fixed_params["solver"],
        penalty=fixed_params["penalty"],
        C=trial.suggest_float("C", fixed_params["c_low"], fixed_params["c_high"], log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        max_iter=trial.suggest_categorical("max_iter", [1000, 2000]),
        tol=trial.suggest_float("tol", fixed_params["tol_low"], fixed_params["tol_high"], log=True),
        random_state=RANDOM_STATE
    )


def create_random_forest_model_by_trial(trial, fixed_params):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int(
            "n_estimators",
            fixed_params["n_estimators_low"],
            fixed_params["n_estimators_high"],
            step=50
        ),
        max_depth=trial.suggest_categorical("max_depth", fixed_params["depth_choices"]),
        min_samples_split=trial.suggest_int(
            "min_samples_split",
            fixed_params["split_low"],
            fixed_params["split_high"]
        ),
        min_samples_leaf=trial.suggest_int(
            "min_samples_leaf",
            fixed_params["leaf_low"],
            fixed_params["leaf_high"]
        ),
        max_features=fixed_params["max_features"],
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_lightgbm_model_by_trial(trial, fixed_params):
    return LGBMClassifier(
        objective="binary",
        n_estimators=trial.suggest_int(
            "n_estimators",
            fixed_params["n_estimators_low"],
            fixed_params["n_estimators_high"],
            step=25
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            fixed_params["learning_rate_low"],
            fixed_params["learning_rate_high"],
            log=True
        ),
        max_depth=trial.suggest_categorical("max_depth", fixed_params["depth_choices"]),
        num_leaves=trial.suggest_int(
            "num_leaves",
            fixed_params["num_leaves_low"],
            fixed_params["num_leaves_high"],
            step=8
        ),
        min_child_samples=trial.suggest_int(
            "min_child_samples",
            fixed_params["min_child_low"],
            fixed_params["min_child_high"],
            step=5
        ),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )


def create_model_by_trial(trial, model_name, fixed_params):
    if model_name == "logistic_regression":
        return create_logistic_model_by_trial(trial, fixed_params)

    if model_name == "random_forest":
        return create_random_forest_model_by_trial(trial, fixed_params)

    if model_name == "lightgbm":
        return create_lightgbm_model_by_trial(trial, fixed_params)

    raise ValueError(f"Unknown classification model for Optuna: {model_name}")


def build_optuna_search_space(model_name, grid_best_params):
    if model_name == "logistic_regression":
        best_c = grid_best_params["model__C"]
        best_tol = grid_best_params["model__tol"]

        return {
            "solver": grid_best_params["model__solver"],
            "penalty": grid_best_params["model__penalty"],
            "c_low": max(best_c / 5, 1e-4),
            "c_high": best_c * 5,
            "tol_low": best_tol / 10,
            "tol_high": best_tol * 10
        }

    if model_name == "random_forest":
        best_n_estimators = grid_best_params["model__n_estimators"]
        best_max_depth = grid_best_params["model__max_depth"]
        best_min_samples_split = grid_best_params["model__min_samples_split"]
        best_min_samples_leaf = grid_best_params["model__min_samples_leaf"]
        best_max_features = grid_best_params["model__max_features"]

        n_estimators_low = max(best_n_estimators - 100, 100)
        n_estimators_high = best_n_estimators + 100

        split_low = max(best_min_samples_split - 1, 2)
        split_high = best_min_samples_split + 3

        leaf_low = max(best_min_samples_leaf - 1, 1)
        leaf_high = best_min_samples_leaf + 2

        if best_max_depth is None:
            depth_choices = [None, 8, 10, 12]
        else:
            numeric_depths = sorted(list(set([
                max(best_max_depth - 2, 3),
                best_max_depth,
                best_max_depth + 2
            ])))
            depth_choices = [None] + numeric_depths

        return {
            "n_estimators_low": n_estimators_low,
            "n_estimators_high": n_estimators_high,
            "depth_choices": depth_choices,
            "split_low": split_low,
            "split_high": split_high,
            "leaf_low": leaf_low,
            "leaf_high": leaf_high,
            "max_features": best_max_features
        }

    if model_name == "lightgbm":
        best_n_estimators = grid_best_params["model__n_estimators"]
        best_learning_rate = grid_best_params["model__learning_rate"]
        best_max_depth = grid_best_params["model__max_depth"]
        best_num_leaves = grid_best_params["model__num_leaves"]
        best_min_child_samples = grid_best_params["model__min_child_samples"]

        n_estimators_low = max(best_n_estimators - 50, 50)
        n_estimators_high = best_n_estimators + 100

        learning_rate_low = max(best_learning_rate / 2, 0.01)
        learning_rate_high = min(best_learning_rate * 2, 0.3)

        if best_max_depth == -1:
            depth_choices = [-1, 6, 8, 10]
        else:
            depth_choices = sorted(list(set([
                -1,
                max(best_max_depth - 2, 3),
                best_max_depth,
                best_max_depth + 2
            ])))

        num_leaves_low = max(best_num_leaves - 16, 15)
        num_leaves_high = best_num_leaves + 32

        min_child_low = max(best_min_child_samples - 10, 5)
        min_child_high = best_min_child_samples + 20

        return {
            "n_estimators_low": n_estimators_low,
            "n_estimators_high": n_estimators_high,
            "learning_rate_low": learning_rate_low,
            "learning_rate_high": learning_rate_high,
            "depth_choices": depth_choices,
            "num_leaves_low": num_leaves_low,
            "num_leaves_high": num_leaves_high,
            "min_child_low": min_child_low,
            "min_child_high": min_child_high
        }

    raise ValueError(f"Unknown model name when building Optuna space: {model_name}")


def run_advanced_classification_experiment(
    model_name,
    feature_set_name,
    numeric_features,
    categorical_features,
    X_train,
    X_test,
    y_train,
    y_test,
    cv,
    n_trials=20
):
    print(f"\n{'=' * 70}")
    print(f"START EXPERIMENT: {model_name} | {feature_set_name}")
    print(f"{'=' * 70}")

    X_train_sub = X_train[numeric_features + categorical_features].copy()
    X_test_sub = X_test[numeric_features + categorical_features].copy()

    preprocessor = get_preprocessor_for_model(
        model_name,
        numeric_features,
        categorical_features
    )

    base_model = create_classification_model(model_name)

    base_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    param_grid = get_grid_param_spaces()[model_name]

    print(f"\n==== START GRID SEARCH ({model_name}) ====")

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid_search.fit(X_train_sub, y_train)

    print(f"\n==== BEST PARAMS (GRID SEARCH) ====")
    print("Best CV F1:", round(grid_search.best_score_, 6))
    print("Best Params:", grid_search.best_params_)

    fixed_params = build_optuna_search_space(model_name, grid_search.best_params_)

    def objective(trial):
        model = create_model_by_trial(trial, model_name, fixed_params)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            estimator=pipeline,
            X=X_train_sub,
            y=y_train,
            cv=cv,
            scoring=f1_scorer,
            n_jobs=-1
        )

        return scores.mean()

    print(f"\n==== START OPTUNA ({model_name}) ====")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    print(f"\n==== BEST PARAMS (OPTUNA) ====")
    print("Best CV F1:", round(study.best_value, 6))
    print("Best Params:", study.best_params)

    final_model = create_model_by_trial(
        optuna.trial.FixedTrial(study.best_params),
        model_name,
        fixed_params
    )

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", final_model)
    ])

    final_pipeline.fit(X_train_sub, y_train)

    y_pred = final_pipeline.predict(X_test_sub)
    y_proba = final_pipeline.predict_proba(X_test_sub)[:, 1]

    metrics = evaluate_classification(
        model_name=model_name,
        feature_set_name=feature_set_name,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )

    metrics["Grid_Best_F1"] = grid_search.best_score_
    metrics["Grid_Best_Params"] = str(grid_search.best_params_)
    metrics["Optuna_Best_F1"] = study.best_value
    metrics["Optuna_Best_Params"] = str(study.best_params)
    metrics["Tuning_Method"] = "GridSearch+Optuna"

    print(f"\n==== FINAL TEST METRICS ====")
    print("Accuracy          :", round(metrics["Accuracy"], 6))
    print("Balanced Accuracy :", round(metrics["Balanced_Accuracy"], 6))
    print("Precision Class 1 :", round(metrics["Precision_Class_1"], 6))
    print("Recall Class 1    :", round(metrics["Recall_Class_1"], 6))
    print("F1 Class 1        :", round(metrics["F1_Class_1"], 6))
    print("ROC AUC           :", round(metrics["ROC_AUC"], 6))
    print("PR AUC            :", round(metrics["PR_AUC"], 6))

    return metrics, final_pipeline


def train_classification_models_advanced(n_trials=20):
    df = load_data(DATA_PATH)

    set_mlflow_experiment(CLASSIFICATION_EXPERIMENT)

    X = df[CLASSIFICATION_ALL_FEATURES].copy()
    y = df[CLASSIFICATION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    feature_sets = {
        "original_features": CLASSIFICATION_NUMERIC_MODEL_1,
        "with_feature_engineering": CLASSIFICATION_NUMERIC_MODEL_2
    }

    model_names = [
        "logistic_regression",
        "random_forest",
        "lightgbm"
    ]

    all_results = []
    all_pipelines = {}

    for feature_set_name, numeric_features in feature_sets.items():
        for model_name in model_names:
            with mlflow.start_run(run_name=f"{model_name}_{feature_set_name}"):

                metrics, pipeline = run_advanced_classification_experiment(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    numeric_features=numeric_features,
                    categorical_features=CLASSIFICATION_CATEGORICAL,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    cv=cv,
                    n_trials=n_trials
                )

                log_params({
                    "task": "classification",
                    "target": CLASSIFICATION_TARGET,
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "n_numeric_features": len(numeric_features),
                    "n_categorical_features": len(CLASSIFICATION_CATEGORICAL),
                    "n_trials": n_trials,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE
                })

                log_metrics({
                    "accuracy": metrics["Accuracy"],
                    "balanced_accuracy": metrics["Balanced_Accuracy"],
                    "precision_class_1": metrics["Precision_Class_1"],
                    "recall_class_1": metrics["Recall_Class_1"],
                    "f1_class_1": metrics["F1_Class_1"],
                    "roc_auc": metrics["ROC_AUC"],
                    "pr_auc": metrics["PR_AUC"],
                    "grid_best_f1": metrics["Grid_Best_F1"],
                    "optuna_best_f1": metrics["Optuna_Best_F1"]
                })

                log_model(pipeline, artifact_path="model")

                all_results.append(metrics)
                all_pipelines[(model_name, feature_set_name)] = pipeline

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by="F1_Class_1",
        ascending=False
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(CLASSIFICATION_RESULTS_PATH), exist_ok=True)
    results_df.to_csv(CLASSIFICATION_RESULTS_PATH, index=False)

    best_row = results_df.iloc[0]
    best_key = (best_row["Model"], best_row["Feature_Set"])
    best_pipeline = all_pipelines[best_key]

    os.makedirs(os.path.dirname(CLASSIFICATION_MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, CLASSIFICATION_MODEL_PATH)

    with mlflow.start_run(run_name="classification_summary"):
        log_params({
            "task": "classification_summary",
            "best_model": best_row["Model"],
            "best_feature_set": best_row["Feature_Set"]
        })
        log_metrics({
            "best_f1_class_1": best_row["F1_Class_1"],
            "best_accuracy": best_row["Accuracy"],
            "best_roc_auc": best_row["ROC_AUC"]
        })
        mlflow.log_artifact(CLASSIFICATION_RESULTS_PATH)
        mlflow.log_artifact(CLASSIFICATION_MODEL_PATH)

    print(f"\n{'=' * 70}")
    print("BEST CLASSIFICATION MODEL")
    print(f"{'=' * 70}")
    print(best_row)

    print(f"\nClassification results saved to: {CLASSIFICATION_RESULTS_PATH}")
    print(f"Best classification model saved to: {CLASSIFICATION_MODEL_PATH}")

    return results_df, best_pipeline, best_row