import os
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from optuna.samplers import TPESampler

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PATH,
    REGRESSION_TARGET,
    REGRESSION_NUMERIC_MODEL_1,
    REGRESSION_NUMERIC_MODEL_2,
    REGRESSION_CATEGORICAL,
    TEST_SIZE,
    RANDOM_STATE,
    REGRESSION_RESULTS_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_EXPERIMENT
)

from src.data_ingestion import load_data
from src.model_factory import get_regression_models
from src.preprocessing import build_tree_preprocessor
from src.evaluate import evaluate_regression
from src.mlflow_utils import (
    set_mlflow_experiment,
    log_params,
    log_metrics,
    log_model
)


def get_feature_types(df):
    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_features = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_regression_dataset(df, feature_list, target_column):
    X = df[feature_list].copy()
    y = df[target_column].copy()
    return X, y


def get_grid_param_spaces():
    return {
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        "xgboost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        },
        "lightgbm": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [-1, 5, 10],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.8, 1.0]
        }
    }


def create_model_by_trial(trial, model_name):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

    if model_name == "lightgbm":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 100),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            random_state=RANDOM_STATE,
            verbose=-1
        )

    raise ValueError(f"Unknown model name: {model_name}")


def optuna_objective(trial, model_name, X_train, y_train_log, preprocessor, cv_strategy):
    model = create_model_by_trial(trial, model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train_log,
        cv=cv_strategy,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    return scores.mean()


def run_baseline_models(X_train, X_test, y_train_log, y_test_log, feature_set_name):
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)
    models = get_regression_models()

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"BASELINE | {model_name} | {feature_set_name}")
        print(f"{'=' * 70}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train_log)
        y_pred_log = pipeline.predict(X_test)

        metrics = evaluate_regression(
            model_name=model_name,
            feature_set_name=feature_set_name,
            y_true_log=y_test_log,
            y_pred_log=y_pred_log
        )

        metrics["Tuning_Method"] = "Baseline"
        metrics["Experiment_Name"] = f"{model_name} | {feature_set_name} | Baseline"

        results.append(metrics)
        fitted_models[(model_name, feature_set_name, "Baseline")] = pipeline

        print("RMSE:", round(metrics["RMSE"], 6))
        print("R2  :", round(metrics["R2"], 6))

    return pd.DataFrame(results), fitted_models


def run_grid_search_models(X_train, X_test, y_train_log, y_test_log, feature_set_name):
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)
    models = get_regression_models()
    param_spaces = get_grid_param_spaces()

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"GRID SEARCH | {model_name} | {feature_set_name}")
        print(f"{'=' * 70}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_spaces[model_name],
            cv=cv_strategy,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train_log)

        best_model = grid_search.best_estimator_
        y_pred_log = best_model.predict(X_test)

        metrics = evaluate_regression(
            model_name=model_name,
            feature_set_name=feature_set_name,
            y_true_log=y_test_log,
            y_pred_log=y_pred_log
        )

        metrics["Tuning_Method"] = "GridSearch"
        metrics["Experiment_Name"] = f"{model_name} | {feature_set_name} | GridSearch"
        metrics["Best_Params"] = str(grid_search.best_params_)
        metrics["Best_CV_Score_Neg_RMSE"] = grid_search.best_score_

        results.append(metrics)
        fitted_models[(model_name, feature_set_name, "GridSearch")] = best_model

        print("Best Params:", grid_search.best_params_)
        print("RMSE:", round(metrics["RMSE"], 6))
        print("R2  :", round(metrics["R2"], 6))

    return pd.DataFrame(results), fitted_models


def run_optuna_models(X_train, X_test, y_train_log, y_test_log, feature_set_name, n_trials=20):
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}

    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        print(f"\n{'=' * 70}")
        print(f"OPTUNA | {model_name} | {feature_set_name}")
        print(f"{'=' * 70}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE)
        )

        study.optimize(
            lambda trial: optuna_objective(
                trial, model_name, X_train, y_train_log, preprocessor, cv_strategy
            ),
            n_trials=n_trials,
            show_progress_bar=False
        )

        best_model = create_model_by_trial(
            optuna.trial.FixedTrial(study.best_params),
            model_name
        )

        best_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])

        best_pipeline.fit(X_train, y_train_log)
        y_pred_log = best_pipeline.predict(X_test)

        metrics = evaluate_regression(
            model_name=model_name,
            feature_set_name=feature_set_name,
            y_true_log=y_test_log,
            y_pred_log=y_pred_log
        )

        metrics["Tuning_Method"] = "Optuna"
        metrics["Experiment_Name"] = f"{model_name} | {feature_set_name} | Optuna"
        metrics["Best_Params"] = str(study.best_params)
        metrics["Best_CV_Score_Neg_RMSE"] = study.best_value

        results.append(metrics)
        fitted_models[(model_name, feature_set_name, "Optuna")] = best_pipeline

        print("Best Params:", study.best_params)
        print("RMSE:", round(metrics["RMSE"], 6))
        print("R2  :", round(metrics["R2"], 6))

    return pd.DataFrame(results), fitted_models


def log_regression_result_rows(result_df, fitted_models):
    for _, row in result_df.iterrows():
        run_name = f"{row['Model']}_{row['Feature_Set']}_{row['Tuning_Method']}"

        with mlflow.start_run(run_name=run_name):
            log_params({
                "task": "regression",
                "target": REGRESSION_TARGET,
                "model_name": row["Model"],
                "feature_set": row["Feature_Set"],
                "tuning_method": row["Tuning_Method"],
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE
            })

            run_metrics = {
                "mae": row["MAE"],
                "mse": row["MSE"],
                "rmse": row["RMSE"],
                "r2": row["R2"]
            }

            if "Best_CV_Score_Neg_RMSE" in row and pd.notna(row["Best_CV_Score_Neg_RMSE"]):
                run_metrics["best_cv_score_neg_rmse"] = row["Best_CV_Score_Neg_RMSE"]

            log_metrics(run_metrics)

            model_key = (row["Model"], row["Feature_Set"], row["Tuning_Method"])
            if model_key in fitted_models:
                log_model(fitted_models[model_key], artifact_path="model")


def train_regression_models(n_trials=20):
    df = load_data(DATA_PATH)

    set_mlflow_experiment(REGRESSION_EXPERIMENT)

    df_reg = df[df[REGRESSION_TARGET] > 0].copy()

    feature_list_model_1 = REGRESSION_NUMERIC_MODEL_1 + REGRESSION_CATEGORICAL
    feature_list_model_2 = REGRESSION_NUMERIC_MODEL_2 + REGRESSION_CATEGORICAL

    X_full, y = build_regression_dataset(
        df_reg,
        feature_list_model_1,
        REGRESSION_TARGET
    )

    X_selected, _ = build_regression_dataset(
        df_reg,
        feature_list_model_2,
        REGRESSION_TARGET
    )

    train_idx, test_idx = train_test_split(
        df_reg.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train_full = X_full.loc[train_idx].copy()
    X_test_full = X_full.loc[test_idx].copy()

    X_train_selected = X_selected.loc[train_idx].copy()
    X_test_selected = X_selected.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    all_results = []
    all_models = {}

    feature_set_data = {
        "original_features": (X_train_full, X_test_full),
        "selected_fe_features": (X_train_selected, X_test_selected)
    }

    for feature_set_name, (X_train_fs, X_test_fs) in feature_set_data.items():
        baseline_df, baseline_models = run_baseline_models(
            X_train_fs, X_test_fs, y_train_log, y_test_log, feature_set_name
        )
        log_regression_result_rows(baseline_df, baseline_models)

        grid_df, grid_models = run_grid_search_models(
            X_train_fs, X_test_fs, y_train_log, y_test_log, feature_set_name
        )
        log_regression_result_rows(grid_df, grid_models)

        optuna_df, optuna_models = run_optuna_models(
            X_train_fs, X_test_fs, y_train_log, y_test_log, feature_set_name, n_trials=n_trials
        )
        log_regression_result_rows(optuna_df, optuna_models)

        all_results.extend([baseline_df, grid_df, optuna_df])
        all_models.update(baseline_models)
        all_models.update(grid_models)
        all_models.update(optuna_models)

    results_df = pd.concat(all_results, axis=0, ignore_index=True)
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    os.makedirs(os.path.dirname(REGRESSION_RESULTS_PATH), exist_ok=True)
    results_df.to_csv(REGRESSION_RESULTS_PATH, index=False)

    best_row = results_df.iloc[0]
    best_key = (
        best_row["Model"],
        best_row["Feature_Set"],
        best_row["Tuning_Method"]
    )
    best_pipeline = all_models[best_key]

    os.makedirs(os.path.dirname(REGRESSION_MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, REGRESSION_MODEL_PATH)

    with mlflow.start_run(run_name="regression_summary"):
        log_params({
            "task": "regression_summary",
            "best_model": best_row["Model"],
            "best_feature_set": best_row["Feature_Set"],
            "best_tuning_method": best_row["Tuning_Method"]
        })
        log_metrics({
            "best_rmse": best_row["RMSE"],
            "best_mae": best_row["MAE"],
            "best_r2": best_row["R2"]
        })
        mlflow.log_artifact(REGRESSION_RESULTS_PATH)
        mlflow.log_artifact(REGRESSION_MODEL_PATH)

    print(f"\n{'=' * 70}")
    print("BEST REGRESSION MODEL")
    print(f"{'=' * 70}")
    print(best_row)

    print(f"\nRegression results saved to: {REGRESSION_RESULTS_PATH}")
    print(f"Best regression model saved to: {REGRESSION_MODEL_PATH}")

    return results_df, best_pipeline, best_row