from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import RANDOM_STATE


def get_classification_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1
        )
    }


def get_regression_models():
    return {
        "random_forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        "lightgbm": LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1
        )
    }