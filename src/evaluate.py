import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# EVALUATION UNTUK CLASSIFICATION
# ============================================

def evaluate_classification(model_name, feature_set_name, y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    results = {
        "Model": model_name,
        "Feature_Set": feature_set_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision_Class_1": precision,
        "Recall_Class_1": recall,
        "F1_Class_1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }

    return results


# ============================================
# EVALUATION UNTUK REGRESSION
# y_true_log dan y_pred_log masih dalam bentuk log1p
# lalu dikembalikan ke skala asli dengan expm1
# ============================================

def evaluate_regression(model_name, feature_set_name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    results = {
        "Model": model_name,
        "Feature_Set": feature_set_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return results