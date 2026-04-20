# ============================================
# DATA PATH
# ============================================

DATA_PATH = "data/B.csv"

# ============================================
# TARGET
# ============================================

CLASSIFICATION_TARGET = "placement_status"
REGRESSION_TARGET = "salary_package_lpa"

# ============================================
# SPLIT
# ============================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================
# CLASSIFICATION FEATURES
# ============================================

# Model 1: original features only
CLASSIFICATION_NUMERIC_MODEL_1 = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "cgpa",
    "entrance_exam_score",
    "technical_skill_score",
    "soft_skill_score",
    "internship_count",
    "live_projects",
    "certifications",
    "work_experience_months",
    "attendance_percentage",
    "backlogs"
]

# Model 2: original + feature engineering
CLASSIFICATION_NUMERIC_MODEL_2 = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "cgpa",
    "entrance_exam_score",
    "technical_skill_score",
    "soft_skill_score",
    "internship_count",
    "live_projects",
    "certifications",
    "work_experience_months",
    "attendance_percentage",
    "backlogs",
    "avg_academic",
    "weighted_score",
    "improvement_degree",
    "improvement_cgpa",
    "total_experience",
    "avg_skill",
    "academic_skill",
    "skill_experience",
    "employability_score"
]

CLASSIFICATION_CATEGORICAL = [
    "gender",
    "extracurricular_activities"
]

CLASSIFICATION_ALL_FEATURES = list(set(
    CLASSIFICATION_NUMERIC_MODEL_2 + CLASSIFICATION_CATEGORICAL
))

# ============================================
# REGRESSION FEATURES
# ============================================

# Original numerical features
REGRESSION_NUMERIC_ORIGINAL = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "cgpa",
    "entrance_exam_score",
    "technical_skill_score",
    "soft_skill_score",
    "internship_count",
    "live_projects",
    "work_experience_months",
    "certifications",
    "attendance_percentage",
    "backlogs"
]

# Engineered numerical features
REGRESSION_NUMERIC_ENGINEERED = [
    "improvement_degree",
    "improvement_cgpa",
    "avg_skill",
    "academic_skill",
    "skill_experience"
]

REGRESSION_CATEGORICAL = [
    "gender",
    "extracurricular_activities"
]

# Model 1: original only
REGRESSION_NUMERIC_MODEL_1 = REGRESSION_NUMERIC_ORIGINAL

# Model 2: original + engineered
REGRESSION_NUMERIC_MODEL_2 = (
    REGRESSION_NUMERIC_ORIGINAL + REGRESSION_NUMERIC_ENGINEERED
)

REGRESSION_ALL_FEATURES = list(set(
    REGRESSION_NUMERIC_MODEL_2 + REGRESSION_CATEGORICAL
))

# ============================================
# OUTPUT
# ============================================

CLASSIFICATION_RESULTS_PATH = "outputs/classification/classification_results.csv"
REGRESSION_RESULTS_PATH = "outputs/regression/regression_results.csv"

# ============================================
# MODEL SAVE
# ============================================

CLASSIFICATION_MODEL_PATH = "models/best_classification.pkl"
REGRESSION_MODEL_PATH = "models/best_regression.pkl"

# ============================================
# MLFLOW
# ============================================

CLASSIFICATION_EXPERIMENT = "classification_experiment"
REGRESSION_EXPERIMENT = "regression_experiment"