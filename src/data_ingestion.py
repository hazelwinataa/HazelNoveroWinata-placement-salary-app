import pandas as pd


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # improvement features
    df["improvement_degree"] = df["degree_percentage"] - df["hsc_percentage"]
    df["improvement_cgpa"] = (df["cgpa"] * 10) - df["degree_percentage"]

    # skill features
    df["avg_skill"] = (
        df["technical_skill_score"] + df["soft_skill_score"]
    ) / 2

    df["skill_score_total"] = (
        df["technical_skill_score"] + df["soft_skill_score"]
    )

    # academic features
    df["avg_academic"] = (
        df["ssc_percentage"] +
        df["hsc_percentage"] +
        df["degree_percentage"] +
        (df["cgpa"] * 10)
    ) / 4

    df["weighted_score"] = (
        0.2 * df["ssc_percentage"] +
        0.2 * df["hsc_percentage"] +
        0.3 * df["degree_percentage"] +
        0.3 * (df["cgpa"] * 10)
    )

    # experience features
    df["total_experience"] = (
        df["internship_count"] +
        df["live_projects"] +
        df["certifications"] +
        df["work_experience_months"]
    )

    # interaction features
    df["academic_skill"] = (
        df["avg_academic"] * df["avg_skill"]
    ) / 100

    df["skill_experience"] = (
        df["skill_score_total"] *
        (1 + df["internship_count"] + df["live_projects"])
    )

    # employability score
    df["employability_score"] = (
        0.35 * df["technical_skill_score"] +
        0.25 * df["soft_skill_score"] +
        0.15 * df["internship_count"] +
        0.15 * df["live_projects"] +
        0.10 * df["certifications"]
    )

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = create_engineered_features(df)
    return df