import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Placement & Salary Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONSTANTS
# =========================================================
FEATURE_COLUMNS = [
    "gender",
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
    "backlogs",
    "extracurricular_activities",
]

ENGINEERED_FEATURE_COLUMNS = [
    "avg_skill",
    "academic_skill",
    "improvement_cgpa",
    "improvement_degree",
    "skill_experience",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

CLASSIFICATION_TARGET = "placement_status"
REGRESSION_TARGET = "salary_package_lpa"

PLACEMENT_LABELS = {
    0: "Not Placed",
    1: "Placed",
}

BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_MODEL_PATH = BASE_DIR / "models" / "best_classification.pkl"
REGRESSION_MODEL_PATH = BASE_DIR / "models" / "best_regression.pkl"

DATA_CANDIDATES = [
    BASE_DIR / "data" / "B.csv",
    BASE_DIR.parent / "data" / "B.csv",
    BASE_DIR / "B.csv",
]


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .sub-text {
            font-size: 1rem;
            color: #A0AEC0;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .card {
            background-color: #111827;
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .small-note {
            color: #94A3B8;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
@st.cache_resource
def load_classification_model():
    if not CLASSIFICATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classification model not found: {CLASSIFICATION_MODEL_PATH}"
        )
    return joblib.load(CLASSIFICATION_MODEL_PATH)


@st.cache_resource
def load_regression_model():
    if not REGRESSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Regression model not found: {REGRESSION_MODEL_PATH}"
        )
    return joblib.load(REGRESSION_MODEL_PATH)


@st.cache_data
def load_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_csv(path), path
    return None, None


def get_default_values(dataframe: pd.DataFrame | None) -> dict:
    if dataframe is None:
        return {
            "gender": "Male",
            "ssc_percentage": 70.0,
            "hsc_percentage": 72.0,
            "degree_percentage": 75.0,
            "cgpa": 7.5,
            "entrance_exam_score": 68.0,
            "technical_skill_score": 72.0,
            "soft_skill_score": 70.0,
            "internship_count": 1,
            "live_projects": 1,
            "work_experience_months": 6,
            "certifications": 1,
            "attendance_percentage": 85.0,
            "backlogs": 0,
            "extracurricular_activities": "Yes",
        }

    return {
        "gender": dataframe["gender"].mode().iloc[0],
        "ssc_percentage": float(dataframe["ssc_percentage"].median()),
        "hsc_percentage": float(dataframe["hsc_percentage"].median()),
        "degree_percentage": float(dataframe["degree_percentage"].median()),
        "cgpa": float(dataframe["cgpa"].median()),
        "entrance_exam_score": float(dataframe["entrance_exam_score"].median()),
        "technical_skill_score": float(dataframe["technical_skill_score"].median()),
        "soft_skill_score": float(dataframe["soft_skill_score"].median()),
        "internship_count": int(dataframe["internship_count"].median()),
        "live_projects": int(dataframe["live_projects"].median()),
        "work_experience_months": int(dataframe["work_experience_months"].median()),
        "certifications": int(dataframe["certifications"].median()),
        "attendance_percentage": float(dataframe["attendance_percentage"].median()),
        "backlogs": int(dataframe["backlogs"].median()),
        "extracurricular_activities": dataframe["extracurricular_activities"].mode().iloc[0],
    }


def build_input_dataframe(input_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    engineered_df = df.copy()

    engineered_df["avg_skill"] = (
        engineered_df["technical_skill_score"] + engineered_df["soft_skill_score"]
    ) / 2

    engineered_df["academic_skill"] = (
        engineered_df["ssc_percentage"]
        + engineered_df["hsc_percentage"]
        + engineered_df["degree_percentage"]
    ) / 3

    engineered_df["improvement_cgpa"] = (
        engineered_df["cgpa"] - (engineered_df["degree_percentage"] / 10)
    )

    engineered_df["improvement_degree"] = (
        engineered_df["degree_percentage"] - engineered_df["hsc_percentage"]
    )

    engineered_df["skill_experience"] = (
        engineered_df["internship_count"]
        + engineered_df["live_projects"]
        + engineered_df["certifications"]
    )

    return engineered_df


def format_placement_output(prediction_value):
    try:
        prediction_int = int(prediction_value)
        return PLACEMENT_LABELS.get(prediction_int, str(prediction_value))
    except Exception:
        return str(prediction_value)


def get_probability_dataframe(model, input_df: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_names = [
            PLACEMENT_LABELS.get(int(cls), str(cls))
            if str(cls).isdigit() or isinstance(cls, (int, np.integer))
            else str(cls)
            for cls in model.classes_
        ]
        return pd.DataFrame({
            "Class": class_names,
            "Probability": probabilities,
        })
    return None


def create_salary_band(predicted_salary: float, dataframe: pd.DataFrame | None) -> str:
    if dataframe is None:
        if predicted_salary < 5:
            return "Low"
        if predicted_salary < 9:
            return "Medium"
        return "High"

    placed_df = dataframe[dataframe[CLASSIFICATION_TARGET] == 1].copy()
    if placed_df.empty:
        return "N/A"

    q1 = placed_df[REGRESSION_TARGET].quantile(0.25)
    q3 = placed_df[REGRESSION_TARGET].quantile(0.75)

    if predicted_salary < q1:
        return "Below Typical Placed Range"
    if predicted_salary > q3:
        return "Above Typical Placed Range"
    return "Within Typical Placed Range"


def safe_salary_prediction(model, input_df: pd.DataFrame) -> float:
    prediction = float(model.predict(input_df)[0])
    return max(prediction, 0.0)


def add_metric_cards(dataframe: pd.DataFrame | None):
    col1, col2, col3, col4 = st.columns(4)

    if dataframe is None:
        col1.metric("Total Records", "-")
        col2.metric("Placement Rate", "-")
        col3.metric("Avg Salary (Overall)", "-")
        col4.metric("Avg Salary (Placed)", "-")
        return

    placement_rate = dataframe[CLASSIFICATION_TARGET].mean() * 100
    avg_salary_overall = dataframe[REGRESSION_TARGET].mean()

    placed_df = dataframe[dataframe[CLASSIFICATION_TARGET] == 1]
    avg_salary_placed = placed_df[REGRESSION_TARGET].mean() if not placed_df.empty else 0.0

    col1.metric("Total Records", f"{len(dataframe):,}")
    col2.metric("Placement Rate", f"{placement_rate:.2f}%")
    col3.metric("Avg Salary (Overall)", f"{avg_salary_overall:.2f} LPA")
    col4.metric("Avg Salary (Placed)", f"{avg_salary_placed:.2f} LPA")


def render_dataset_visuals(dataframe: pd.DataFrame | None):
    if dataframe is None:
        st.info("Dataset preview tidak tersedia karena file B.csv belum ditemukan di folder deployment/data atau data root.")
        return

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        placement_map_df = dataframe.copy()
        placement_map_df["placement_label"] = placement_map_df[CLASSIFICATION_TARGET].map({
            0: "Not Placed",
            1: "Placed",
        })

        placement_count_df = (
            placement_map_df["placement_label"]
            .value_counts()
            .reset_index()
        )
        placement_count_df.columns = ["Placement Status", "Count"]

        fig_placement = px.bar(
            placement_count_df,
            x="Placement Status",
            y="Count",
            title="Placement Status Distribution",
            text_auto=True,
        )
        fig_placement.update_layout(height=420)
        st.plotly_chart(fig_placement, use_container_width=True)

    with chart_col2:
        placed_df = dataframe[dataframe[CLASSIFICATION_TARGET] == 1].copy()

        if not placed_df.empty:
            fig_salary = px.histogram(
                placed_df,
                x=REGRESSION_TARGET,
                nbins=30,
                title="Salary Distribution for Placed Students",
            )
            fig_salary.update_layout(
                xaxis_title="Salary Package (LPA)",
                yaxis_title="Count",
                height=420,
            )
            st.plotly_chart(fig_salary, use_container_width=True)
        else:
            st.warning("Tidak ada data placed students untuk salary visualization.")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig_scatter = px.scatter(
            dataframe,
            x="cgpa",
            y="technical_skill_score",
            color=dataframe[CLASSIFICATION_TARGET].map({0: "Not Placed", 1: "Placed"}),
            title="CGPA vs Technical Skill Score",
            labels={"color": "Placement Status"},
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_col4:
        numeric_columns = [
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
            "backlogs",
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET,
        ]

        corr_df = dataframe[numeric_columns].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# LOAD ASSETS
# =========================================================
try:
    classification_model = load_classification_model()
    regression_model = load_regression_model()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()

dataset_df, dataset_path = load_dataset()
default_values = get_default_values(dataset_df)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🎓 Prediction Panel")
    st.markdown(
        "Gunakan panel ini untuk memilih task, melihat info dataset, dan menjalankan prediksi."
    )

    prediction_task = st.radio(
        "Select Task",
        ["Classification", "Regression"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ℹ️ App Features")
    st.markdown(
        """
        - Prediksi **placement status**
        - Prediksi **salary package**
        - Input melalui **form**
        - Visualisasi **probability** dan **dataset**
        - Tampilan siap deploy ke **Streamlit Cloud**
        """
    )

    st.markdown("---")
    st.markdown("### 📁 Resource Status")
    st.write(f"Classification Model: {'✅ Loaded' if classification_model is not None else '❌ Missing'}")
    st.write(f"Regression Model: {'✅ Loaded' if regression_model is not None else '❌ Missing'}")
    st.write(f"Dataset B.csv: {'✅ Loaded' if dataset_df is not None else '⚠️ Optional / Not Found'}")

    if dataset_path:
        st.caption(f"Dataset source: {dataset_path}")

    st.markdown("---")
    st.markdown("### 🧾 Feature Checklist")
    for column_name in ALL_FEATURE_COLUMNS:
        st.caption(f"• {column_name}")


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">Student Placement & Salary Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Aplikasi monolithic berbasis Streamlit untuk memprediksi status placement dan salary package mahasiswa menggunakan model hasil tahap pipeline.</div>',
    unsafe_allow_html=True,
)

add_metric_cards(dataset_df)

tab_prediction, tab_visual, tab_about = st.tabs(
    ["🔮 Prediction Center", "📊 Dataset Overview", "📘 Model Guide"]
)

# =========================================================
# TAB 1: PREDICTION CENTER
# =========================================================
with tab_prediction:
    st.markdown('<div class="section-title">Input Form</div>', unsafe_allow_html=True)
    st.caption("Isi seluruh fitur berikut agar konsisten dengan fitur saat training model.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0 if default_values["gender"] == "Male" else 1,
            )
            ssc_percentage = st.number_input(
                "SSC Percentage",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["ssc_percentage"]),
                step=0.1,
            )
            hsc_percentage = st.number_input(
                "HSC Percentage",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["hsc_percentage"]),
                step=0.1,
            )
            degree_percentage = st.number_input(
                "Degree Percentage",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["degree_percentage"]),
                step=0.1,
            )
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=float(default_values["cgpa"]),
                step=0.1,
            )

        with col2:
            entrance_exam_score = st.number_input(
                "Entrance Exam Score",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["entrance_exam_score"]),
                step=0.1,
            )
            technical_skill_score = st.number_input(
                "Technical Skill Score",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["technical_skill_score"]),
                step=0.1,
            )
            soft_skill_score = st.number_input(
                "Soft Skill Score",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["soft_skill_score"]),
                step=0.1,
            )
            internship_count = st.number_input(
                "Internship Count",
                min_value=0,
                max_value=50,
                value=int(default_values["internship_count"]),
                step=1,
            )
            live_projects = st.number_input(
                "Live Projects",
                min_value=0,
                max_value=50,
                value=int(default_values["live_projects"]),
                step=1,
            )

        with col3:
            work_experience_months = st.number_input(
                "Work Experience Months",
                min_value=0,
                max_value=240,
                value=int(default_values["work_experience_months"]),
                step=1,
            )
            certifications = st.number_input(
                "Certifications",
                min_value=0,
                max_value=50,
                value=int(default_values["certifications"]),
                step=1,
            )
            attendance_percentage = st.number_input(
                "Attendance Percentage",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["attendance_percentage"]),
                step=0.1,
            )
            backlogs = st.number_input(
                "Backlogs",
                min_value=0,
                max_value=50,
                value=int(default_values["backlogs"]),
                step=1,
            )
            extracurricular_activities = st.selectbox(
                "Extracurricular Activities",
                options=["Yes", "No"],
                index=0 if default_values["extracurricular_activities"] == "Yes" else 1,
            )

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

    if submitted:
        input_dict = {
            "gender": gender,
            "ssc_percentage": ssc_percentage,
            "hsc_percentage": hsc_percentage,
            "degree_percentage": degree_percentage,
            "cgpa": cgpa,
            "entrance_exam_score": entrance_exam_score,
            "technical_skill_score": technical_skill_score,
            "soft_skill_score": soft_skill_score,
            "internship_count": internship_count,
            "live_projects": live_projects,
            "work_experience_months": work_experience_months,
            "certifications": certifications,
            "attendance_percentage": attendance_percentage,
            "backlogs": backlogs,
            "extracurricular_activities": extracurricular_activities,
        }

        input_df = build_input_dataframe(input_dict)
        input_df = apply_feature_engineering(input_df)

        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(input_df, use_container_width=True)

        try:
            if prediction_task == "Classification":
                raw_prediction = classification_model.predict(input_df)[0]
                prediction_label = format_placement_output(raw_prediction)

                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    if prediction_label.lower() == "placed":
                        st.success(f"Predicted Placement Status: {prediction_label}")
                    else:
                        st.warning(f"Predicted Placement Status: {prediction_label}")

                with result_col2:
                    estimated_salary = safe_salary_prediction(regression_model, input_df)
                    st.info(f"Estimated Salary Package: {estimated_salary:.2f} LPA")

                probability_df = get_probability_dataframe(classification_model, input_df)

                if probability_df is not None:
                    st.markdown('<div class="section-title">Prediction Probability</div>', unsafe_allow_html=True)

                    fig_probability = px.bar(
                        probability_df,
                        x="Class",
                        y="Probability",
                        text_auto=".2f",
                        title="Placement Probability",
                    )
                    fig_probability.update_layout(height=420)
                    st.plotly_chart(fig_probability, use_container_width=True)

                st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
                st.write(
                    f"""
                    Model klasifikasi memprediksi bahwa mahasiswa ini memiliki status **{prediction_label}**.
                    Selain itu, model regresi memperkirakan **salary package** sekitar **{estimated_salary:.2f} LPA**.
                    """
                )

            else:
                predicted_salary = safe_salary_prediction(regression_model, input_df)
                salary_band = create_salary_band(predicted_salary, dataset_df)

                result_col1, result_col2 = st.columns(2)
                result_col1.success(f"Predicted Salary Package: {predicted_salary:.2f} LPA")
                result_col2.info(f"Salary Band: {salary_band}")

                placement_probability_df = get_probability_dataframe(classification_model, input_df)
                placement_label = format_placement_output(classification_model.predict(input_df)[0])

                if placement_probability_df is not None:
                    st.markdown('<div class="section-title">Placement Probability</div>', unsafe_allow_html=True)

                    fig_probability = px.bar(
                        placement_probability_df,
                        x="Class",
                        y="Probability",
                        text_auto=".2f",
                        title="Classification Probability for the Same Input",
                    )
                    fig_probability.update_layout(height=420)
                    st.plotly_chart(fig_probability, use_container_width=True)

                if dataset_df is not None:
                    placed_df = dataset_df[dataset_df[CLASSIFICATION_TARGET] == 1].copy()
                    if not placed_df.empty:
                        comparison_df = pd.DataFrame({
                            "Category": [
                                "Predicted Salary",
                                "Placed Students Mean Salary",
                                "Placed Students Median Salary",
                            ],
                            "Value": [
                                predicted_salary,
                                placed_df[REGRESSION_TARGET].mean(),
                                placed_df[REGRESSION_TARGET].median(),
                            ],
                        })

                        st.markdown('<div class="section-title">Salary Comparison</div>', unsafe_allow_html=True)
                        fig_comparison = px.bar(
                            comparison_df,
                            x="Category",
                            y="Value",
                            text_auto=".2f",
                            title="Predicted Salary vs Dataset Salary Benchmarks",
                        )
                        fig_comparison.update_layout(
                            xaxis_title="Category",
                            yaxis_title="Salary Package (LPA)",
                            height=450,
                        )
                        st.plotly_chart(fig_comparison, use_container_width=True)

                st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
                st.write(
                    f"""
                    Model regresi memperkirakan **salary package** sebesar **{predicted_salary:.2f} LPA**.
                    Untuk input yang sama, model klasifikasi memprediksi status placement **{placement_label}**.
                    """
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# =========================================================
# TAB 2: DATASET OVERVIEW
# =========================================================
with tab_visual:
    st.markdown('<div class="section-title">Dataset Overview & Visualization</div>', unsafe_allow_html=True)
    st.caption("Visualisasi ini dibuat untuk mempercantik tampilan aplikasi sekaligus memberi konteks data kepada pengguna.")
    render_dataset_visuals(dataset_df)

# =========================================================
# TAB 3: MODEL GUIDE
# =========================================================
with tab_about:
    st.markdown('<div class="section-title">About This App</div>', unsafe_allow_html=True)
    st.write(
        """
        Aplikasi ini adalah **monolithic deployment** berbasis Streamlit.
        Artinya, antarmuka pengguna dan logika prediksi dijalankan dalam satu aplikasi yang sama.
        """
    )

    st.markdown("### 🎯 Prediction Targets")
    st.markdown(
        f"""
        - **Classification target**: `{CLASSIFICATION_TARGET}`
        - **Regression target**: `{REGRESSION_TARGET}`
        """
    )

    st.markdown("### 🧩 Features Used")
    st.code("\n".join(ALL_FEATURE_COLUMNS), language="text")

    st.markdown("### 🛠️ Technical Notes")
    st.markdown(
        """
        - Model dimuat dari file `.pkl`
        - Input prediksi dibentuk menjadi `pandas.DataFrame`
        - Feature engineering diterapkan sebelum proses prediksi
        - App menampilkan hasil prediksi, probabilitas, dan visualisasi dataset
        - Visualisasi dibuat untuk mendukung UX dan interpretasi hasil
        """
    )

    st.markdown("### 📌 Deployment Notes")
    st.markdown(
        """
        Untuk deploy ke Streamlit Community Cloud:
        1. Push folder deployment ke GitHub
        2. Pastikan `requirements.txt` tersedia
        3. Deploy melalui Streamlit Cloud
        4. Lampirkan URL aplikasi aktif untuk pengumpulan
        """
    )

st.markdown("---")
st.caption("Built with Streamlit for Model Deployment Assignment")