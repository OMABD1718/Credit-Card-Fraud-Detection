import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "final_fraud_detection_bundle.joblib"
)

COMPARISON_PATH = os.path.join(
    ARTIFACT_DIR,
    "final_model_comparison.csv"
)

SHAP_PATH = os.path.join(
    ARTIFACT_DIR,
    "top30_shap_features.csv"
)

SHAP_PLOT_PATH = os.path.join(
    ARTIFACT_DIR,
    "shap_feature_importance.png"
)


# =========================================================
# LOAD SAVED ARTIFACTS
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None

    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_shap_ranking():
    if not os.path.exists(SHAP_PATH):
        return None

    return pd.read_csv(SHAP_PATH)


bundle = load_model()
comparison_df = load_comparison()
shap_df = load_shap_ranking()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Fraud Prediction",
        "Model Comparison",
        "Explainable AI"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "M.Tech Data Science Project"
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title(
        "Credit Card Fraud Detection"
    )

    st.markdown(
        """
        ### Project objective

        Detect potentially fraudulent transactions using
        machine learning and explain predictions through
        SHAP-based feature selection.

        ### Methodology

        1. IEEE-CIS dataset preparation
        2. Exploratory data analysis
        3. Missing-value handling
        4. SHAP-based feature selection
        5. XGBoost hyperparameter optimization
        6. Comparison of XGBoost, LightGBM and CatBoost
        7. Fraud prediction and explainability
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        "590,540"
    )

    col2.metric(
        "Fraudulent Transactions",
        "20,663"
    )

    col3.metric(
        "Selected Features",
        "30 / 432"
    )

    st.info(
        "The dataset contains approximately 3.50% "
        "fraudulent transactions."
    )

    st.warning(
        "This is a research prototype. Predictions should "
        "not be used as the sole basis for real financial "
        "decisions."
    )


# =========================================================
# FRAUD PREDICTION
# =========================================================

elif page == "Fraud Prediction":

    st.title("Transaction Fraud Prediction")

    if bundle is None:

        st.error(
            "The saved model bundle was not found. "
            "Check the artifacts directory."
        )

    else:

        model = bundle["model"]
        required_features = bundle["features"]

        threshold = st.slider(
            "Fraud classification threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05
        )

        st.markdown(
            """
            Upload a CSV file containing the 30 features
            used by the trained model.

            The uploaded data must use the same feature
            definitions and compatible formats as the
            training dataset.
            """
        )

        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                input_df = pd.read_csv(uploaded_file)

                st.subheader("Uploaded Data")
                st.dataframe(
                    input_df.head(),
                    use_container_width=True
                )

                missing_features = [
                    feature
                    for feature in required_features
                    if feature not in input_df.columns
                ]

                if missing_features:

                    st.error(
                        "The CSV is missing required features: "
                        + ", ".join(missing_features)
                    )

                else:

                    X_input = input_df[
                        required_features
                    ].copy()

                    if X_input.empty:
                        st.error(
                            "The uploaded CSV contains no transactions."
                        )

                    else:

                        with st.spinner(
                            "Generating predictions..."
                        ):

                            probabilities = (
                                model.predict_proba(
                                    X_input
                                )[:, 1]
                            )

                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                        results_df = input_df.copy()

                        results_df[
                            "Fraud_Probability"
                        ] = probabilities

                        results_df[
                            "Predicted_Class"
                        ] = predictions

                        results_df[
                            "Prediction"
                        ] = np.where(
                            predictions == 1,
                            "Potential Fraud",
                            "Likely Legitimate"
                        )

                        results_df[
                            "Risk_Category"
                        ] = np.select(
                            [
                                probabilities >= 0.70,
                                probabilities >= 0.30
                            ],
                            [
                                "High",
                                "Medium"
                            ],
                            default="Low"
                        )

                        st.subheader("Prediction Summary")

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Transactions Analysed",
                            len(results_df)
                        )

                        c2.metric(
                            "Potential Fraud",
                            int(predictions.sum())
                        )

                        c3.metric(
                            "Flagged Percentage",
                            f"{predictions.mean() * 100:.2f}%"
                        )

                        st.subheader("Prediction Results")

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )

                        csv_data = results_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Predictions CSV",
                            data=csv_data,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as error:

                st.error(
                    f"Could not process the uploaded file: {error}"
                )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    if comparison_df is None:

        st.warning(
            "The model comparison CSV was not found."
        )

    else:

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        available_metrics = [
            metric
            for metric in [
                "ROC_AUC",
                "PR_AUC",
                "Precision",
                "Recall",
                "F1_Score"
            ]
            if metric in comparison_df.columns
        ]

        if available_metrics:

            metric = st.selectbox(
                "Select metric",
                available_metrics,
                index=(
                    available_metrics.index("PR_AUC")
                    if "PR_AUC" in available_metrics
                    else 0
                )
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.bar(
                comparison_df["Model"],
                comparison_df[metric]
            )

            ax.set_title(
                f"Model Comparison: {metric}"
            )

            ax.set_ylabel(metric)
            ax.tick_params(
                axis="x",
                labelrotation=15
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    st.title("Explainable AI — SHAP Feature Importance")

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) helps explain
        how features contribute to model predictions.

        The following chart displays the global feature
        ranking obtained from the initial XGBoost model.
        """
    )

    if shap_df is not None:

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(30, len(shap_df)),
            value=min(20, len(shap_df))
        )

        display_df = shap_df.head(top_n).copy()

        st.dataframe(
            display_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        display_df = display_df.sort_values(
            "Mean_Abs_SHAP",
            ascending=True
        )

        ax.barh(
            display_df["Feature"],
            display_df["Mean_Abs_SHAP"]
        )

        ax.set_xlabel("Mean Absolute SHAP Value")
        ax.set_title("Global SHAP Feature Importance")

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    elif os.path.exists(SHAP_PLOT_PATH):

        st.image(
            SHAP_PLOT_PATH,
            caption="Saved SHAP feature-importance plot",
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP feature ranking and plot were not found."
        )

    st.info(
        "Global SHAP importance measures the magnitude of "
        "feature contributions. It does not establish "
        "causation or explain the direction of an individual "
        "transaction's prediction."
    )