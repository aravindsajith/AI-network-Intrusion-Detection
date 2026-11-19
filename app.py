
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# BLACK AND WHITE DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #111111;
}

h1, h2, h3, h4, p, label {
    color: #FFFFFF !important;
}

.stButton > button {
    background-color: #FFFFFF;
    color: #000000;
    border-radius: 6px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #cccccc;
    color: #000000;
}

[data-testid="stMetric"] {
    background-color: #151515;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    svm = joblib.load("svm_model.joblib")
    cnn = load_model("cnn_model.keras")
    lstm = load_model("lstm_model.keras")

    scaler = joblib.load("scaler.joblib")
    prep = joblib.load("preprocessing_info.joblib")

    return svm, cnn, lstm, scaler, prep


st.title("AI-Based Network Intrusion Detection")

st.caption(
    "Machine Learning and Deep Learning | UNSW-NB15 Dataset"
)

try:

    svm, cnn, lstm, scaler, prep = load_models()

    st.success("All three models loaded successfully!")

except Exception as e:

    st.error("Unable to load trained models.")
    st.exception(e)
    st.stop()


# =====================================================
# DATA PREPROCESSING
# =====================================================

def preprocess_data(df):

    X = df.drop(
        columns=["label", "attack_cat", "id"],
        errors="ignore"
    ).copy()

    num_cols = list(prep["num_cols"])
    cat_cols = list(prep["cat_cols"])

    expected_columns = num_cols + cat_cols

    missing = [
        col for col in expected_columns
        if col not in X.columns
    ]

    extra = [
        col for col in X.columns
        if col not in expected_columns
    ]

    if missing or extra:

        raise ValueError(
            f"CSV columns do not match training data. "
            f"Missing: {missing}. Extra: {extra}"
        )

    X = X[expected_columns]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X[num_cols] = X[num_cols].fillna(
        prep["medians"]
    )

    X[cat_cols] = X[cat_cols].fillna("missing")

    X = pd.get_dummies(X)

    X = X.reindex(
        columns=prep["feature_columns"],
        fill_value=0
    )

    X_scaled = scaler.transform(X)

    return X_scaled


# =====================================================
# MODEL PREDICTION
# =====================================================

def predict_model(model_name, X_scaled):

    if model_name == "SVM":

        predictions = svm.predict(X_scaled)
        scores = svm.decision_function(X_scaled)

    elif model_name == "CNN":

        data = X_scaled.astype("float32").reshape(
            -1, X_scaled.shape[1], 1
        )

        scores = cnn.predict(
            data,
            verbose=0
        ).ravel()

        predictions = (scores >= 0.5).astype(int)

    else:

        data = X_scaled.astype("float32").reshape(
            -1, X_scaled.shape[1], 1
        )

        scores = lstm.predict(
            data,
            verbose=0
        ).ravel()

        predictions = (scores >= 0.5).astype(int)

    return predictions, scores


# =====================================================
# CALCULATE EVALUATION METRICS
# =====================================================

def calculate_metrics(actual, predictions, scores):

    return {
        "Accuracy (%)": accuracy_score(
            actual, predictions
        ) * 100,

        "Precision (attack)": precision_score(
            actual, predictions, zero_division=0
        ),

        "Recall (attack)": recall_score(
            actual, predictions, zero_division=0
        ),

        "F1 Score (attack)": f1_score(
            actual, predictions, zero_division=0
        ),

        "ROC AUC": roc_auc_score(
            actual, scores
        ),

        "PR AUC": average_precision_score(
            actual, scores
        )
    }


# =====================================================
# NAVIGATION
# =====================================================

st.sidebar.title("NAVIGATION")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Intrusion Detection",
        "Model Performance",
        "Advanced Evaluation",
        "Detection Results",
        "About Project"
    ]
)


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Dashboard Overview")

    st.write("""
    An AI-based network intrusion detection system
    that classifies network traffic as normal or malicious
    using machine learning and deep learning.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Available Models", "3")
    c2.metric("Dataset", "UNSW-NB15")
    c3.metric("Classification", "Binary")

    st.divider()

    st.subheader("Detection Models")

    c1, c2, c3 = st.columns(3)

    c1.markdown("### SVM")
    c1.write("Support Vector Machine")

    c2.markdown("### CNN")
    c2.write("Convolutional Neural Network")

    c3.markdown("### LSTM")
    c3.write("Long Short-Term Memory")

    st.divider()

    st.subheader("Project Workflow")

    st.write("""
    1. Upload network traffic dataset.
    2. Preprocess the network traffic records.
    3. Select a trained model.
    4. Detect normal and malicious traffic.
    5. Evaluate and compare model performance.
    """)

    st.info(
        "This prototype analyzes uploaded CSV records. "
        "It does not monitor live network traffic."
    )


# =====================================================
# INTRUSION DETECTION
# =====================================================

elif page == "Intrusion Detection":

    st.header("Network Intrusion Detection")

    model_name = st.selectbox(
        "Select Detection Model",
        ["SVM", "CNN", "LSTM"]
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.write("Dataset shape:", df.shape)

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            # -----------------------------------------
            # RUN SELECTED MODEL
            # -----------------------------------------

            if st.button("Run Intrusion Detection"):

                with st.spinner("Analyzing network traffic..."):

                    X_scaled = preprocess_data(df)

                    start = time.time()

                    predictions, scores = predict_model(
                        model_name,
                        X_scaled
                    )

                    prediction_time = time.time() - start

                    results = df.copy()

                    results["Predicted Label"] = predictions

                    results["Prediction"] = np.where(
                        predictions == 1,
                        "Attack",
                        "Normal"
                    )

                    st.session_state["results"] = results
                    st.session_state["predictions"] = predictions
                    st.session_state["scores"] = scores
                    st.session_state["selected_model"] = model_name
                    st.session_state["prediction_time"] = prediction_time

                    if "label" in df.columns:

                        st.session_state["uploaded_labels"] = (
                            pd.to_numeric(
                                df["label"],
                                errors="coerce"
                            )
                        )

                    else:

                        st.session_state["uploaded_labels"] = None

                st.success("Detection completed successfully!")

            # -----------------------------------------
            # EVALUATE ALL THREE MODELS
            # -----------------------------------------

            if st.button("Evaluate All Three Models"):

                try:

                    with st.spinner("Evaluating all models..."):

                        X_scaled = preprocess_data(df)

                        all_results = {}

                        for name in ["SVM", "CNN", "LSTM"]:

                            predictions, scores = predict_model(
                                name,
                                X_scaled
                            )

                            all_results[name] = {
                                "predictions": predictions,
                                "scores": scores
                            }

                        st.session_state["all_model_results"] = all_results

                        if "label" in df.columns:

                            st.session_state["uploaded_labels"] = (
                                pd.to_numeric(
                                    df["label"],
                                    errors="coerce"
                                )
                            )

                        else:

                            st.session_state["uploaded_labels"] = None

                    st.success(
                        "All three models have been evaluated!"
                    )

                except Exception as e:

                    st.error("Unable to evaluate all models.")
                    st.exception(e)

        except Exception as e:

            st.error("Error while processing the uploaded dataset.")
            st.exception(e)

    # ---------------------------------------------
    # DISPLAY DETECTION RESULTS
    # ---------------------------------------------

    if "results" in st.session_state:

        results = st.session_state["results"]

        predictions = st.session_state["predictions"]

        st.divider()

        st.subheader("Detection Summary")

        total = len(predictions)

        attacks = int(np.sum(predictions == 1))

        normal = int(np.sum(predictions == 0))

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Records", total)
        c2.metric("Attack Records", attacks)
        c3.metric("Normal Records", normal)

        st.subheader("Traffic Distribution")

        distribution = pd.DataFrame(
            {"Records": [normal, attacks]},
            index=["Normal", "Attack"]
        )

        st.bar_chart(distribution)

        st.subheader("Prediction Details")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.download_button(
            "Download Prediction Results",
            data=results.to_csv(index=False),
            file_name="intrusion_detection_results.csv",
            mime="text/csv"
        )

        st.write(
            "Selected Model:",
            st.session_state["selected_model"]
        )

        st.write(
            "Prediction Time:",
            f'{st.session_state["prediction_time"]:.2f} seconds'
        )

        actual = st.session_state.get("uploaded_labels")

        if actual is not None:

            if actual.notna().all() and actual.isin([0, 1]).all():

                actual = actual.astype(int)

                accuracy = accuracy_score(
                    actual,
                    predictions
                )

                st.metric(
                    "Test Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


# =====================================================
# MODEL PERFORMANCE COMPARISON
# =====================================================

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.write(
        "Compare SVM, CNN and LSTM using multiple metrics."
    )

    try:

        comparison = pd.read_csv("model_comparison.csv")

        # Clean column names
        comparison.columns = (
            comparison.columns.astype(str).str.strip()
        )

        # Normalize common alternative column names
        aliases = {
            "model": "Model",
            "accuracy": "Accuracy (%)",
            "accuracy (%)": "Accuracy (%)",
            "precision": "Precision (attack)",
            "precision (attack)": "Precision (attack)",
            "recall": "Recall (attack)",
            "recall (attack)": "Recall (attack)",
            "f1 score": "F1 Score (attack)",
            "f1 score (attack)": "F1 Score (attack)",
            "f1-score": "F1 Score (attack)",
            "f1-score (attack)": "F1 Score (attack)",
            "training time": "Training Time (s)",
            "training time (s)": "Training Time (s)",
            "prediction time": "Prediction Time (s)",
            "prediction time (s)": "Prediction Time (s)"
        }

        rename_columns = {}

        for col in comparison.columns:

            normalized = col.lower().strip()

            if normalized in aliases:

                rename_columns[col] = aliases[normalized]

        comparison = comparison.rename(
            columns=rename_columns
        )

        # If F1 is absent but precision and recall exist,
        # calculate it from the available metrics.
        if (
            "F1 Score (attack)" not in comparison.columns
            and "Precision (attack)" in comparison.columns
            and "Recall (attack)" in comparison.columns
        ):

            p = comparison["Precision (attack)"]
            r = comparison["Recall (attack)"]

            comparison["F1 Score (attack)"] = np.where(
                (p + r) == 0,
                0,
                2 * p * r / (p + r)
            )

        required = [
            "Model",
            "Accuracy (%)",
            "Precision (attack)",
            "Recall (attack)",
            "F1 Score (attack)",
            "Training Time (s)",
            "Prediction Time (s)"
        ]

        missing = [
            col for col in required
            if col not in comparison.columns
        ]

        if missing:

            st.error(
                f"Missing comparison columns: {missing}"
            )

            st.write(
                "Available columns:",
                comparison.columns.tolist()
            )

            st.stop()

        # Convert numeric columns safely
        numeric_columns = required[1:]

        for col in numeric_columns:

            comparison[col] = pd.to_numeric(
                comparison[col],
                errors="coerce"
            )

        comparison = comparison.dropna(
            subset=required
        )

        if comparison.empty:

            st.error("No valid comparison data found.")
            st.stop()

        st.subheader("Complete Model Comparison")

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # PERFORMANCE HIGHLIGHTS

        best_accuracy = comparison.loc[
            comparison["Accuracy (%)"].idxmax()
        ]

        best_f1 = comparison.loc[
            comparison["F1 Score (attack)"].idxmax()
        ]

        fastest_training = comparison.loc[
            comparison["Training Time (s)"].idxmin()
        ]

        fastest_prediction = comparison.loc[
            comparison["Prediction Time (s)"].idxmin()
        ]

        st.subheader("Performance Highlights")

        c1, c2 = st.columns(2)

        c1.metric(
            "Best Accuracy",
            best_accuracy["Model"],
            f'{best_accuracy["Accuracy (%)"]:.2f}%'
        )

        c2.metric(
            "Best Attack F1 Score",
            best_f1["Model"],
            f'{best_f1["F1 Score (attack)"]:.3f}'
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Fastest Training",
            fastest_training["Model"],
            f'{fastest_training["Training Time (s)"]:.2f} sec'
        )

        c4.metric(
            "Fastest Prediction",
            fastest_prediction["Model"],
            f'{fastest_prediction["Prediction Time (s)"]:.2f} sec'
        )

        st.divider()

        # ACCURACY

        st.subheader("1. Accuracy Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Accuracy (%)"]]
        )

        # PRECISION

        st.subheader("2. Attack Precision Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Precision (attack)"]]
        )

        # RECALL

        st.subheader("3. Attack Recall Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Recall (attack)"]]
        )

        # F1 SCORE

        st.subheader("4. Attack F1-Score Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["F1 Score (attack)"]]
        )

        # CLASSIFICATION METRICS

        st.subheader("5. Overall Classification Metrics")

        classification_metrics = comparison.set_index(
            "Model"
        )[
            [
                "Accuracy (%)",
                "Precision (attack)",
                "Recall (attack)",
                "F1 Score (attack)"
            ]
        ]

        st.bar_chart(classification_metrics)

        # TRAINING TIME

        st.subheader("6. Training Time Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Training Time (s)"]]
        )

        # PREDICTION TIME

        st.subheader("7. Prediction Time Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Prediction Time (s)"]]
        )

        st.divider()

        st.subheader("Model Analysis")

        st.markdown("""
        **CNN**
        - Achieved the highest test accuracy.
        - Achieved the highest attack precision and F1-score.

        **SVM**
        - Had the fastest prediction time.

        **LSTM**
        - Required the longest training and prediction time.
        - Achieved the lowest test accuracy.

        **Overall Conclusion**

        CNN achieved the best classification performance
        in the experiments.
        """)

        st.caption(
            "Results are based on the held-out UNSW-NB15 testing dataset."
        )

        st.download_button(
            "Download Model Comparison",
            comparison.to_csv(index=False),
            "model_comparison_report.csv",
            "text/csv"
        )

    except Exception as e:

        st.error("Unable to display model comparisons.")
        st.exception(e)


# =====================================================
# ADVANCED EVALUATION
# =====================================================

elif page == "Advanced Evaluation":

    st.header("Advanced Model Evaluation")

    st.write("""
    Compare all three models using confusion matrices,
    ROC curves and Precision-Recall curves.
    """)

    actual = st.session_state.get("uploaded_labels")

    all_results = st.session_state.get("all_model_results")

    if actual is None:

        st.warning(
            "Upload the labelled UNSW-NB15 testing CSV and "
            "click Evaluate All Three Models first."
        )

    elif all_results is None:

        st.warning(
            "Go to Intrusion Detection and click "
            "Evaluate All Three Models."
        )

    else:

        actual = pd.to_numeric(
            actual,
            errors="coerce"
        )

        if (
            not actual.notna().all()
            or not actual.isin([0, 1]).all()
        ):

            st.error("Invalid binary ground-truth labels.")

        elif len(np.unique(actual)) < 2:

            st.warning(
                "Evaluation requires both normal and attack records."
            )

        else:

            actual = actual.astype(int)

            st.success(
                "All three models are ready for comparison."
            )

            # -----------------------------------------
            # CONFUSION MATRICES
            # -----------------------------------------

            st.subheader("1. Confusion Matrix Heatmaps")

            for name in ["SVM", "CNN", "LSTM"]:

                predictions = all_results[name]["predictions"]

                cm = confusion_matrix(
                    actual,
                    predictions,
                    labels=[0, 1]
                )

                fig, ax = plt.subplots(figsize=(5, 4))

                image = ax.imshow(
                    cm,
                    cmap="Greys"
                )

                ax.set_title(name + " Confusion Matrix")

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                ax.set_xticklabels(["Normal", "Attack"])
                ax.set_yticklabels(["Normal", "Attack"])

                for i in range(2):

                    for j in range(2):

                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                            color="red"
                        )

                fig.colorbar(image, ax=ax)

                st.pyplot(fig)

                plt.close(fig)

            st.divider()

            # -----------------------------------------
            # ROC CURVES
            # -----------------------------------------

            st.subheader("2. ROC Curve Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))

            for name in ["SVM", "CNN", "LSTM"]:

                scores = all_results[name]["scores"]

                fpr, tpr, thresholds = roc_curve(
                    actual,
                    scores
                )

                auc = roc_auc_score(
                    actual,
                    scores
                )

                ax.plot(
                    fpr,
                    tpr,
                    label=f"{name} (AUC = {auc:.3f})"
                )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random Classifier"
            )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")

            ax.set_title("ROC Curve Comparison")

            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            plt.close(fig)

            st.divider()

            # -----------------------------------------
            # PRECISION-RECALL CURVES
            # -----------------------------------------

            st.subheader("3. Precision-Recall Curve Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))

            for name in ["SVM", "CNN", "LSTM"]:

                scores = all_results[name]["scores"]

                precision, recall, thresholds = precision_recall_curve(
                    actual,
                    scores
                )

                ap = average_precision_score(
                    actual,
                    scores
                )

                ax.plot(
                    recall,
                    precision,
                    label=f"{name} (AP = {ap:.3f})"
                )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            ax.set_title("Precision-Recall Curve Comparison")

            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            plt.close(fig)

            st.divider()

            # -----------------------------------------
            # ADVANCED METRICS
            # -----------------------------------------

            st.subheader("4. Advanced Metrics Comparison")

            rows = []

            for name in ["SVM", "CNN", "LSTM"]:

                predictions = all_results[name]["predictions"]

                scores = all_results[name]["scores"]

                metrics = calculate_metrics(
                    actual,
                    predictions,
                    scores
                )

                metrics["Model"] = name

                rows.append(metrics)

            metrics_df = pd.DataFrame(rows)

            metrics_df = metrics_df[
                [
                    "Model",
                    "Accuracy (%)",
                    "Precision (attack)",
                    "Recall (attack)",
                    "F1 Score (attack)",
                    "ROC AUC",
                    "PR AUC"
                ]
            ]

            st.dataframe(
                metrics_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Advanced Evaluation Report",
                metrics_df.to_csv(index=False),
                "advanced_evaluation_report.csv",
                "text/csv"
            )


# =====================================================
# DETECTION RESULTS
# =====================================================

elif page == "Detection Results":

    st.header("Detection Results")

    if "results" in st.session_state:

        st.dataframe(
            st.session_state["results"],
            use_container_width=True
        )

        st.download_button(
            "Download Detection Results",
            st.session_state["results"].to_csv(index=False),
            "detection_results.csv",
            "text/csv"
        )

    else:

        st.info(
            "No detection results available. "
            "Run intrusion detection first."
        )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    **AI-Based Network Intrusion Detection Using
    Machine Learning and Deep Learning**

    Dataset: UNSW-NB15

    Models:

    1. Support Vector Machine (SVM)
    2. Convolutional Neural Network (CNN)
    3. Long Short-Term Memory (LSTM)

    Objective:

    To classify network traffic as normal or malicious
    and compare the performance of three models.

    Technologies:

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - TensorFlow
    - Streamlit
    - Matplotlib

    This prototype analyzes uploaded CSV records.
    It does not monitor live network traffic or
    automatically block attacks.
    """)

    st.divider()

    st.subheader("Dataset Information")

    st.write("""
    UNSW-NB15 is a network intrusion detection dataset
    containing normal and malicious network traffic.

    0 = Normal Traffic

    1 = Attack Traffic
    """)


# =====================================================
# SIDEBAR FOOTER
# =====================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI Network Intrusion Detection | M.Tech Project"
)
