
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111111;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton > button {
    background-color: white;
    color: black;
    border-radius: 6px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #cccccc;
    color: black;
}
[data-testid="stMetric"] {
    background-color: #151515;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    svm = joblib.load("svm_model.joblib")
    cnn = load_model("cnn_model.keras")
    lstm = load_model("lstm_model.keras")

    scaler = joblib.load("scaler.joblib")
    prep = joblib.load("preprocessing_info.joblib")

    return svm, cnn, lstm, scaler, prep


st.title("AI-Based Network Intrusion Detection")
st.caption("Machine Learning and Deep Learning | UNSW-NB15 Dataset")

try:
    svm, cnn, lstm, scaler, prep = load_models()
    st.success("All three models loaded successfully!")

except Exception as e:
    st.error("Unable to load the trained models.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("NAVIGATION")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Intrusion Detection",
        "Model Performance",
        "Detection Results",
        "About Project"
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Dashboard Overview")

    st.write("""
    This system uses machine learning and deep learning
    models to identify normal and malicious network traffic.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Available Models", "3")
    c2.metric("Dataset", "UNSW-NB15")
    c3.metric("Detection Type", "Binary Classification")

    st.divider()

    st.subheader("Available Detection Models")

    c1, c2, c3 = st.columns(3)

    c1.markdown("### SVM")
    c1.write("Support Vector Machine")

    c2.markdown("### CNN")
    c2.write("Convolutional Neural Network")

    c3.markdown("### LSTM")
    c3.write("Long Short-Term Memory")

    st.info(
        "Navigate to Intrusion Detection to upload a CSV "
        "file and classify network traffic."
    )

# --------------------------------------------------
# INTRUSION DETECTION
# --------------------------------------------------

elif page == "Intrusion Detection":

    st.header("Network Intrusion Detection")

    model_name = st.selectbox(
        "Choose a detection model",
        ["SVM", "CNN", "LSTM"]
    )

    uploaded_file = st.file_uploader(
        "Upload your network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Dataset Preview")

            st.write("Dataset shape:", df.shape)

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            if st.button("Run Intrusion Detection"):

                with st.spinner("Analyzing network traffic..."):

                    # Save actual labels separately for evaluation
                    actual_labels = (
                        df["label"].copy()
                        if "label" in df.columns
                        else None
                    )

                    # Remove target and identifier columns
                    X = df.drop(
                        columns=["label", "attack_cat", "id"],
                        errors="ignore"
                    ).copy()

                    # Expected original input columns
                    num_cols = list(prep["num_cols"])
                    cat_cols = list(prep["cat_cols"])

                    expected_raw = num_cols + cat_cols

                    missing = [
                        col for col in expected_raw
                        if col not in X.columns
                    ]

                    extra = [
                        col for col in X.columns
                        if col not in expected_raw
                    ]

                    if missing or extra:

                        st.error(
                            "Uploaded CSV columns do not match "
                            "the training dataset."
                        )

                        if missing:
                            st.write("Missing columns:", missing)

                        if extra:
                            st.write("Unexpected columns:", extra)

                        st.stop()

                    # Match training input schema
                    X = X[expected_raw]

                    # Handle missing and infinite values
                    X = X.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    X[num_cols] = X[num_cols].fillna(
                        prep["medians"]
                    )

                    X[cat_cols] = X[cat_cols].fillna("missing")

                    # One-hot encoding
                    X = pd.get_dummies(X)

                    # Match original training features
                    X = X.reindex(
                        columns=prep["feature_columns"],
                        fill_value=0
                    )

                    # Apply the previously trained scaler
                    X_scaled = scaler.transform(X)

                    # Prediction
                    start_time = time.time()

                    if model_name == "SVM":

                        predictions = svm.predict(X_scaled)

                    elif model_name == "CNN":

                        cnn_input = X_scaled.astype(
                            "float32"
                        ).reshape(
                            -1, X_scaled.shape[1], 1
                        )

                        probabilities = cnn.predict(
                            cnn_input,
                            verbose=0
                        ).ravel()

                        predictions = (
                            probabilities >= 0.5
                        ).astype(int)

                    else:

                        lstm_input = X_scaled.astype(
                            "float32"
                        ).reshape(
                            -1, X_scaled.shape[1], 1
                        )

                        probabilities = lstm.predict(
                            lstm_input,
                            verbose=0
                        ).ravel()

                        predictions = (
                            probabilities >= 0.5
                        ).astype(int)

                    prediction_time = (
                        time.time() - start_time
                    )

                # Create result dataframe
                results = df.copy()

                results["Predicted Label"] = predictions

                results["Prediction"] = np.where(
                    predictions == 1,
                    "Attack",
                    "Normal"
                )

                # Store results
                st.session_state["results"] = results
                st.session_state["predictions"] = predictions
                st.session_state["actual_labels"] = actual_labels
                st.session_state["prediction_time"] = prediction_time
                st.session_state["selected_model"] = model_name

                st.success("Intrusion detection completed!")

        except Exception as e:

            st.error("An error occurred during detection.")
            st.exception(e)

# --------------------------------------------------
# DETECTION RESULTS
# --------------------------------------------------

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
            {
                "Number of Records": [normal, attacks]
            },
            index=["Normal Traffic", "Attack Traffic"]
        )

        st.bar_chart(distribution)

        st.subheader("Prediction Details")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.download_button(
            "Download Detection Results",
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

        # Evaluate only when valid ground-truth labels exist
        actual = st.session_state["actual_labels"]

        if actual is not None:

            actual = pd.to_numeric(
                actual,
                errors="coerce"
            )

            if actual.notna().all() and actual.isin([0, 1]).all():

                actual = actual.astype(int)

                accuracy = accuracy_score(
                    actual,
                    predictions
                )

                st.subheader("Test Dataset Evaluation")

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                st.write("Confusion Matrix")

                cm = confusion_matrix(
                    actual,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Normal", "Actual Attack"],
                    columns=["Predicted Normal", "Predicted Attack"]
                )

                st.dataframe(cm_df)

                report = classification_report(
                    actual,
                    predictions,
                    target_names=["Normal", "Attack"],
                    output_dict=True,
                    zero_division=0
                )

                st.write("Classification Report")

                st.dataframe(
                    pd.DataFrame(report).transpose()
                )

# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.write(
        "Compare SVM, CNN and LSTM using different "
        "performance and computational metrics."
    )

    try:

        comparison = pd.read_csv("model_comparison.csv")

        required_columns = [
            "Model",
            "Accuracy (%)",
            "Precision (attack)",
            "Recall (attack)",
            "F1 Score (attack)",
            "Training Time (s)",
            "Prediction Time (s)"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in comparison.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns in comparison CSV: {missing_columns}"
            )
            st.stop()

        # Best model summaries
        best_accuracy = comparison.loc[
            comparison["Accuracy (%)"].idxmax()
        ]

        best_precision = comparison.loc[
            comparison["Precision (attack)"].idxmax()
        ]

        best_recall = comparison.loc[
            comparison["Recall (attack)"].idxmax()
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

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Best Accuracy",
            best_accuracy["Model"],
            f'{best_accuracy["Accuracy (%)"]:.2f}%'
        )

        c2.metric(
            "Best Attack F1",
            best_f1["Model"],
            f'{best_f1["F1 Score (attack)"]:.3f}'
        )

        c3.metric(
            "Best Attack Recall",
            best_recall["Model"],
            f'{best_recall["Recall (attack)"]:.3f}'
        )

        c4, c5 = st.columns(2)

        c4.metric(
            "Fastest Training",
            fastest_training["Model"],
            f'{fastest_training["Training Time (s)"]:.2f} sec'
        )

        c5.metric(
            "Fastest Prediction",
            fastest_prediction["Model"],
            f'{fastest_prediction["Prediction Time (s)"]:.2f} sec'
        )

        st.divider()

        # Full comparison table
        st.subheader("Complete Comparison Table")

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Accuracy
        st.subheader("1. Accuracy Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Accuracy (%)"]]
        )

        # Precision
        st.subheader("2. Attack Precision Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Precision (attack)"]]
        )

        # Recall
        st.subheader("3. Attack Recall Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["Recall (attack)"]]
        )

        # F1 score
        st.subheader("4. Attack F1-Score Comparison")

        st.bar_chart(
            comparison.set_index("Model")[["F1 Score (attack)"]]
        )

        st.divider()

        # Classification metrics
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

        st.divider()

        # Training time
        st.subheader("6. Model Training Time")

        st.bar_chart(
            comparison.set_index("Model")[["Training Time (s)"]]
        )

        # Prediction time
        st.subheader("7. Model Prediction Time")

        st.bar_chart(
            comparison.set_index("Model")[["Prediction Time (s)"]]
        )

        st.divider()

        # Model interpretation
        st.subheader("Model Analysis")

        st.markdown("""
        **CNN**
        - Achieved the highest test accuracy.
        - Achieved the highest attack precision and F1-score.

        **SVM**
        - Had the fastest prediction time.
        - Provides a comparatively lightweight prediction process.

        **LSTM**
        - Required the longest training time.
        - Had the lowest test accuracy among the three models.

        **Overall Observation**

        CNN performed best in terms of test accuracy and attack
        F1-score, while SVM was the fastest during prediction.
        """)

        st.caption(
            "All metrics are based on the held-out UNSW-NB15 testing dataset."
        )

    except Exception as e:

        st.error("Unable to display model comparisons.")
        st.exception(e)

# --------------------------------------------------
# DETECTION HISTORY
# --------------------------------------------------

elif page == "Detection Results":

    st.header("Detection Results")

    if "results" in st.session_state:

        st.dataframe(
            st.session_state["results"],
            use_container_width=True
        )

    else:

        st.info(
            "No detection results are available yet. "
            "Run an intrusion detection first."
        )

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    AI-Based Network Intrusion Detection Using Machine Learning
    and Deep Learning.

    **Dataset:** UNSW-NB15

    **Models Used:**

    1. Support Vector Machine (SVM)
    2. Convolutional Neural Network (CNN)
    3. Long Short-Term Memory (LSTM)

    **Objective:**

    To classify network traffic as normal or malicious and
    compare the performance of three machine learning models.

    **Technology Used:**

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - TensorFlow and Keras
    - Streamlit

    This is a prototype that analyzes uploaded CSV records.
    It does not monitor live network traffic or automatically
    block attacks.
    """)

    st.divider()

    st.subheader("Project Dataset")

    st.write("UNSW-NB15 Network Intrusion Detection Dataset")

    st.write(
        "The system performs binary classification: "
        "0 = Normal Traffic, 1 = Attack Traffic."
    )

st.sidebar.divider()
st.sidebar.caption("AI Network Intrusion Detection | M.Tech Project")
