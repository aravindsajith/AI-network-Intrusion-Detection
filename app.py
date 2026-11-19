
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# Minimal black and white styling
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #111111;
}
h1, h2, h3, p, label {
    color: #FFFFFF !important;
}
.stButton > button {
    background-color: #FFFFFF;
    color: #000000;
    border-radius: 6px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #cccccc;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Load trained models
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
    st.success("All models loaded successfully")
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Intrusion Detection", "Model Performance", "About Project"]
)

if page == "Overview":
    st.header("Dashboard Overview")
    st.write("An AI-based system for identifying normal and malicious network traffic.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Available Models", "3")
    col2.metric("Dataset", "UNSW-NB15")
    col3.metric("Detection Type", "Binary Classification")

    st.info("Upload a network traffic CSV file in the Intrusion Detection section to begin.")

elif page == "Intrusion Detection":
    st.header("Network Traffic Analysis")

    model_name = st.selectbox(
        "Select Detection Model",
        ["SVM", "CNN", "LSTM"]
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if st.button("Run Intrusion Detection"):
            with st.spinner("Analyzing network traffic..."):

                # Keep ground-truth labels separate from prediction features
                actual_labels = (
                    df["label"].copy()
                    if "label" in df.columns else None
                )

                X = df.drop(
                    columns=["label", "attack_cat", "id"],
                    errors="ignore"
                ).copy()

                expected_raw = prep["num_cols"] + prep["cat_cols"]

                missing = [c for c in expected_raw if c not in X.columns]
                extra = [c for c in X.columns if c not in expected_raw]

                if missing or extra:
                    st.error(
                        f"CSV columns do not match the training data. "
                        f"Missing: {missing}. Extra: {extra}."
                    )
                    st.stop()

                X = X[expected_raw]

                num_cols = prep["num_cols"]
                cat_cols = prep["cat_cols"]

                X = X.replace([np.inf, -np.inf], np.nan)
                X[num_cols] = X[num_cols].fillna(prep["medians"])
                X[cat_cols] = X[cat_cols].fillna("missing")

                X = pd.get_dummies(X)
                X = X.reindex(
                    columns=prep["feature_columns"],
                    fill_value=0
                )

                X_scaled = scaler.transform(X)

                start = time.time()

                if model_name == "SVM":
                    predictions = svm.predict(X_scaled)

                elif model_name == "CNN":
                    data = X_scaled.astype("float32").reshape(
                        -1, 194, 1
                    )
                    predictions = (
                        cnn.predict(data, verbose=0).ravel() >= 0.5
                    ).astype(int)

                else:
                    data = X_scaled.astype("float32").reshape(
                        -1, 194, 1
                    )
                    predictions = (
                        lstm.predict(data, verbose=0).ravel() >= 0.5
                    ).astype(int)

                elapsed = time.time() - start

            results = df.copy()
            results["Predicted Label"] = predictions
            results["Prediction"] = np.where(
                predictions == 1, "Attack", "Normal"
            )

            st.session_state["results"] = results
            st.session_state["predictions"] = predictions
            st.session_state["actual_labels"] = actual_labels
            st.session_state["elapsed"] = elapsed
            st.session_state["model_name"] = model_name

            st.success("Detection completed!")

    if "results" in st.session_state:
        results = st.session_state["results"]
        predictions = st.session_state["predictions"]

        st.subheader("Detection Results")

        total = len(predictions)
        attacks = int(np.sum(predictions == 1))
        normal = int(np.sum(predictions == 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", total)
        c2.metric("Attack Records", attacks)
        c3.metric("Normal Records", normal)

        st.write("### Prediction Distribution")
        st.bar_chart(
            pd.DataFrame({
                "Records": [normal, attacks]
            }, index=["Normal", "Attack"])
        )

        st.write("### Prediction Details")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Prediction Results",
            results.to_csv(index=False),
            "intrusion_detection_results.csv",
            "text/csv"
        )

        actual = st.session_state["actual_labels"]
        if actual is not None:
            st.write("### Test Dataset Evaluation")
            actual = pd.to_numeric(actual, errors="coerce")
            valid = actual.notna()

            if valid.all() and actual.isin([0, 1]).all():
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(actual.astype(int), predictions)
                st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
            else:
                st.warning("Ground-truth labels are not valid binary labels.")

        st.caption(
            f"Model: {st.session_state['model_name']} | "
            f"Prediction time: {st.session_state['elapsed']:.2f} seconds"
        )

elif page == "Model Performance":
    st.header("Model Performance Comparison")

    try:
        comparison = pd.read_csv("model_comparison.csv")
        st.dataframe(comparison, use_container_width=True)

        if "Model" in comparison.columns and "Accuracy (%)" in comparison.columns:
            st.subheader("Accuracy Comparison")
            st.bar_chart(
                comparison.set_index("Model")[["Accuracy (%)"]]
            )

    except Exception as e:
        st.error(f"Could not load comparison data: {e}")

elif page == "About Project":
    st.header("About the Project")

    st.write("""
    This project uses machine learning and deep learning
    techniques to classify network traffic as normal or malicious.

    **Dataset:** UNSW-NB15

    **Models:**
    - Support Vector Machine (SVM)
    - Convolutional Neural Network (CNN)
    - Long Short-Term Memory (LSTM)

    **Objective:** Compare the performance of three models
    for binary network intrusion detection.

    This prototype analyzes uploaded CSV records. It does not
    monitor live network traffic or automatically block attacks.
    """)
