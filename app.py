
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
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DARK MINIMALIST THEME
# =========================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0B0E14;
    color: #F1F3F6;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #10141D;
    border-right: 1px solid #252C38;
}

/* Headings */

h1, h2, h3, h4 {
    color: #F5F7FA !important;
    letter-spacing: -0.4px;
}

/* Normal text */

p, label, span {
    color: #D5DAE3;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #151B26;
    border: 1px solid #283243;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #AAB5C5 !important;
}

[data-testid="stMetricValue"] {
    color: #F5F7FA !important;
}

/* Buttons */

.stButton > button {
    background-color: #E8EDF5;
    color: #10141D;
    border: 1px solid #E8EDF5;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1rem;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #C7D3E3;
    color: #10141D;
    border-color: #C7D3E3;
}

/* Download buttons */

.stDownloadButton > button {
    background-color: #182333;
    color: #E8EDF5;
    border: 1px solid #34445A;
    border-radius: 8px;
}

/* Input boxes */

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stFileUploader {
    background-color: #151B26;
    border-radius: 8px;
}

/* Data tables */

[data-testid="stDataFrame"] {
    border: 1px solid #283243;
    border-radius: 10px;
    overflow: hidden;
}

/* Tabs */

.stTabs [data-baseweb="tab"] {
    color: #AAB5C5;
}

/* Dividers */

hr {
    border-color: #283243;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Chart containers */

[data-testid="stVegaLiteChart"],
[data-testid="stPyplot"] {
    background-color: #151B26;
    border: 1px solid #283243;
    border-radius: 10px;
    padding: 8px;
}

/* Sidebar radio buttons */

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}

/* Captions */

.stCaption {
    color: #8995A7;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PAGE TITLE
# =========================================================

st.title("AI-Based Network Intrusion Detection")

st.caption(
    "NETWORK SECURITY ANALYTICS  /  UNSW-NB15 DATASET"
)


# =========================================================
# 4. LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():

    svm = joblib.load("svm_model.joblib")

    cnn = load_model("cnn_model.keras")

    lstm = load_model("lstm_model.keras")

    scaler = joblib.load("scaler.joblib")

    prep = joblib.load("preprocessing_info.joblib")

    return svm, cnn, lstm, scaler, prep


try:

    svm, cnn, lstm, scaler, prep = load_models()

    st.success("All three trained models loaded successfully.")

except Exception as e:

    st.error("Unable to load trained models.")

    st.exception(e)

    st.stop()


# =========================================================
# 5. TABLE STYLING FUNCTION
# =========================================================

def style_table(dataframe):

    return (
        dataframe.style
        .format(precision=3)
        .set_properties(**{
            "background-color": "#151B26",
            "color": "#E8EDF5",
            "border-color": "#283243",
            "padding": "9px"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#202A3A"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "bold"),
                    ("border-color", "#34445A"),
                    ("padding", "10px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border-color", "#283243")
                ]
            }
        ])
    )


# =========================================================
# 6. DATA PREPROCESSING
# =========================================================

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
            f"Missing columns: {missing}. "
            f"Extra columns: {extra}."
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


# =========================================================
# 7. MODEL PREDICTION
# =========================================================

def predict_model(model_name, X_scaled):

    if model_name == "SVM":

        predictions = svm.predict(X_scaled)

        scores = svm.decision_function(X_scaled)

    elif model_name == "CNN":

        data = X_scaled.astype("float32").reshape(
            -1,
            X_scaled.shape[1],
            1
        )

        scores = cnn.predict(
            data,
            verbose=0
        ).ravel()

        predictions = (scores >= 0.5).astype(int)

    else:

        data = X_scaled.astype("float32").reshape(
            -1,
            X_scaled.shape[1],
            1
        )

        scores = lstm.predict(
            data,
            verbose=0
        ).ravel()

        predictions = (scores >= 0.5).astype(int)

    return predictions, scores


# =========================================================
# 8. METRIC CALCULATION
# =========================================================

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


# =========================================================
# 9. NAVIGATION
# =========================================================

st.sidebar.title("SECURITY DASHBOARD")

st.sidebar.caption("AI-POWERED NETWORK ANALYSIS")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Intrusion Detection",
        "Model Performance",
        "Advanced Evaluation",
        "Detection Results",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "UNSW-NB15  |  SVM  |  CNN  |  LSTM"
)


# =========================================================
# 10. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.header("Dashboard Overview")

    st.write(
        "An AI-based network intrusion detection system "
        "that classifies network traffic as normal or malicious."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Available Models",
            "03"
        )

    with c2:

        st.metric(
            "Dataset",
            "UNSW-NB15"
        )

    with c3:

        st.metric(
            "Detection Type",
            "Binary"
        )

    st.divider()

    st.subheader("Detection Models")

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(border=True):

            st.markdown("### SVM")

            st.caption("SUPPORT VECTOR MACHINE")

            st.write(
                "A supervised learning model "
                "used to classify network traffic."
            )

    with c2:

        with st.container(border=True):

            st.markdown("### CNN")

            st.caption("CONVOLUTIONAL NEURAL NETWORK")

            st.write(
                "A deep learning model used to "
                "identify patterns in input features."
            )

    with c3:

        with st.container(border=True):

            st.markdown("### LSTM")

            st.caption("LONG SHORT-TERM MEMORY")

            st.write(
                "A recurrent neural network "
                "architecture applied to feature sequences."
            )

    st.divider()

    st.subheader("Project Workflow")

    steps = st.columns(5)

    workflow = [
        ("01", "Upload CSV"),
        ("02", "Preprocess"),
        ("03", "Select Model"),
        ("04", "Detect"),
        ("05", "Evaluate")
    ]

    for col, (number, label) in zip(steps, workflow):

        with col:

            with st.container(border=True):

                st.caption(number)

                st.markdown(f"**{label}**")

    st.divider()

    st.info(
        "This prototype analyzes uploaded CSV records. "
        "It does not monitor live network traffic "
        "or automatically block attacks."
    )


# =========================================================
# 11. INTRUSION DETECTION PAGE
# =========================================================

elif page == "Intrusion Detection":

    st.header("Intrusion Detection")

    st.caption(
        "UPLOAD NETWORK TRAFFIC DATA AND RUN A DETECTION MODEL"
    )

    model_name = st.selectbox(
        "Select Detection Model",
        ["SVM", "CNN", "LSTM"]
    )

    uploaded_file = st.file_uploader(
        "Upload UNSW-NB15 CSV Dataset",
        type=["csv"],
        help="Upload a CSV containing the original UNSW-NB15 features."
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.session_state["uploaded_df"] = df

            st.subheader("Dataset Preview")

            c1, c2 = st.columns(2)

            c1.metric(
                "Total Records",
                f"{len(df):,}"
            )

            c2.metric(
                "Total Columns",
                len(df.columns)
            )

            st.dataframe(
                df.head(8),
                use_container_width=True,
                height=260
            )

            st.divider()

            # RUN SELECTED MODEL

            if st.button(
                "Run Intrusion Detection",
                use_container_width=True
            ):

                with st.spinner("Analyzing network traffic..."):

                    X_scaled = preprocess_data(df)

                    start_time = time.time()

                    predictions, scores = predict_model(
                        model_name,
                        X_scaled
                    )

                    prediction_time = time.time() - start_time

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

            st.divider()

            # EVALUATE ALL THREE MODELS

            if st.button(
                "Evaluate All Three Models",
                use_container_width=True
            ):

                try:

                    with st.spinner("Evaluating all three models..."):

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
                        "All three models evaluated successfully!"
                    )

                except Exception as e:

                    st.error("Unable to evaluate all models.")

                    st.exception(e)

        except Exception as e:

            st.error("Unable to process the uploaded dataset.")

            st.exception(e)

    # DISPLAY RESULTS

    if "results" in st.session_state:

        results = st.session_state["results"]

        predictions = st.session_state["predictions"]

        st.divider()

        st.subheader("Detection Summary")

        total = len(predictions)

        attacks = int(np.sum(predictions == 1))

        normal = int(np.sum(predictions == 0))

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Records", f"{total:,}")

        c2.metric("Attack Records", f"{attacks:,}")

        c3.metric("Normal Records", f"{normal:,}")

        st.caption(
            f"Model used: {st.session_state['selected_model']}"
        )

        st.caption(
            f"Prediction time: "
            f"{st.session_state['prediction_time']:.2f} seconds"
        )

        st.divider()

        st.subheader("Traffic Distribution")

        distribution = pd.DataFrame(
            {
                "Records": [normal, attacks]
            },
            index=["Normal", "Attack"]
        )

        st.bar_chart(
            distribution,
            height=230
        )

        st.divider()

        st.subheader("Prediction Details")

        st.dataframe(
            results,
            use_container_width=True,
            height=400
        )

        st.download_button(
            "Download Prediction Results",
            data=results.to_csv(index=False),
            file_name="intrusion_detection_results.csv",
            mime="text/csv"
        )

        actual = st.session_state.get("uploaded_labels")

        if actual is not None:

            if (
                actual.notna().all()
                and actual.isin([0, 1]).all()
            ):

                accuracy = accuracy_score(
                    actual.astype(int),
                    predictions
                )

                st.metric(
                    "Test Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


# =========================================================
# 12. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.caption(
        "COMPARATIVE ANALYSIS OF THE THREE TRAINED MODELS"
    )

    try:

        comparison = pd.read_csv("model_comparison.csv")

        comparison.columns = (
            comparison.columns.astype(str).str.strip()
        )

        # STANDARDIZE COLUMN NAMES

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

            key = col.lower().strip()

            if key in aliases:

                rename_columns[col] = aliases[key]

        comparison = comparison.rename(
            columns=rename_columns
        )

        # CALCULATE F1 IF MISSING

        if (
            "F1 Score (attack)" not in comparison.columns
            and "Precision (attack)" in comparison.columns
            and "Recall (attack)" in comparison.columns
        ):

            p = comparison["Precision (attack)"]

            r = comparison["Recall (attack)"]

            comparison["F1 Score (attack)"] = np.where(
                p + r == 0,
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

            st.error(f"Missing columns: {missing}")

            st.write(
                "Available columns:",
                comparison.columns.tolist()
            )

            st.stop()

        for col in required[1:]:

            comparison[col] = pd.to_numeric(
                comparison[col],
                errors="coerce"
            )

        comparison = comparison.dropna(
            subset=required
        )

        if comparison.empty:

            st.error("No valid model comparison data found.")

            st.stop()

        # COMPARISON TABLE

        st.subheader("Complete Comparison")

        st.dataframe(
            style_table(comparison[required]),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # HIGHLIGHTS

        best_accuracy = comparison.loc[
            comparison["Accuracy (%)"].idxmax()
        ]

        best_f1 = comparison.loc[
            comparison["F1 Score (attack)"].idxmax()
        ]

        fastest_prediction = comparison.loc[
            comparison["Prediction Time (s)"].idxmin()
        ]

        fastest_training = comparison.loc[
            comparison["Training Time (s)"].idxmin()
        ]

        st.subheader("Performance Highlights")

        c1, c2, c3, c4 = st.columns(4)

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
            "Fastest Prediction",
            fastest_prediction["Model"],
            f'{fastest_prediction["Prediction Time (s)"]:.2f} sec'
        )

        c4.metric(
            "Fastest Training",
            fastest_training["Model"],
            f'{fastest_training["Training Time (s)"]:.2f} sec'
        )

        st.divider()

        # CLASSIFICATION METRICS

        st.subheader("Classification Metrics")

        metrics = comparison.set_index("Model")[
            [
                "Accuracy (%)",
                "Precision (attack)",
                "Recall (attack)",
                "F1 Score (attack)"
            ]
        ]

        st.bar_chart(
            metrics,
            height=260
        )

        # INDIVIDUAL METRICS

        left, right = st.columns(2)

        with left:

            st.subheader("Accuracy")

            st.bar_chart(
                comparison.set_index("Model")[["Accuracy (%)"]],
                height=220
            )

            st.subheader("Attack Precision")

            st.bar_chart(
                comparison.set_index("Model")[["Precision (attack)"]],
                height=220
            )

            st.subheader("Training Time")

            st.bar_chart(
                comparison.set_index("Model")[["Training Time (s)"]],
                height=220
            )

        with right:

            st.subheader("Attack Recall")

            st.bar_chart(
                comparison.set_index("Model")[["Recall (attack)"]],
                height=220
            )

            st.subheader("Attack F1 Score")

            st.bar_chart(
                comparison.set_index("Model")[["F1 Score (attack)"]],
                height=220
            )

            st.subheader("Prediction Time")

            st.bar_chart(
                comparison.set_index("Model")[["Prediction Time (s)"]],
                height=220
            )

        st.divider()

        st.subheader("Model Analysis")

        with st.container(border=True):

            st.markdown("""
            **CNN — Best Overall Performance**

            - Highest test accuracy.
            - Highest attack precision and F1-score.

            **SVM — Fastest Prediction**

            - Lowest prediction time among the three models.

            **LSTM — Computationally Expensive**

            - Highest training and prediction time.
            - Lowest test accuracy in these experiments.
            """)

        st.download_button(
            "Download Model Comparison Report",
            comparison[required].to_csv(index=False),
            "model_comparison_report.csv",
            "text/csv"
        )

        st.caption(
            "Metrics are based on the held-out UNSW-NB15 testing dataset."
        )

    except Exception as e:

        st.error("Unable to display model comparison.")

        st.exception(e)


# =========================================================
# 13. ADVANCED EVALUATION PAGE
# =========================================================

elif page == "Advanced Evaluation":

    st.header("Advanced Model Evaluation")

    st.caption(
        "CONFUSION MATRICES  /  ROC CURVES  /  PRECISION-RECALL"
    )

    actual = st.session_state.get("uploaded_labels")

    all_results = st.session_state.get("all_model_results")

    if actual is None:

        st.warning(
            "Upload the labelled UNSW-NB15 testing CSV "
            "and click Evaluate All Three Models first."
        )

    elif all_results is None:

        st.warning(
            "Go to Intrusion Detection and click "
            "Evaluate All Three Models first."
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

            st.error("The uploaded dataset has invalid binary labels.")

        elif len(np.unique(actual)) < 2:

            st.warning(
                "Both normal and attack records are required for evaluation."
            )

        else:

            actual = actual.astype(int)

            st.success(
                "All three models are ready for evaluation."
            )

            # -----------------------------------------
            # CONFUSION MATRICES
            # -----------------------------------------

            st.divider()

            st.subheader("01. Confusion Matrix Comparison")

            matrix_columns = st.columns(3)

            for index, name in enumerate(
                ["SVM", "CNN", "LSTM"]
            ):

                predictions = all_results[name]["predictions"]

                cm = confusion_matrix(
                    actual,
                    predictions,
                    labels=[0, 1]
                )

                fig, ax = plt.subplots(
                    figsize=(3.2, 2.8)
                )

                fig.patch.set_facecolor("#151B26")

                ax.set_facecolor("#151B26")

                image = ax.imshow(
                    cm,
                    cmap="Blues"
                )

                ax.set_title(
                    name,
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    pad=10
                )

                ax.set_xlabel(
                    "Predicted",
                    color="#D5DAE3",
                    fontsize=9
                )

                ax.set_ylabel(
                    "Actual",
                    color="#D5DAE3",
                    fontsize=9
                )

                ax.set_xticks([0, 1])

                ax.set_yticks([0, 1])

                ax.set_xticklabels(
                    ["Normal", "Attack"],
                    color="white",
                    fontsize=8
                )

                ax.set_yticklabels(
                    ["Normal", "Attack"],
                    color="white",
                    fontsize=8
                )

                threshold = cm.max() / 2

                for i in range(2):

                    for j in range(2):

                        ax.text(
                            j,
                            i,
                            f"{cm[i, j]:,}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="white" if cm[i, j] > threshold else "#10141D"
                        )

                ax.tick_params(
                    colors="#D5DAE3"
                )

                for spine in ax.spines.values():

                    spine.set_color("#34445A")

                fig.tight_layout()

                with matrix_columns[index]:

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                plt.close(fig)

            st.divider()

            # -----------------------------------------
            # ROC CURVE AND PR CURVE
            # -----------------------------------------

            roc_column, pr_column = st.columns(2)

            # ROC CURVE

            with roc_column:

                st.subheader("02. ROC Curve")

                fig, ax = plt.subplots(
                    figsize=(5, 3.4)
                )

                fig.patch.set_facecolor("#151B26")

                ax.set_facecolor("#151B26")

                for name in ["SVM", "CNN", "LSTM"]:

                    scores = all_results[name]["scores"]

                    fpr, tpr, _ = roc_curve(
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
                        linewidth=2,
                        label=f"{name} ({auc:.3f})"
                    )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    color="#8995A7",
                    linewidth=1
                )

                ax.set_xlabel(
                    "False Positive Rate",
                    color="white",
                    fontsize=9
                )

                ax.set_ylabel(
                    "True Positive Rate",
                    color="white",
                    fontsize=9
                )

                ax.set_title(
                    "ROC Curve Comparison",
                    color="white",
                    fontsize=11
                )

                ax.tick_params(colors="white")

                ax.grid(
                    alpha=0.15,
                    color="white"
                )

                ax.legend(
                    fontsize=8,
                    facecolor="#202A3A",
                    edgecolor="#34445A",
                    labelcolor="white"
                )

                for spine in ax.spines.values():

                    spine.set_color("#34445A")

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            # PRECISION-RECALL CURVE

            with pr_column:

                st.subheader("03. Precision-Recall Curve")

                fig, ax = plt.subplots(
                    figsize=(5, 3.4)
                )

                fig.patch.set_facecolor("#151B26")

                ax.set_facecolor("#151B26")

                for name in ["SVM", "CNN", "LSTM"]:

                    scores = all_results[name]["scores"]

                    precision, recall, _ = precision_recall_curve(
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
                        linewidth=2,
                        label=f"{name} ({ap:.3f})"
                    )

                ax.set_xlabel(
                    "Recall",
                    color="white",
                    fontsize=9
                )

                ax.set_ylabel(
                    "Precision",
                    color="white",
                    fontsize=9
                )

                ax.set_title(
                    "Precision-Recall Comparison",
                    color="white",
                    fontsize=11
                )

                ax.tick_params(colors="white")

                ax.grid(
                    alpha=0.15,
                    color="white"
                )

                ax.legend(
                    fontsize=8,
                    facecolor="#202A3A",
                    edgecolor="#34445A",
                    labelcolor="white"
                )

                for spine in ax.spines.values():

                    spine.set_color("#34445A")

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            st.divider()

            # -----------------------------------------
            # ADVANCED METRICS TABLE
            # -----------------------------------------

            st.subheader("04. Advanced Metrics Comparison")

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
                style_table(metrics_df),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Advanced Evaluation Report",
                metrics_df.to_csv(index=False),
                "advanced_evaluation_report.csv",
                "text/csv"
            )


# =========================================================
# 14. DETECTION RESULTS PAGE
# =========================================================

elif page == "Detection Results":

    st.header("Detection Results")

    st.caption(
        "PREDICTIONS GENERATED FROM YOUR UPLOADED DATASET"
    )

    if "results" in st.session_state:

        results = st.session_state["results"]

        predictions = st.session_state["predictions"]

        st.subheader("Detection Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Records",
            f"{len(predictions):,}"
        )

        c2.metric(
            "Attack Records",
            f"{int(np.sum(predictions == 1)):,}"
        )

        c3.metric(
            "Normal Records",
            f"{int(np.sum(predictions == 0)):,}"
        )

        st.divider()

        st.subheader("Prediction Table")

        st.dataframe(
            results,
            use_container_width=True,
            height=450
        )

        st.download_button(
            "Download Detection Results",
            results.to_csv(index=False),
            "detection_results.csv",
            "text/csv"
        )

    else:

        st.info(
            "No detection results available. "
            "Go to Intrusion Detection and run a model first."
        )


# =========================================================
# 15. ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.header("About the Project")

    st.caption(
        "PROJECT INFORMATION"
    )

    with st.container(border=True):

        st.subheader(
            "AI-Based Network Intrusion Detection"
        )

        st.write(
            "Using Machine Learning and Deep Learning"
        )

        st.write(
            "The objective is to classify network traffic "
            "as normal or malicious and compare the performance "
            "of three trained models."
        )

    st.divider()

    st.subheader("Dataset")

    st.write(
        "**UNSW-NB15** — A network intrusion detection dataset "
        "containing normal and malicious network traffic."
    )

    st.divider()

    st.subheader("Models Used")

    c1, c2, c3 = st.columns(3)

    c1.markdown("**SVM**")

    c1.caption("Support Vector Machine")

    c2.markdown("**CNN**")

    c2.caption("Convolutional Neural Network")

    c3.markdown("**LSTM**")

    c3.caption("Long Short-Term Memory")

    st.divider()

    st.subheader("Technologies")

    st.write(
        "Python  |  Pandas  |  NumPy  |  Scikit-learn  |  "
        "TensorFlow  |  Streamlit  |  Matplotlib"
    )

    st.divider()

    st.subheader("Classification Labels")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 0")

            st.write("Normal Traffic")

    with c2:

        with st.container(border=True):

            st.markdown("### 1")

            st.write("Attack Traffic")

    st.info(
        "This is a dataset-based prototype. "
        "It does not monitor live network traffic "
        "or automatically block attacks."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI NETWORK INTRUSION DETECTION"
)

st.sidebar.caption(
    "M.Tech AI/ML Project"
)
