import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# =========================================================
# 1. PAGE CONFIG + THEME
# =========================================================
st.set_page_config(
    page_title="NetGuard | AI Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BG = "#0B0F16"
PANEL = "#121A26"
PANEL_ALT = "#172233"
BORDER = "#263449"
TEXT = "#F2F5F9"
MUTED = "#9AA9BC"
ACCENT = "#8DB7D8"
TEAL = "#65C7B5"
RED = "#E58B91"
GOLD = "#D8B879"

st.markdown(f"""
<style>
:root {{
    color-scheme: dark;
}}
.stApp {{
    background: {BG};
    color: {TEXT};
}}
.block-container {{
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
}}
[data-testid="stSidebar"] {{
    background: #0F1621;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}
h1, h2, h3, h4 {{
    color: {TEXT} !important;
    letter-spacing: -0.035em;
}}
p, label, li {{
    color: #D4DCE7;
}}
small, .stCaption {{
    color: {MUTED} !important;
}}
hr {{
    border-color: {BORDER};
}}
[data-testid="stMetric"] {{
    background: linear-gradient(145deg, #172233, #121A26);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px 17px;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.8rem;
}}
.stButton > button, .stDownloadButton > button {{
    border-radius: 9px;
    min-height: 2.55rem;
    font-weight: 600;
    border: 1px solid #40536D;
    transition: all 0.15s ease;
}}
.stButton > button {{
    background: #DCE8F2;
    color: #101722;
}}
.stButton > button:hover {{
    background: #BFD4E5;
    color: #101722;
    border-color: #BFD4E5;
}}
.stDownloadButton > button {{
    background: #182536;
    color: #EAF1F8;
}}
.stDownloadButton > button:hover {{
    background: #24364B;
    color: white;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}
[data-testid="stFileUploader"] section {{
    background: {PANEL};
    border: 1px dashed #40536D;
    border-radius: 12px;
}}
[data-testid="stAlert"] {{
    border-radius: 11px;
}}
[data-testid="stTabs"] button {{
    color: {MUTED};
}}
[data-testid="stPlotlyChart"], [data-testid="stPyplot"], [data-testid="stVegaLiteChart"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 7px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 13px;
}}
/* Sidebar radio navigation */
[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: 5px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    background: #141E2B;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 9px 11px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    border-color: #40536D;
}}
/* Compact section labels */
.eyebrow {{
    color: {ACCENT};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}}
.hero {{
    background: linear-gradient(115deg, #172638 0%, #121A26 58%, #17212E 100%);
    border: 1px solid #2B4058;
    border-radius: 18px;
    padding: 25px 27px;
    margin: 4px 0 20px 0;
}}
.hero-title {{
    color: #F4F7FA;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.04em;
}}
.hero-sub {{
    color: #AAB8C9;
    font-size: 0.95rem;
    margin-top: 7px;
}}
.pill {{
    display: inline-block;
    border: 1px solid #38536A;
    background: #1A2B3C;
    color: #B9D6E9;
    border-radius: 30px;
    padding: 5px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
.info-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 13px;
    padding: 16px 17px;
    min-height: 115px;
}}
.info-card-title {{
    color: {MUTED};
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.info-card-value {{
    color: {TEXT};
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 9px;
}}
.info-card-note {{
    color: {MUTED};
    font-size: 0.78rem;
    margin-top: 4px;
}}
.footer {{
    color: #728197;
    font-size: 0.75rem;
    border-top: 1px solid {BORDER};
    padding-top: 14px;
    margin-top: 28px;
}}
</style>
""", unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def info_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{label}</div>
            <div class="info-card-value">{value}</div>
            <div class="info-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def styled_table(df):
    return (
        df.style
        .format(precision=3)
        .set_properties(**{
            "background-color": PANEL,
            "color": TEXT,
            "border-color": BORDER,
            "padding": "8px"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#202D3F"),
                ("color", "#F3F6FA"),
                ("font-weight", "700"),
                ("border-color", BORDER),
                ("padding", "10px")
            ]},
            {"selector": "td", "props": [("border-color", BORDER)]}
        ])
    )


def style_matplotlib(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#D6DFEA", labelsize=8)
    ax.xaxis.label.set_color("#C8D3E0")
    ax.yaxis.label.set_color("#C8D3E0")
    ax.title.set_color(TEXT)
    ax.grid(alpha=0.14, color="#B7C5D6")
    for spine in ax.spines.values():
        spine.set_color(BORDER)


# =========================================================
# 2. LOAD TRAINED MODEL FILES
# =========================================================
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_model.joblib")
    cnn_model = load_model("cnn_model.keras")
    lstm_model = load_model("lstm_model.keras")
    scaler = joblib.load("scaler.joblib")
    prep = joblib.load("preprocessing_info.joblib")
    return svm_model, cnn_model, lstm_model, scaler, prep


try:
    svm_model, cnn_model, lstm_model, scaler, prep = load_models()
except Exception as error:
    st.error("The trained model files could not be loaded.")
    st.code(str(error))
    st.info(
        "Check that svm_model.joblib, cnn_model.keras, lstm_model.keras, "
        "scaler.joblib and preprocessing_info.joblib are in the repository root."
    )
    st.stop()


# =========================================================
# 3. PREPROCESSING + PREDICTION
# =========================================================
def preprocess_data(raw_df):
    X = raw_df.drop(
        columns=["label", "attack_cat", "id"],
        errors="ignore"
    ).copy()

    num_cols = list(prep["num_cols"])
    cat_cols = list(prep["cat_cols"])
    expected_raw = num_cols + cat_cols

    missing = [col for col in expected_raw if col not in X.columns]
    extra = [col for col in X.columns if col not in expected_raw]

    if missing or extra:
        raise ValueError(
            "Uploaded CSV columns do not match the training features. "
            f"Missing columns: {missing}; unexpected columns: {extra}"
        )

    X = X[expected_raw]
    X = X.replace([np.inf, -np.inf], np.nan)

    X[num_cols] = X[num_cols].fillna(prep["medians"])
    X[cat_cols] = X[cat_cols].fillna("missing")

    X = pd.get_dummies(X)
    X = X.reindex(columns=prep["feature_columns"], fill_value=0)

    return scaler.transform(X)


def predict_model(model_name, X_scaled):
    if model_name == "SVM":
        scores = svm_model.decision_function(X_scaled)
        predictions = (scores >= 0).astype(int)
    else:
        shaped = X_scaled.astype("float32").reshape(
            X_scaled.shape[0], X_scaled.shape[1], 1
        )
        model = cnn_model if model_name == "CNN" else lstm_model
        scores = model.predict(shaped, verbose=0).ravel()
        predictions = (scores >= 0.5).astype(int)

    return predictions, scores


def get_metrics(y_true, y_pred, scores):
    return {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Precision (attack)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (attack)": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score (attack)": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, scores),
        "PR AUC": average_precision_score(y_true, scores)
    }


# =========================================================
# 4. SIDEBAR + HEADER
# =========================================================
with st.sidebar:
    st.markdown("## 🛡️ NETGUARD")
    st.caption("AI NETWORK SECURITY")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Intrusion Detection",
            "Model Performance",
            "Advanced Evaluation",
            "Detection Results",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.markdown(
        '<span class="pill">● SVM READY</span> '
        '<span class="pill">● CNN READY</span> '
        '<span class="pill">● LSTM READY</span>',
        unsafe_allow_html=True
    )
    st.caption("UNSW-NB15 · Binary classification")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED CYBERSECURITY ANALYTICS</div>
        <div class="hero-title">Network Intrusion Detection</div>
        <div class="hero-sub">
            Analyze network traffic, identify potential attacks and compare
            machine learning and deep learning models.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. OVERVIEW
# =========================================================
if page == "Overview":
    section_label("Workspace / Overview")
    st.header("Security at a glance")
    st.caption("A dataset-based prototype for binary network traffic classification.")

    comparison = None
    try:
        comparison = pd.read_csv("model_comparison.csv")
        comparison.columns = comparison.columns.astype(str).str.strip()
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        info_card("Detection models", "03", "SVM · CNN · LSTM")
    with c2:
        info_card("Dataset", "UNSW-NB15", "Benchmark intrusion dataset")
    with c3:
        info_card("Classification", "Binary", "Normal or attack")
    with c4:
        info_card("Analysis mode", "CSV-based", "Uploaded traffic records")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Analysis workflow")
        workflow = [
            ("01", "Upload dataset", "Import a compatible UNSW-NB15 CSV file."),
            ("02", "Preprocess features", "Apply the saved training preprocessing and scaler."),
            ("03", "Run detection", "Choose SVM, CNN or LSTM to classify records."),
            ("04", "Evaluate models", "Compare metrics, confusion matrices and curves.")
        ]
        for number, title, description in workflow:
            st.markdown(
                f"""
                <div class="info-card" style="min-height:0; margin-bottom:9px;">
                    <span class="pill">{number}</span>
                    <b style="color:#F2F5F9; margin-left:8px;">{title}</b>
                    <div class="info-card-note" style="margin-top:8px;">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:
        st.subheader("Model snapshot")
        if comparison is not None and not comparison.empty:
            # Normalize common CSV column headings.
            aliases = {
                "model": "Model",
                "accuracy (%)": "Accuracy (%)",
                "accuracy": "Accuracy (%)",
                "f1 score (attack)": "F1 Score (attack)",
                "f1 score": "F1 Score (attack)"
            }
            comparison = comparison.rename(
                columns={c: aliases.get(c.lower().strip(), c) for c in comparison.columns}
            )
            if {"Model", "Accuracy (%)"}.issubset(comparison.columns):
                for _, row in comparison.iterrows():
                    st.markdown(
                        f"""
                        <div class="info-card" style="min-height:0; margin-bottom:9px;">
                            <div style="display:flex; justify-content:space-between;">
                                <b style="color:#F2F5F9;">{row['Model']}</b>
                                <b style="color:#B9D6E9;">{float(row['Accuracy (%)']):.2f}%</b>
                            </div>
                            <div style="background:#28384B; height:6px; border-radius:8px; margin-top:10px;">
                                <div style="background:#8DB7D8; width:{max(0,min(100,float(row['Accuracy (%)'])))}%; height:6px; border-radius:8px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("Model comparison file does not contain the expected columns.")
        else:
            st.info("The model_comparison.csv file was not found.")

    st.divider()
    st.subheader("Project scope")
    st.info(
        "This prototype analyzes uploaded CSV records. It does not capture live "
        "network packets, continuously monitor a network, or automatically block attacks."
    )


# =========================================================
# 6. INTRUSION DETECTION
# =========================================================
elif page == "Intrusion Detection":
    section_label("Workspace / Detection")
    st.header("Run intrusion detection")
    st.caption("Upload a compatible dataset and select a trained model.")

    model_choice = st.selectbox("Detection model", ["SVM", "CNN", "LSTM"])
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="Use a CSV with the same original feature columns as the UNSW-NB15 training data."
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state["uploaded_df"] = raw_df

            st.markdown("#### Dataset preview")
            a, b, c = st.columns(3)
            a.metric("Records", f"{len(raw_df):,}")
            b.metric("Columns", f"{len(raw_df.columns):,}")
            c.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

            with st.expander("Preview first 10 rows", expanded=True):
                st.dataframe(raw_df.head(10), use_container_width=True, height=260)

            st.write("")
            run_col, eval_col = st.columns(2)

            with run_col:
                run_clicked = st.button("Run selected model", use_container_width=True)
            with eval_col:
                eval_clicked = st.button("Evaluate all three models", use_container_width=True)

            if run_clicked:
                with st.spinner(f"Running {model_choice} detection..."):
                    X_scaled = preprocess_data(raw_df)
                    start = time.perf_counter()
                    pred, scores = predict_model(model_choice, X_scaled)
                    elapsed = time.perf_counter() - start

                    output = raw_df.copy()
                    output["Predicted Label"] = pred
                    output["Prediction"] = np.where(pred == 1, "Attack", "Normal")
                    output["Model Used"] = model_choice

                    st.session_state["results"] = output
                    st.session_state["predictions"] = pred
                    st.session_state["scores"] = scores
                    st.session_state["selected_model"] = model_choice
                    st.session_state["prediction_time"] = elapsed
                    st.session_state["detection_source_name"] = uploaded_file.name

                    if "label" in raw_df.columns:
                        labels = pd.to_numeric(raw_df["label"], errors="coerce")
                        st.session_state["uploaded_labels"] = labels
                    else:
                        st.session_state["uploaded_labels"] = None

                st.success(f"{model_choice} detection completed.")

            if eval_clicked:
                with st.spinner("Evaluating SVM, CNN and LSTM..."):
                    X_scaled = preprocess_data(raw_df)
                    all_results = {}
                    for model_name in ["SVM", "CNN", "LSTM"]:
                        pred, scores = predict_model(model_name, X_scaled)
                        all_results[model_name] = {
                            "predictions": pred,
                            "scores": scores
                        }

                    st.session_state["all_model_results"] = all_results
                    st.session_state["evaluation_source_name"] = uploaded_file.name
                    if "label" in raw_df.columns:
                        st.session_state["uploaded_labels"] = pd.to_numeric(
                            raw_df["label"], errors="coerce"
                        )
                    else:
                        st.session_state["uploaded_labels"] = None

                st.success("All three models have been evaluated.")

        except Exception as error:
            st.error("The uploaded CSV could not be processed.")
            st.code(str(error))

    if "results" in st.session_state:
        st.divider()
        st.subheader("Latest detection summary")
        pred = st.session_state["predictions"]
        total = len(pred)
        attacks = int(np.sum(pred == 1))
        normal = int(np.sum(pred == 0))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records analyzed", f"{total:,}")
        m2.metric("Attack predictions", f"{attacks:,}")
        m3.metric("Normal predictions", f"{normal:,}")
        m4.metric("Prediction time", f"{st.session_state.get('prediction_time', 0):.2f}s")

        chart_col, details_col = st.columns([0.8, 1.2], gap="large")
        with chart_col:
            st.markdown("#### Traffic classification")
            fig, ax = plt.subplots(figsize=(4.2, 2.8))
            ax.bar(["Normal", "Attack"], [normal, attacks], color=[TEAL, RED], width=0.55)
            ax.set_ylabel("Records")
            ax.set_title("Predicted traffic distribution")
            style_matplotlib(fig, ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with details_col:
            st.markdown("#### Detection details")
            st.markdown(
                f"""
                <div class="info-card" style="min-height:0; margin-bottom:10px;">
                    <div class="info-card-title">Model selected</div>
                    <div class="info-card-value">{st.session_state.get('selected_model', '—')}</div>
                    <div class="info-card-note">Source: {st.session_state.get('detection_source_name', 'Uploaded CSV')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if total:
                st.progress(attacks / total, text=f"{attacks / total * 100:.1f}% of records predicted as attacks")

        with st.expander("View complete prediction table"):
            st.dataframe(st.session_state["results"], use_container_width=True, height=350)

        st.download_button(
            "Download detection results",
            st.session_state["results"].to_csv(index=False),
            "intrusion_detection_results.csv",
            "text/csv",
            use_container_width=False
        )


# =========================================================
# 7. MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    section_label("Workspace / Model analytics")
    st.header("Model performance")
    st.caption("Compare the three trained models using the held-out test results.")

    try:
        comp = pd.read_csv("model_comparison.csv")
        comp.columns = comp.columns.astype(str).str.strip()

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
        comp = comp.rename(columns={
            col: aliases.get(col.lower().strip(), col)
            for col in comp.columns
        })

        required = [
            "Model", "Accuracy (%)", "Precision (attack)",
            "Recall (attack)", "F1 Score (attack)",
            "Training Time (s)", "Prediction Time (s)"
        ]

        if "F1 Score (attack)" not in comp.columns and {
            "Precision (attack)", "Recall (attack)"
        }.issubset(comp.columns):
            p = pd.to_numeric(comp["Precision (attack)"], errors="coerce")
            r = pd.to_numeric(comp["Recall (attack)"], errors="coerce")
            comp["F1 Score (attack)"] = np.where(p + r == 0, 0, 2 * p * r / (p + r))

        missing = [col for col in required if col not in comp.columns]
        if missing:
            st.error(f"Missing comparison columns: {missing}")
            st.write("Available columns:", comp.columns.tolist())
            st.stop()

        for col in required[1:]:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
        comp = comp.dropna(subset=required)

        if comp.empty:
            st.warning("No valid rows were found in model_comparison.csv.")
            st.stop()

        best_acc = comp.loc[comp["Accuracy (%)"].idxmax()]
        best_f1 = comp.loc[comp["F1 Score (attack)"].idxmax()]
        fastest_pred = comp.loc[comp["Prediction Time (s)"].idxmin()]

        k1, k2, k3 = st.columns(3)
        k1.metric("Highest accuracy", str(best_acc["Model"]), f"{best_acc['Accuracy (%)']:.2f}%")
        k2.metric("Best attack F1-score", str(best_f1["Model"]), f"{best_f1['F1 Score (attack)']:.3f}")
        k3.metric("Fastest prediction", str(fastest_pred["Model"]), f"{fastest_pred['Prediction Time (s)']:.2f}s")

        st.divider()
        st.subheader("Complete comparison")
        st.dataframe(styled_table(comp[required]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Classification metrics")
        metric_cols = [
            "Accuracy (%)", "Precision (attack)",
            "Recall (attack)", "F1 Score (attack)"
        ]
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        x = np.arange(len(comp))
        width = 0.18
        palette = [ACCENT, TEAL, GOLD, RED]
        # Convert ratio-based metrics to percentages so all four
        # metrics are shown on the same 0-100 scale.
        plot_values = comp[metric_cols].copy()
        for metric in ["Precision (attack)", "Recall (attack)", "F1 Score (attack)"]:
            plot_values[metric] = plot_values[metric] * 100

        for i, metric in enumerate(metric_cols):
            ax.bar(
                x + (i - 1.5) * width,
                plot_values[metric],
                width,
                label=metric.replace(" (%)", ""),
                color=palette[i]
            )
        ax.set_xticks(x)
        ax.set_xticklabels(comp["Model"])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Model metric comparison")
        ax.legend(fontsize=8, ncol=2, facecolor=PANEL_ALT, edgecolor=BORDER, labelcolor=TEXT)
        style_matplotlib(fig, ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        left, right = st.columns(2)
        with left:
            st.subheader("Training time")
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.bar(comp["Model"], comp["Training Time (s)"], color=ACCENT, width=0.55)
            ax.set_ylabel("Seconds")
            ax.set_title("Model training time")
            style_matplotlib(fig, ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.subheader("Prediction time")
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.bar(comp["Model"], comp["Prediction Time (s)"], color=TEAL, width=0.55)
            ax.set_ylabel("Seconds")
            ax.set_title("Model prediction time")
            style_matplotlib(fig, ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()
        st.subheader("Interpretation")
        st.markdown(
            """
            - **Accuracy** measures the proportion of all records classified correctly.
            - **Attack precision** measures how many records predicted as attacks were actually attacks.
            - **Attack recall** measures how many actual attacks were detected.
            - **Attack F1-score** balances attack precision and attack recall.
            - **Training and prediction time** show computational cost in the recorded experiment.
            """
        )
        st.download_button(
            "Download comparison report",
            comp[required].to_csv(index=False),
            "model_comparison_report.csv",
            "text/csv"
        )
        st.caption("The comparison values are read from model_comparison.csv and reflect the recorded held-out test evaluation.")

    except FileNotFoundError:
        st.error("model_comparison.csv was not found in the repository.")
    except Exception as error:
        st.error("Unable to display model comparison.")
        st.code(str(error))


# =========================================================
# 8. ADVANCED EVALUATION
# =========================================================
elif page == "Advanced Evaluation":
    section_label("Workspace / Evaluation")
    st.header("Advanced evaluation")
    st.caption("Confusion matrices, ROC curves and precision-recall curves.")

    actual = st.session_state.get("uploaded_labels")
    all_results = st.session_state.get("all_model_results")

    if actual is None or all_results is None:
        st.info(
            "Upload the labelled UNSW-NB15 testing CSV in Intrusion Detection "
            "and select 'Evaluate all three models' first."
        )
    else:
        actual = pd.to_numeric(actual, errors="coerce")
        if actual.isna().any() or not actual.isin([0, 1]).all():
            st.error("The uploaded label column must contain only 0 and 1.")
        elif len(np.unique(actual)) < 2:
            st.warning("Both normal and attack labels are needed for these evaluation plots.")
        else:
            actual = actual.astype(int)

            st.subheader("01 · Confusion matrix comparison")
            matrix_cols = st.columns(3)
            for index, name in enumerate(["SVM", "CNN", "LSTM"]):
                pred = all_results[name]["predictions"]
                cm = confusion_matrix(actual, pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(3.1, 2.65))
                ax.imshow(cm, cmap="Blues")
                ax.set_title(name, fontsize=11, fontweight="bold")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_xticks([0, 1], ["Normal", "Attack"])
                ax.set_yticks([0, 1], ["Normal", "Attack"])
                for row in range(2):
                    for col in range(2):
                        ax.text(
                            col, row, f"{cm[row, col]:,}",
                            ha="center", va="center",
                            color="white" if cm[row, col] > cm.max() / 2 else "#142033",
                            fontsize=9, fontweight="bold"
                        )
                style_matplotlib(fig, ax)
                fig.tight_layout()
                with matrix_cols[index]:
                    st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
            roc_col, pr_col = st.columns(2, gap="large")

            with roc_col:
                st.subheader("02 · ROC curves")
                fig, ax = plt.subplots(figsize=(5.2, 3.4))
                palette = {"SVM": ACCENT, "CNN": TEAL, "LSTM": GOLD}
                for name in ["SVM", "CNN", "LSTM"]:
                    scores = all_results[name]["scores"]
                    fpr, tpr, _ = roc_curve(actual, scores)
                    auc = roc_auc_score(actual, scores)
                    ax.plot(fpr, tpr, label=f"{name} · AUC {auc:.3f}", color=palette[name], linewidth=2)
                ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.set_title("Receiver operating characteristic")
                ax.legend(fontsize=8, facecolor=PANEL_ALT, edgecolor=BORDER, labelcolor=TEXT)
                style_matplotlib(fig, ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with pr_col:
                st.subheader("03 · Precision-recall curves")
                fig, ax = plt.subplots(figsize=(5.2, 3.4))
                for name in ["SVM", "CNN", "LSTM"]:
                    scores = all_results[name]["scores"]
                    precision, recall, _ = precision_recall_curve(actual, scores)
                    ap = average_precision_score(actual, scores)
                    ax.plot(recall, precision, label=f"{name} · AP {ap:.3f}", color=palette[name], linewidth=2)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-recall comparison")
                ax.legend(fontsize=8, facecolor=PANEL_ALT, edgecolor=BORDER, labelcolor=TEXT)
                style_matplotlib(fig, ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
            st.subheader("04 · Advanced metrics")
            rows = []
            for name in ["SVM", "CNN", "LSTM"]:
                pred = all_results[name]["predictions"]
                scores = all_results[name]["scores"]
                row = get_metrics(actual, pred, scores)
                row["Model"] = name
                rows.append(row)
            metrics_df = pd.DataFrame(rows)[
                ["Model", "Accuracy (%)", "Precision (attack)", "Recall (attack)",
                 "F1 Score (attack)", "ROC AUC", "PR AUC"]
            ]
            st.dataframe(styled_table(metrics_df), use_container_width=True, hide_index=True)
            st.download_button(
                "Download advanced evaluation",
                metrics_df.to_csv(index=False),
                "advanced_evaluation_report.csv",
                "text/csv"
            )
            st.caption(
                "CNN and LSTM scores are sigmoid outputs. SVM scores are decision-function values. "
                "The curves are calculated from the uploaded dataset labels."
            )


# =========================================================
# 9. DETECTION RESULTS
# =========================================================
elif page == "Detection Results":
    section_label("Workspace / Results")
    st.header("Detection results")
    st.caption("Review and export the most recent model predictions.")

    if "results" not in st.session_state:
        st.info("No saved detection results. Run a model from the Intrusion Detection page first.")
    else:
        output = st.session_state["results"]
        pred = st.session_state["predictions"]
        total = len(pred)
        attacks = int(np.sum(pred == 1))
        normal = int(np.sum(pred == 0))

        a, b, c = st.columns(3)
        a.metric("Total records", f"{total:,}")
        b.metric("Predicted attacks", f"{attacks:,}")
        c.metric("Predicted normal", f"{normal:,}")

        st.markdown(
            f'<span class="pill">MODEL · {st.session_state.get("selected_model", "—")}</span>',
            unsafe_allow_html=True
        )
        st.write("")
        st.subheader("Prediction distribution")
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ax.barh(["Normal", "Attack"], [normal, attacks], color=[TEAL, RED], height=0.48)
        ax.set_xlabel("Number of records")
        ax.set_title("Predicted traffic classes")
        style_matplotlib(fig, ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()
        st.subheader("Prediction records")
        filter_choice = st.selectbox("Filter records", ["All records", "Attack only", "Normal only"])
        if filter_choice == "Attack only":
            view = output[output["Predicted Label"] == 1]
        elif filter_choice == "Normal only":
            view = output[output["Predicted Label"] == 0]
        else:
            view = output

        st.caption(f"Showing {len(view):,} of {len(output):,} records.")
        st.dataframe(view, use_container_width=True, height=420)
        st.download_button(
            "Export displayed records",
            view.to_csv(index=False),
            "filtered_detection_results.csv",
            "text/csv"
        )


# =========================================================
# 10. ABOUT PROJECT
# =========================================================
elif page == "About Project":
    section_label("Workspace / Project information")
    st.header("About this project")

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">M.TECH · AI / MACHINE LEARNING</div>
            <div class="hero-title">AI-Based Network Intrusion Detection</div>
            <div class="hero-sub">
                A comparative study of Support Vector Machine, Convolutional
                Neural Network and Long Short-Term Memory models for
                classifying network traffic.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Objective")
    st.write(
        "To build a dataset-based intrusion detection prototype that classifies "
        "network traffic records into normal and attack categories, and compares "
        "the performance of three supervised learning models."
    )

    st.divider()
    st.subheader("Models used")
    m1, m2, m3 = st.columns(3)
    with m1:
        info_card("SVM", "Support Vector Machine", "Supervised machine learning")
    with m2:
        info_card("CNN", "Convolutional Neural Network", "Deep learning architecture")
    with m3:
        info_card("LSTM", "Long Short-Term Memory", "Recurrent neural network architecture")

    st.divider()
    st.subheader("Dataset")
    st.write(
        "**UNSW-NB15** is used as the network traffic dataset. "
        "The prepared CSV files contain labelled normal and attack records."
    )

    st.subheader("Technology stack")
    st.write(
        "Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · "
        "Matplotlib · Streamlit"
    )

    st.warning(
        "Scope limitation: this prototype predicts labels for uploaded CSV records. "
        "It does not perform live packet capture, real-time network monitoring, "
        "or automatic attack blocking. CNN and LSTM receive feature-wise reshaped "
        "records, not a verified time-ordered network-flow sequence."
    )


# =========================================================
# 11. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        NETGUARD · AI NETWORK INTRUSION DETECTION
        <span style="float:right;">UNSW-NB15 · SVM · CNN · LSTM</span>
    </div>
    """,
    unsafe_allow_html=True
)
