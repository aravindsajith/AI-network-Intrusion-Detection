import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# =========================================================
# 1. PAGE CONFIG + DESIGN SYSTEM
# One neutral palette, one accent. Nothing else.
# =========================================================
st.set_page_config(
    page_title="NetGuard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

BG = "#0C0A00"
SURFACE = "#3E3E3E"
BORDER = "#2C7EF4"
GRID = "#657997"
TEXT = "#F2F2F4"
MUTED = "#9699BF"
SOFT = "#B4B4BC"

# Pastel palette
BLUE = "#88A9C3"        # light blue
YELLOW = "#CBC8B1"      # light yellow
LAVENDER = "#EAEBFF"    # lavender
ORANGE = "#F5B98A"      # light orange
GREY = "#8E8E96"        # grey

ACCENT = BLUE
NORMAL = BLUE           # "normal" traffic
ATTACK = ORANGE         # "attack" traffic

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {{ color-scheme: dark; }}

.stApp {{ background: {BG}; color: {TEXT}; }}
.stApp, .stMarkdown, p, label, li, button, input, textarea, h1, h2, h3, h4 {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* Remove Streamlit chrome */
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stToolbar"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{
    max-width: 1180px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}}

/* Typography */
h1, h2, h3, h4 {{ color: {TEXT} !important; letter-spacing: -0.02em; font-weight: 500 !important; }}
h3 {{ font-size: 1.02rem !important; margin-top: 0.5rem; }}
p, li {{ color: {SOFT}; font-size: 0.93rem; line-height: 1.65; }}
.stCaption, small, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
hr {{ border: 0; border-top: 1px solid {BORDER}; margin: 2.2rem 0; }}

.page-title {{
    font-size: 2rem; font-weight: 300; letter-spacing: -0.03em; color: {TEXT};
    line-height: 1.15;
}}
.page-sub {{ color: {MUTED}; font-size: 0.92rem; margin: 0.55rem 0 2.4rem 0; }}

/* Stats */
.stat {{ padding: 0.2rem 0 1.4rem 0; }}
.stat-label {{
    color: {MUTED}; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
}}
.stat-value {{
    color: {TEXT}; font-size: 1.9rem; font-weight: 300;
    letter-spacing: -0.03em; margin-top: 0.45rem; line-height: 1.1;
}}
.stat-note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.4rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 2.4rem; }}
.brand {{ font-size: 1.05rem; font-weight: 500; letter-spacing: 0.02em; color: {TEXT}; }}
.brand span {{ background: linear-gradient(135deg, {BLUE}, {LAVENDER}, {ORANGE}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-right: 0.5rem; }}
.brand-sub {{ color: {MUTED}; font-size: 0.74rem; margin: 0.3rem 0 2.4rem 0; }}

[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.15rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.5rem 0.2rem; border-radius: 6px; cursor: pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{
    color: {MUTED}; font-size: 0.9rem; transition: color .15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {{ color: {TEXT}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
    color: {TEXT}; font-weight: 500;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    box-shadow: inset 2px 0 0 {ACCENT}; padding-left: 0.8rem;
}}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    border-radius: 8px; min-height: 2.6rem; font-weight: 500; font-size: 0.88rem;
    background: transparent; color: {TEXT}; border: 1px solid #34343A;
    transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {ACCENT}; color: {ACCENT}; background: transparent;
}}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: {ACCENT}; color: #0A0A0B; border-color: {ACCENT};
}}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
    background: #BCDCF6; color: #0A0A0B; border-color: #BCDCF6;
}}

/* Inputs, uploader, tables, alerts */
[data-testid="stFileUploader"] section {{
    background: {SURFACE}; border: 1px dashed #34343A; border-radius: 10px;
}}
div[data-baseweb="select"] > div {{ background: {SURFACE}; border-color: {BORDER}; border-radius: 8px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; }}
[data-testid="stAlert"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; color: {SOFT};
}}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 10px; background: transparent; }}
[data-testid="stProgress"] > div > div > div > div {{ background: linear-gradient(90deg, {LAVENDER}, {ORANGE}); }}

.model-row {{ padding: 0.9rem 0; border-bottom: 1px solid {BORDER}; }}
.model-row-head {{ display: flex; justify-content: space-between; font-size: 0.9rem; color: {TEXT}; }}
.model-row-head span:last-child {{ color: {SOFT}; font-variant-numeric: tabular-nums; }}
.bar-track {{ background: {BORDER}; height: 2px; margin-top: 0.8rem; }}
.bar-fill {{ background: linear-gradient(90deg, {BLUE}, {LAVENDER}); height: 2px; }}

.step {{ display: flex; gap: 1.2rem; padding: 0.85rem 0; border-bottom: 1px solid {BORDER}; }}
.step-n {{ color: {ACCENT}; font-size: 0.78rem; font-weight: 500; min-width: 1.4rem; padding-top: 0.15rem; }}
.step-t {{ color: {TEXT}; font-size: 0.92rem; font-weight: 500; }}
.step-d {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.2rem; }}

.footer {{ color: #55555D; font-size: 0.72rem; margin-top: 4rem; letter-spacing: 0.04em; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------
def page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )


def stat(label, value, note="", color=TEXT):
    st.markdown(
        f"""
        <div class="stat">
            <div class="stat-label">{label}</div>
            <div class="stat-value" style="color:{color};">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def gap(rem=1.5):
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


def styled_table(df):
    return (
        df.style
        .format(precision=3)
        .set_properties(**{"background-color": SURFACE, "color": SOFT, "border-color": BORDER})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", SURFACE), ("color", MUTED),
                ("font-weight", "500"), ("border-color", BORDER)
            ]}
        ])
    )


def style_matplotlib(fig, ax, grid=True):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    if grid:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def minimal_legend(ax, **kwargs):
    ax.legend(frameon=False, fontsize=8, labelcolor=SOFT, **kwargs)


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
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="brand"><span>◆</span>NetGuard</div>'
        '<div class="brand-sub">Network intrusion detection</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Detection",
            "Performance",
            "Evaluation",
            "Results",
            "About"
        ],
        label_visibility="collapsed"
    )

    gap(2)
    st.caption("SVM · CNN · LSTM")
    st.caption("UNSW-NB15")


# =========================================================
# 5. OVERVIEW
# =========================================================
if page == "Overview":
    page_header(
        "Network intrusion detection",
        "Classify network traffic as normal or attack, and compare three trained models."
    )

    comparison = None
    try:
        comparison = pd.read_csv("model_comparison.csv")
        comparison.columns = comparison.columns.astype(str).str.strip()
    except Exception:
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        stat("Models", "3", "SVM · CNN · LSTM", BLUE)
    with c2:
        stat("Dataset", "UNSW-NB15", "Benchmark intrusion data", LAVENDER)
    with c3:
        stat("Task", "Binary", "Normal or attack", YELLOW)

    gap(1)
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Workflow")
        workflow = [
            ("01", "Upload", "Import a compatible UNSW-NB15 CSV."),
            ("02", "Preprocess", "Apply the saved scaler and feature pipeline."),
            ("03", "Detect", "Run SVM, CNN or LSTM on the records."),
            ("04", "Evaluate", "Compare metrics, matrices and curves.")
        ]
        step_colors = [BLUE, LAVENDER, YELLOW, ORANGE]
        for (number, title, description), sc in zip(workflow, step_colors):
            st.markdown(
                f"""
                <div class="step">
                    <div class="step-n" style="color:{sc};">{number}</div>
                    <div>
                        <div class="step-t">{title}</div>
                        <div class="step-d">{description}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:
        st.subheader("Accuracy by model")
        if comparison is not None and not comparison.empty:
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
                    acc = float(row["Accuracy (%)"])
                    st.markdown(
                        f"""
                        <div class="model-row">
                            <div class="model-row-head">
                                <span>{row['Model']}</span><span>{acc:.2f}%</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width:{max(0, min(100, acc))}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("Model comparison file does not contain the expected columns.")
        else:
            st.info("The model_comparison.csv file was not found.")

    gap(1)
    st.caption(
        "This prototype analyzes uploaded CSV records. It does not capture live packets, "
        "monitor a network continuously, or block attacks."
    )


# =========================================================
# 6. DETECTION
# =========================================================
elif page == "Detection":
    page_header("Detection", "Upload a compatible dataset and run a trained model.")

    model_choice = st.selectbox("Model", ["SVM", "CNN", "LSTM"])
    uploaded_file = st.file_uploader(
        "Network traffic CSV",
        type=["csv"],
        help="Use a CSV with the same original feature columns as the UNSW-NB15 training data."
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state["uploaded_df"] = raw_df

            gap(1)
            a, b, c = st.columns(3)
            with a:
                stat("Records", f"{len(raw_df):,}", "", BLUE)
            with b:
                stat("Columns", f"{len(raw_df.columns):,}", "", LAVENDER)
            with c:
                stat("File size", f"{uploaded_file.size / 1024:.1f} KB", "", YELLOW)

            with st.expander("Preview first 10 rows"):
                st.dataframe(raw_df.head(10), use_container_width=True, height=260)

            gap(0.5)
            run_col, eval_col = st.columns(2)
            with run_col:
                run_clicked = st.button(
                    "Run selected model", type="primary", use_container_width=True
                )
            with eval_col:
                eval_clicked = st.button("Evaluate all three models", use_container_width=True)

            if run_clicked:
                with st.spinner(f"Running {model_choice}..."):
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
        st.subheader("Latest summary")
        pred = st.session_state["predictions"]
        total = len(pred)
        attacks = int(np.sum(pred == 1))
        normal = int(np.sum(pred == 0))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            stat("Records", f"{total:,}", "", BLUE)
        with m2:
            stat("Attacks", f"{attacks:,}", "", ORANGE)
        with m3:
            stat("Normal", f"{normal:,}", "", BLUE)
        with m4:
            stat("Time", f"{st.session_state.get('prediction_time', 0):.2f}s",
                 st.session_state.get("selected_model", ""), LAVENDER)

        if total:
            st.progress(attacks / total, text=f"{attacks / total * 100:.1f}% predicted as attacks")

        gap(1)
        fig, ax = plt.subplots(figsize=(6.5, 2.0))
        ax.barh(["Normal", "Attack"], [normal, attacks], color=[NORMAL, ATTACK], height=0.5)
        ax.invert_yaxis()
        style_matplotlib(fig, ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Prediction table"):
            st.dataframe(st.session_state["results"], use_container_width=True, height=350)

        st.download_button(
            "Download results",
            st.session_state["results"].to_csv(index=False),
            "intrusion_detection_results.csv",
            "text/csv"
        )


# =========================================================
# 7. PERFORMANCE
# =========================================================
elif page == "Performance":
    page_header("Performance", "Held-out test results for the three trained models.")

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
        with k1:
            stat("Highest accuracy", str(best_acc["Model"]), f"{best_acc['Accuracy (%)']:.2f}%", BLUE)
        with k2:
            stat("Best attack F1", str(best_f1["Model"]), f"{best_f1['F1 Score (attack)']:.3f}", LAVENDER)
        with k3:
            stat("Fastest prediction", str(fastest_pred["Model"]),
                 f"{fastest_pred['Prediction Time (s)']:.2f}s", YELLOW)

        st.divider()
        st.subheader("Comparison")
        st.dataframe(styled_table(comp[required]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Classification metrics")
        metric_cols = [
            "Accuracy (%)", "Precision (attack)",
            "Recall (attack)", "F1 Score (attack)"
        ]
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        x = np.arange(len(comp))
        width = 0.18
        palette = [BLUE, LAVENDER, YELLOW, ORANGE]

        # Ratio-based metrics converted to percentages: one 0-100 scale.
        plot_values = comp[metric_cols].copy()
        for metric in ["Precision (attack)", "Recall (attack)", "F1 Score (attack)"]:
            plot_values[metric] = plot_values[metric] * 100

        for i, metric in enumerate(metric_cols):
            ax.bar(
                x + (i - 1.5) * width,
                plot_values[metric],
                width,
                label=metric.replace(" (%)", "").replace(" (attack)", ""),
                color=palette[i]
            )
        ax.set_xticks(x)
        ax.set_xticklabels(comp["Model"])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        style_matplotlib(fig, ax)
        ax.grid(axis="x", visible=False)
        minimal_legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        gap(1)
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Training time")
            fig, ax = plt.subplots(figsize=(5, 2.6))
            ax.bar(comp["Model"], comp["Training Time (s)"], color=LAVENDER, width=0.45)
            ax.set_ylabel("Seconds")
            style_matplotlib(fig, ax)
            ax.grid(axis="x", visible=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.subheader("Prediction time")
            fig, ax = plt.subplots(figsize=(5, 2.6))
            ax.bar(comp["Model"], comp["Prediction Time (s)"], color=YELLOW, width=0.45)
            ax.set_ylabel("Seconds")
            style_matplotlib(fig, ax)
            ax.grid(axis="x", visible=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        gap(1)
        with st.expander("How to read these metrics"):
            st.markdown(
                """
                - **Accuracy**: share of all records classified correctly.
                - **Precision**: of records predicted as attacks, how many truly were.
                - **Recall**: of actual attacks, how many were detected.
                - **F1 score**: balance between attack precision and recall.
                - **Training / prediction time**: computational cost in the recorded experiment.
                """
            )

        st.download_button(
            "Download comparison",
            comp[required].to_csv(index=False),
            "model_comparison_report.csv",
            "text/csv"
        )
        st.caption("Values are read from model_comparison.csv and reflect the recorded held-out test evaluation.")

    except FileNotFoundError:
        st.error("model_comparison.csv was not found in the repository.")
    except Exception as error:
        st.error("Unable to display model comparison.")
        st.code(str(error))


# =========================================================
# 8. EVALUATION
# =========================================================
elif page == "Evaluation":
    page_header("Evaluation", "Confusion matrices, ROC curves and precision-recall curves.")

    actual = st.session_state.get("uploaded_labels")
    all_results = st.session_state.get("all_model_results")

    if actual is None or all_results is None:
        st.info(
            "Upload the labelled UNSW-NB15 testing CSV on the Detection page "
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
            palette = {"SVM": BLUE, "CNN": LAVENDER, "LSTM": ORANGE}
            mono_cmap = LinearSegmentedColormap.from_list("pastel", [BG, "#1C2833", BLUE])

            st.subheader("Confusion matrices")
            matrix_cols = st.columns(3, gap="large")
            for index, name in enumerate(["SVM", "CNN", "LSTM"]):
                pred = all_results[name]["predictions"]
                cm = confusion_matrix(actual, pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(3.1, 2.7))
                ax.imshow(cm, cmap=mono_cmap)
                ax.set_title(name, loc="left")
                ax.set_xticks([0, 1], ["Normal", "Attack"])
                ax.set_yticks([0, 1], ["Normal", "Attack"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for row in range(2):
                    for col in range(2):
                        ax.text(
                            col, row, f"{cm[row, col]:,}",
                            ha="center", va="center",
                            color="#0A0A0B" if cm[row, col] > cm.max() * 0.6 else TEXT,
                            fontsize=9, fontweight="medium"
                        )
                style_matplotlib(fig, ax, grid=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.tight_layout()
                with matrix_cols[index]:
                    st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
            roc_col, pr_col = st.columns(2, gap="large")

            with roc_col:
                st.subheader("ROC curves")
                fig, ax = plt.subplots(figsize=(5.2, 3.4))
                for name in ["SVM", "CNN", "LSTM"]:
                    scores = all_results[name]["scores"]
                    fpr, tpr, _ = roc_curve(actual, scores)
                    auc = roc_auc_score(actual, scores)
                    ax.plot(fpr, tpr, label=f"{name}  {auc:.3f}", color=palette[name], linewidth=1.6)
                ax.plot([0, 1], [0, 1], "--", color=BORDER, linewidth=1)
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                style_matplotlib(fig, ax)
                minimal_legend(ax, loc="lower right")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with pr_col:
                st.subheader("Precision-recall curves")
                fig, ax = plt.subplots(figsize=(5.2, 3.4))
                for name in ["SVM", "CNN", "LSTM"]:
                    scores = all_results[name]["scores"]
                    precision, recall, _ = precision_recall_curve(actual, scores)
                    ap = average_precision_score(actual, scores)
                    ax.plot(recall, precision, label=f"{name}  {ap:.3f}", color=palette[name], linewidth=1.6)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                style_matplotlib(fig, ax)
                minimal_legend(ax, loc="lower left")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
            st.subheader("Metrics")
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
                "Download evaluation",
                metrics_df.to_csv(index=False),
                "advanced_evaluation_report.csv",
                "text/csv"
            )
            st.caption(
                "CNN and LSTM scores are sigmoid outputs; SVM scores are decision-function values. "
                "Curves are calculated from the uploaded dataset labels."
            )


# =========================================================
# 9. RESULTS
# =========================================================
elif page == "Results":
    page_header("Results", "Review and export the most recent predictions.")

    if "results" not in st.session_state:
        st.info("No saved results yet. Run a model on the Detection page first.")
    else:
        output = st.session_state["results"]
        pred = st.session_state["predictions"]
        total = len(pred)
        attacks = int(np.sum(pred == 1))
        normal = int(np.sum(pred == 0))

        a, b, c, d = st.columns(4)
        with a:
            stat("Records", f"{total:,}", "", BLUE)
        with b:
            stat("Attacks", f"{attacks:,}", "", ORANGE)
        with c:
            stat("Normal", f"{normal:,}", "", BLUE)
        with d:
            stat("Model", st.session_state.get("selected_model", "—"), "", LAVENDER)

        fig, ax = plt.subplots(figsize=(6.5, 2.0))
        ax.barh(["Normal", "Attack"], [normal, attacks], color=[NORMAL, ATTACK], height=0.5)
        ax.invert_yaxis()
        ax.set_xlabel("Records")
        style_matplotlib(fig, ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()
        st.subheader("Records")
        filter_choice = st.selectbox("Filter", ["All records", "Attack only", "Normal only"])
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
# 10. ABOUT
# =========================================================
elif page == "About":
    page_header(
        "About",
        "AI-based network intrusion detection · M.Tech, AI / Machine Learning"
    )

    st.subheader("Objective")
    st.write(
        "A dataset-based prototype that classifies network traffic records as "
        "normal or attack, and compares three supervised learning models."
    )

    gap(1)
    st.subheader("Models")
    m1, m2, m3 = st.columns(3)
    with m1:
        stat("SVM", "Support Vector Machine", "Supervised machine learning", BLUE)
    with m2:
        stat("CNN", "Convolutional Neural Network", "Deep learning", LAVENDER)
    with m3:
        stat("LSTM", "Long Short-Term Memory", "Recurrent neural network", ORANGE)

    st.subheader("Dataset")
    st.write(
        "UNSW-NB15 is used as the network traffic dataset. "
        "The prepared CSV files contain labelled normal and attack records."
    )

    gap(1)
    st.subheader("Stack")
    st.write("Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Matplotlib · Streamlit")

    gap(1)
    st.caption(
        "Scope: this prototype predicts labels for uploaded CSV records. It does not perform "
        "live packet capture, real-time monitoring, or automatic attack blocking. CNN and LSTM "
        "receive feature-wise reshaped records, not a verified time-ordered flow sequence."
    )


# =========================================================
# 11. FOOTER
# =========================================================
st.markdown('<div class="footer">NETGUARD · UNSW-NB15 · SVM · CNN · LSTM</div>', unsafe_allow_html=True)
