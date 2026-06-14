import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    return df, data

@st.cache_resource
def train_model(test_size, random_state):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test, data

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("🩺 Breast Cancer Classification")
st.markdown("Built with **Random Forest** on sklearn's `load_breast_cancer()` dataset.")

st.sidebar.header("Model Settings")
test_size    = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

df, raw_data = load_data()
clf, X_train, X_test, y_train, y_test, data = train_model(test_size, int(random_state))
y_pred = clf.predict(X_test)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Explorer", "📈 Evaluation", "🔍 Feature Importance", "🔬 Predict"]
)

with tab1:
    st.subheader("a) Dataset Overview")
    st.write(f"**Samples:** {df.shape[0]}  |  **Features:** {df.shape[1]-2}")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("b) Class Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = df["diagnosis"].value_counts()
    ax.bar(counts.index, counts.values, color=["#e74c3c", "#2ecc71"])
    ax.set_xlabel("Diagnosis"); ax.set_ylabel("Count")
    ax.set_title("Malignant vs Benign")
    st.pyplot(fig)
    st.write(f"Malignant: **{(df.target==0).sum()}**  |  Benign: **{(df.target==1).sum()}**")

    st.subheader("c) Train / Test Split")
    st.write(f"Train: **{X_train.shape[0]}** samples  |  Test: **{X_test.shape[0]}** samples")

with tab2:
    st.subheader("e) Model Evaluation")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=data.target_names, yticklabels=data.target_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("**Precision & Recall Report**")
        report = classification_report(
            y_test, y_pred, target_names=data.target_names, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ── Tab 3 : Feature Importance ────────────────────────────────────────────────
with tab3:
    st.subheader("f) Top-10 Contributing Features")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    feat_df = pd.DataFrame({
        "Feature": np.array(data.feature_names)[indices],
        "Importance": importances[indices]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Top-10 Feature Importances")
    st.pyplot(fig)

    st.markdown("""
    **Why these features?**
    - **worst radius / area / perimeter** — larger tumour size → higher malignancy risk.
    - **worst / mean concave points & concavity** — irregular, jagged cell boundaries
      are hallmarks of malignant cells.
    - Random Forest ranks features by Gini-impurity reduction across 100 trees,
      making the scores a robust indicator of true discriminative power.
    """)

# ── Tab 4 : Live Prediction ───────────────────────────────────────────────────
with tab4:
    st.subheader("🔬 Predict on Custom Input")
    st.markdown("Adjust the **top-5 important features** below; the rest use median values.")

    top5_idx   = np.argsort(clf.feature_importances_)[::-1][:5]
    top5_names = data.feature_names[top5_idx]
    medians    = np.median(X_train, axis=0)
    sample     = medians.copy()

    cols = st.columns(5)
    for col, idx, name in zip(cols, top5_idx, top5_names):
        lo, hi = float(X_train[:, idx].min()), float(X_train[:, idx].max())
        sample[idx] = col.slider(name, lo, hi, float(medians[idx]), key=str(idx))

    if st.button("Predict"):
        pred       = clf.predict([sample])[0]
        proba      = clf.predict_proba([sample])[0]
        label      = data.target_names[pred]
        confidence = proba[pred] * 100
        color      = "🟢" if pred == 1 else "🔴"
        st.markdown(f"### {color} Prediction: **{label}**")
        st.write(f"Confidence — Malignant: `{proba[0]*100:.1f}%` | Benign: `{proba[1]*100:.1f}%`")
