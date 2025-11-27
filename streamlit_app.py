# streamlit_app.py
import streamlit as st
st.set_page_config(page_title="IA: SVM (Lineal) + GMM", layout="wide")

import io
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, davies_bouldin_score
)
from sklearn.mixture import GaussianMixture

# -----------------------
# Utilidades
# -----------------------
plt.rcParams.update({"figure.max_open_warning": 0})

def fig_to_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def pack_supervised(model, scaler):
    return {"model": model, "scaler": scaler}

def pack_unsupervised(model, scaler, pca=None):
    return {"model": model, "scaler": scaler, "pca": pca}

# -----------------------
# Cargar dataset Iris
# -----------------------
iris = datasets.load_iris()
X = iris.data.copy()
y = iris.target.copy()
feature_names = iris.feature_names
target_names = iris.target_names
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# -----------------------
# Sidebar / modo
# -----------------------
st.title("Tarea: Supervisado (SVM lineal) + No Supervisado (GMM)")
st.sidebar.header("Controles")
modo = st.sidebar.radio("Selecciona modo:", ["Supervisado", "No Supervisado", "Zona de Exportación", "Instrucciones"])

# Global store
_export_store = {"supervised": None, "unsupervised": None}

# -----------------------
# Instrucciones
# -----------------------
if modo == "Instrucciones":
    st.header("Instrucciones rápidas")
    st.markdown("""
    - **Supervisado**: entrena SVM (kernel lineal), muestra métricas, gráfico 2D (PCA), matriz de confusión y predicción interactiva.  
    - **No Supervisado**: entrena GMM, muestra métricas, gráfico 2D (PCA) coloreado por cluster.  
    - **Zona de Exportación**: descarga JSON, CSV, archivos `.pkl` y PNG de gráficos.  
    Dataset usado: **Iris**.
    """)

# -----------------------
# Supervisado
# -----------------------
if modo == "Supervisado":
    st.header("🔵 Modo Supervisado — SVM (Kernel Lineal)")
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    # Split + Escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # Métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1_score": float(f1_score(y_test, y_pred, average="macro"))
    }

    st.subheader("Métricas (conjunto test)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (macro)", f"{metrics['precision']:.4f}")
    c3.metric("Recall (macro)", f"{metrics['recall']:.4f}")
    c4.metric("F1-Score (macro)", f"{metrics['f1_score']:.4f}")

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    Xp = pca.fit_transform(X_all_s)

    st.subheader("📊 Visualizaciones Supervisado")
    plot_col1, plot_col2 = st.columns(2)

    fig_true, ax_true = plt.subplots(figsize=(5,4))
    ax_true.scatter(Xp[:,0], Xp[:,1], c=y, cmap="tab10", s=50, edgecolor="k")
    ax_true.set_xlabel("PCA 1"); ax_true.set_ylabel("PCA 2")
    ax_true.set_title("Iris (PCA 2D) - Etiquetas verdaderas")
    plot_col1.pyplot(fig_true)

    fig_dec, ax_dec = plt.subplots(figsize=(5,4))
    svc_pca = SVC(kernel="linear")
    Xp_train = pca.transform(scaler.transform(X_train))
    svc_pca.fit(Xp_train, y_train)
    x_min, x_max = Xp[:,0].min()-1, Xp[:,0].max()+1
    y_min, y_max = Xp[:,1].min()-1, Xp[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
    Z = svc_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_dec.contourf(xx, yy, Z, alpha=0.15, cmap='tab10')
    ax_dec.scatter(Xp[:,0], Xp[:,1], c=y, cmap='tab10', s=30, edgecolor='k')
    ax_dec.set_xlabel("PCA 1"); ax_dec.set_ylabel("PCA 2")
    ax_dec.set_title("Decisión (PCA 2D) & puntos (true labels)")
    plot_col2.pyplot(fig_dec)

    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion matrix")
    st.pyplot(fig_cm)

    # Predicción interactiva
    st.subheader("Probar predicción manualmente")
    sliders = []
    for fname in feature_names:
        val = st.slider(fname, float(df[fname].min()), float(df[fname].max()), float(df[fname].mean()))
        sliders.append(val)
    sample_s = scaler.transform([sliders])
    pred_idx = int(model.predict(sample_s)[0])
    pred_label = target_names[pred_idx]
    st.write("**Predicción:**", pred_idx, "-", pred_label)

    # Guardar para exportación
    _export_store["supervised"] = {
        "model_obj": pack_supervised(model, scaler),
        "metrics": metrics,
        "last_input": sliders,
        "last_output": {"class": pred_idx, "label": pred_label},
        "figs": {
            "pca_true": fig_to_bytes(fig_true),
            "decision": fig_to_bytes(fig_dec),
            "confusion": fig_to_bytes(fig_cm)
        }
    }

# -----------------------
# No Supervisado
# -----------------------
elif modo == "No Supervisado":
    st.header("🔴 Modo No Supervisado — GMM")
    st.subheader("Vista previa dataset")
    st.dataframe(df.head())

    n_components = st.slider("Número de clusters GMM", 2, 6, 3)
    scaler_unsup = StandardScaler().fit(X)
    X_s = scaler_unsup.transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels_gmm = gmm.fit_predict(X_s)

    sil, db = (float(silhouette_score(X_s, labels_gmm)), float(davies_bouldin_score(X_s, labels_gmm))) if len(set(labels_gmm))>1 else (None,None)

    st.subheader("Métricas de clustering")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Silhouette Score", f"{sil:.4f}" if sil else "N/A")
    mcol2.metric("Davies-Bouldin", f"{db:.4f}" if db else "N/A")

    Xp_unsup = PCA(n_components=2, random_state=42).fit_transform(X_s)
    fig_gmm, ax_gmm = plt.subplots(figsize=(6,4))
    ax_gmm.scatter(Xp_unsup[:,0], Xp_unsup[:,1], c=labels_gmm, cmap='tab10', s=50, edgecolor='k')
    ax_gmm.set_xlabel("PCA 1"); ax_gmm.set_ylabel("PCA 2")
    ax_gmm.set_title("GMM clusters (PCA 2D)")
    st.pyplot(fig_gmm)

    _export_store["unsupervised"] = {
        "model_obj": pack_unsupervised(gmm, scaler_unsup, pca=None),
        "metrics": {"silhouette_score": sil, "davies_bouldin": db},
        "cluster_labels": labels_gmm.tolist(),
        "n_components": int(n_components),
        "figs": {"gmm_pca": fig_to_bytes(fig_gmm)}
    }

# -----------------------
# Zona de Exportación
# -----------------------
elif modo == "Zona de Exportación":
    st.header("📦 Zona de Exportación")
    st.markdown("Primero ejecuta Supervisado y No Supervisado para habilitar descargas.")

    # Dataset CSV
    st.download_button("📄 Descargar Dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_iris.csv", mime="text/csv")

    # Supervisado
    sup = _export_store.get("supervised")
    if sup:
        st.download_button("📊 Métricas Supervisado", json.dumps(sup["metrics"], indent=4),
                           file_name="metrics_supervised.json", mime="application/json")
        st.download_button("📦 Modelo SVM", pickle.dumps(sup["model_obj"]),
                           file_name="svm_model.pkl", mime="application/octet-stream")
        for name, fig_bytes in sup["figs"].items():
            st.download_button(f"🖼️ {name}", fig_bytes, file_name=f"{name}.png", mime="image/png")

    # No Supervisado
    uns = _export_store.get("unsupervised")
    if uns:
        st.download_button("📊 Métricas No Supervisado", json.dumps(uns["metrics"], indent=4),
                           file_name="metrics_unsupervised.json", mime="application/json")
        st.download_button("📦 Modelo GMM", pickle.dumps(uns["model_obj"]),
                           file_name="gmm_model.pkl", mime="application/octet-stream")
        for name, fig_bytes in uns["figs"].items():
            st.download_button(f"🖼️ {name}", fig_bytes, file_name=f"{name}.png", mime="image/png")
