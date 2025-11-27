# streamlit_app.py
# st.set_page_config must be the first Streamlit command
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
# Util / configuraciones
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
# Cargar dataset (Iris)
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
modo = st.sidebar.radio("Selecciona modo:",
                        ["Supervisado", "No Supervisado", "Zona de Exportación", "Instrucciones"])

# Global holders for figures/objects to export
_export_store = {
    "supervised": None,   # dict with model_obj, metrics, last_input, last_output, figs (bytes)
    "unsupervised": None  # dict with model_obj, metrics, cluster_labels, figs (bytes)
}

# -----------------------
# Instrucciones
# -----------------------
if modo == "Instrucciones":
    st.header("Instrucciones rápidas")
    st.markdown("""
    - **Supervisado**: entrena SVM (kernel lineal), muestra métricas, gráfico 2D (PCA), matriz de confusión y predicción interactiva.  
    - **No Supervisado**: entrena GMM, muestra métricas, gráfico 2D (PCA) coloreado por cluster.  
    - **Zona de Exportación**: descarga JSON (para React), CSV, archivos `.pkl` y PNG de gráficos.  
    Dataset usado: **Iris**.
    """)

# -----------------------
# MODO SUPERVISADO
# -----------------------
if modo == "Supervisado":
    st.header("🔵 Modo Supervisado — SVM (Kernel Lineal)")
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    # split + scale + train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1_score": float(f1_score(y_test, y_pred, average="macro"))
    }

    # mostrar métricas
    st.subheader("Métricas (conjunto test)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (macro)", f"{metrics['precision']:.4f}")
    c3.metric("Recall (macro)", f"{metrics['recall']:.4f}")
    c4.metric("F1-Score (macro)", f"{metrics['f1_score']:.4f}")

    # PCA 2D para visualizar
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)  # scale full dataset for consistent plotting
    Xp = pca.fit_transform(X_all_s)

    st.subheader("📊 Visualizaciones Supervisado")
    plot_col1, plot_col2 = st.columns(2)

    # Left: PCA scatter (true labels)
    fig_true, ax_true = plt.subplots(figsize=(5,4))
    sc = ax_true.scatter(Xp[:,0], Xp[:,1], c=y, cmap="tab10", s=50, edgecolor="k")
    ax_true.set_xlabel("PCA 1"); ax_true.set_ylabel("PCA 2")
    ax_true.set_title("Iris (PCA 2D) - Etiquetas verdaderas")
    plot_col1.pyplot(fig_true)

    # Right: decision regions (approx. via SVM on PCA projection) + points
    fig_dec, ax_dec = plt.subplots(figsize=(5,4))
    # train a helper SVM on PCA space (for plotting regions)
    svc_pca = SVC(kernel="linear")
    Xp_train = pca.transform(scaler.transform(X_train))
    svc_pca.fit(Xp_train, y_train)
    # grid
    x_min, x_max = Xp[:,0].min() - 1, Xp[:,0].max() + 1
    y_min, y_max = Xp[:,1].min() - 1, Xp[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svc_pca.predict(grid).reshape(xx.shape)
    ax_dec.contourf(xx, yy, Z, alpha=0.15, cmap='tab10')
    ax_dec.scatter(Xp[:,0], Xp[:,1], c=y, cmap='tab10', s=30, edgecolor='k')
    ax_dec.set_xlabel("PCA 1"); ax_dec.set_ylabel("PCA 2")
    ax_dec.set_title("Decisión (PCA 2D) & puntos (true labels)")
    plot_col2.pyplot(fig_dec)

    # Confusion matrix (test set)
    st.subheader("Matriz de confusión (conjunto test)")
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion matrix")
    st.pyplot(fig_cm)

    # Predicción interactiva
    st.subheader("Probar una predicción manualmente")
    sl = st.slider("Sepal length", float(df[feature_names[0]].min()), float(df[feature_names[0]].max()), float(df[feature_names[0]].mean()))
    sw = st.slider("Sepal width", float(df[feature_names[1]].min()), float(df[feature_names[1]].max()), float(df[feature_names[1]].mean()))
    pl = st.slider("Petal length", float(df[feature_names[2]].min()), float(df[feature_names[2]].max()), float(df[feature_names[2]].mean()))
    pw = st.slider("Petal width", float(df[feature_names[3]].min()), float(df[feature_names[3]].max()), float(df[feature_names[3]].mean()))

    sample = np.array([[sl, sw, pl, pw]])
    sample_s = scaler.transform(sample)
    pred_idx = int(model.predict(sample_s)[0])
    pred_label = target_names[pred_idx]
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(sample_s).tolist()
        except Exception:
            probs = None

    st.write("**Predicción:**", pred_idx, "-", pred_label)
    if probs:
        st.write("Probabilidades:", probs)

    # Guardar todo para exportación
    _export_store["supervised"] = {
        "model_obj": pack_supervised(model, scaler),
        "metrics": metrics,
        "last_input": [float(sl), float(sw), float(pl), float(pw)],
        "last_output": {"class": pred_idx, "label": pred_label},
        "figs": {
            "pca_true": fig_to_bytes(fig_true),
            "decision": fig_to_bytes(fig_dec),
            "confusion": fig_to_bytes(fig_cm)
        }
    }

# -----------------------
# MODO NO SUPERVISADO
# -----------------------
elif modo == "No Supervisado":
    st.header("🔴 Modo No Supervisado — Gaussian Mixture Model (GMM)")
    st.subheader("Vista previa del dataset (sin etiquetas)")
    st.dataframe(df.head())

    n_components = st.slider("Número de componentes (clusters) GMM", 2, 6, 3)

    scaler_unsup = StandardScaler().fit(X)
    X_s = scaler_unsup.transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels_gmm = gmm.fit_predict(X_s)

    # métricas
    if len(set(labels_gmm)) > 1:
        sil = float(silhouette_score(X_s, labels_gmm))
        db = float(davies_bouldin_score(X_s, labels_gmm))
    else:
        sil, db = None, None

    st.subheader("Métricas de clustering")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")
    mcol2.metric("Davies-Bouldin", f"{db:.4f}" if db is not None else "N/A")

    # PCA 2D para la visualización (usar X_s)
    pca_unsup = PCA(n_components=2, random_state=42)
    Xp_unsup = pca_unsup.fit_transform(X_s)

    st.subheader("📊 Visualización GMM (PCA 2D)")
    fig_gmm, ax_gmm = plt.subplots(figsize=(6,4))
    ax_gmm.scatter(Xp_unsup[:,0], Xp_unsup[:,1], c=labels_gmm, cmap='tab10', s=50, edgecolor='k')
    ax_gmm.set_xlabel("PCA 1"); ax_gmm.set_ylabel("PCA 2")
    ax_gmm.set_title("GMM clusters (PCA 2D)")
    st.pyplot(fig_gmm)

    # Guardar para exportación
    _export_store["unsupervised"] = {
        "model_obj": pack_unsupervised(gmm, scaler_unsup, pca_unsup),
        "metrics": {"silhouette_score": sil, "davies_bouldin": db},
        "cluster_labels": labels_gmm.tolist(),
        "n_components": int(n_components),
        "figs": {
            "gmm_pca": fig_to_bytes(fig_gmm)
        }
    }

# -----------------------
# ZONA DE EXPORTACIÓN
# -----------------------
elif modo == "Zona de Exportación":
    st.header("📦 Zona de Exportación — JSON, CSV, PNG y archivos .pkl")
    st.markdown("Primero ejecuta los modos **Supervisado** y **No Supervisado** para generar métricas y modelos. Después podrás descargar todos los archivos aquí.")

    # 1) dataset CSV
    st.subheader("🔹 Dataset")
    df_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Descargar Dataset (CSV)", df_csv, file_name="dataset_iris.csv", mime="text/csv")

    st.write("---")

    # Export supervised if exists
    sup = _export_store.get("supervised")
    if sup is not None:
        st.subheader("🔵 Supervisado (SVM) - Exportar")

        # JSON for React
        sup_json = {
            "model_type": "Supervised",
            "model_name": "SVM (Kernel Lineal)",
            "metrics": sup["metrics"],
            "current_prediction": {
                "input": sup["last_input"],
                "output_class": sup["last_output"]["class"],
                "output_label": sup["last_output"]["label"]
            }
        }
        st.download_button("📥 Descargar JSON Supervisado", data=json.dumps(sup_json, indent=4),
                           file_name="supervised_svm.json", mime="application/json")

        # metrics JSON
        st.download_button("📊 Descargar Métricas Supervisado (JSON)", data=json.dumps({"supervised_metrics": sup["metrics"]}, indent=4),
                           file_name="metrics_supervised.json", mime="application/json")

        # pkl (model + scaler)
        pkl_sup = pickle.dumps(sup["model_obj"])
        st.download_button("📦 Descargar .pkl Supervisado", data=pkl_sup,
                           file_name="svm_supervised.pkl", mime="application/octet-stream")

        # PNGs (if present)
        figs = sup.get("figs", {})
        if "pca_true" in figs:
            st.download_button("🖼️ Descargar PCA (true labels) PNG", data=figs["pca_true"],
                               file_name="svm_pca_true.png", mime="image/png")
        if "decision" in figs:
            st.download_button("🖼️ Descargar Decision Regions PNG", data=figs["decision"],
                               file_name="svm_decision.png", mime="image/png")
        if "confusion" in figs:
            st.download_button("🖼️ Descargar Confusion Matrix PNG", data=figs["confusion"],
                               file_name="svm_confusion.png", mime="image/png")
    else:
        st.info("Primero ejecuta el modo Supervisado para habilitar sus descargas.")

    st.write("---")

    # Export unsupervised if exists
    uns = _export_store.get("unsupervised")
    if uns is not None:
        st.subheader("🔴 No Supervisado (GMM) - Exportar")

        uns_json = {
            "model_type": "Unsupervised",
            "algorithm": "GaussianMixture",
            "parameters": {"n_components": uns["n_components"]},
            "metrics": uns["metrics"],
            "cluster_labels": uns["cluster_labels"]
        }
        st.download_button("📥 Descargar JSON No Supervisado", data=json.dumps(uns_json, indent=4),
                           file_name="unsupervised_gmm.json", mime="application/json")

        st.download_button("📊 Descargar Métricas No Supervisado (JSON)", data=json.dumps({"unsupervised_metrics": uns["metrics"]}, indent=4),
                           file_name="metrics_unsupervised.json", mime="application/json")

        pkl_uns = pickle.dumps(uns["model_obj"])
        st.download_button("📦 Descargar .pkl No Supervisado", data=pkl_uns,
                           file_name="gmm_unsupervised.pkl", mime="application/octet-stream")

        figs_u = uns.get("figs", {})
        if "gmm_pca" in figs_u:
            st.download_button("🖼️ Descargar Gráfico GMM (PNG)", data=figs_u["gmm_pca"],
                               file_name="gmm_pca.png", mime="image/png")
    else:
        st.info("Primero ejecuta el modo No Supervisado para habilitar sus descargas.")
