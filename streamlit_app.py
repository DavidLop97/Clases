# streamlit_app.py
# NOTA: st.set_page_config MUST be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="IA: SVM (Lineal) + GMM", layout="wide")

import json
import pickle
import io

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
# Cargar dataset (Iris)
# -----------------------
iris = datasets.load_iris()
X = iris.data          # features
y = iris.target        # labels
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

# -----------------------
# Instrucciones
# -----------------------
if modo == "Instrucciones":
    st.header("Instrucciones rápidas")
    st.markdown("""
    - **Supervisado**: entrena SVM (kernel lineal), muestra métricas, gráfico 2D (PCA), matriz de confusión y predicción interactiva.  
    - **No Supervisado**: entrena GMM, muestra métricas, gráfico 2D (PCA) coloreado por cluster.  
    - **Zona de Exportación**: descarga JSON (para React) y archivos `.pkl`.  
    Dataset usado: **Iris**.
    """)

# -----------------------
# Funciones utilitarias
# -----------------------
def pack_supervised(model, scaler):
    return {"model": model, "scaler": scaler}

def pack_unsupervised(model, scaler, pca=None):
    return {"model": model, "scaler": scaler, "pca": pca}

def train_svm_full(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1_score": float(f1_score(y_test, y_pred, average="macro"))
    }
    return {"model": model, "scaler": scaler}, metrics, (X_test_s, y_test, y_pred, X_train, X_test, y_train, y_test)

def pca_for_plot(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(X)
    return pca, Xp

def plot_scatter_2d(ax, X2, labels, title="", cmap="tab10", alpha=0.8):
    sc = ax.scatter(X2[:,0], X2[:,1], c=labels, cmap=cmap, s=50, alpha=alpha, edgecolor='k')
    ax.set_title(title)
    return sc

def plot_decision_regions(ax, model, pca, scaler, X, y, grid_steps=200):
    # For visualization train a separate SVM on PCA projection of training data
    X_s = scaler.transform(X)
    Xp = pca.transform(X_s)
    x_min, x_max = Xp[:,0].min() - 1, Xp[:,0].max() + 1
    y_min, y_max = Xp[:,1].min() - 1, Xp[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_steps),
                         np.linspace(y_min, y_max, grid_steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # we need a classifier in PCA space: fit a lightweight SVM on Xp (this is just for plotting)
    svc_pca = SVC(kernel="linear")
    svc_pca.fit(Xp, y)
    Z = svc_pca.predict(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.15, cmap='tab10')

# -----------------------
# Modo Supervisado
# -----------------------
if modo == "Supervisado":
    st.header("🔵 Modo Supervisado — SVM (Kernel Lineal)")
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    # Entrenar modelo (completo)
    packed, metrics, testinfo = train_svm_full(X, y, test_size=0.2)
    model = packed["model"]
    scaler = packed["scaler"]
    X_test_s, y_test, y_pred, X_train_raw, X_test_raw, y_train, y_test_raw = testinfo

    # Mostrar métricas con st.metric
    st.subheader("Métricas (conjunto test)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (macro)", f"{metrics['precision']:.4f}")
    c3.metric("Recall (macro)", f"{metrics['recall']:.4f}")
    c4.metric("F1-Score (macro)", f"{metrics['f1_score']:.4f}")

    # Preparar PCA 2D para gráficos (usar scaler)
    pca_plot, Xp_all = pca_for_plot(X)
    # For decision regions we'll fit a small SVM on PCA projection in plot_decision_regions

    # Gráficos en dos columnas
    st.subheader("📊 Visualizaciones Supervisado")
    col_plot1, col_plot2 = st.columns(2)

    # Left: scatter true labels in PCA space
    fig1, ax1 = plt.subplots(figsize=(5,4))
    plot_scatter_2d(ax1, Xp_all, y, title="Iris (PCA 2D) - Etiquetas verdaderas", cmap="tab10")
    ax1.set_xlabel("PCA 1"); ax1.set_ylabel("PCA 2")
    col_plot1.pyplot(fig1)

    # Right: decision regions + test points
    fig2, ax2 = plt.subplots(figsize=(5,4))
    # decision regions (train on PCA projection, plotting only)
    try:
        plot_decision_regions(ax2, model, pca_plot, scaler, X, y)
    except Exception:
        pass
    # overlay test points (projected)
    Xs_all = scaler.transform(X)
    Xp_all_scaled = pca_plot.transform(Xs_all)
    ax2.scatter(Xp_all_scaled[:,0], Xp_all_scaled[:,1], c=y, cmap='tab10', s=30, edgecolor='k')
    ax2.set_title("Decisión (PCA 2D) & puntos (true labels)")
    ax2.set_xlabel("PCA 1"); ax2.set_ylabel("PCA 2")
    col_plot2.pyplot(fig2)

    # Matriz de confusión (usando test set)
    st.subheader("Matriz de confusión (conjunto test)")
    fig3, ax3 = plt.subplots(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax3, cmap="Blues", colorbar=False)
    ax3.set_title("Confusion matrix")
    st.pyplot(fig3)

    # Predicción interactiva
    st.subheader("Probar una predicción manualmente")
    sl = st.slider("Sepal length", float(df[feature_names[0]].min()), float(df[feature_names[0]].max()), float(df[feature_names[0]].mean()))
    sw = st.slider("Sepal width", float(df[feature_names[1]].min()), float(df[feature_names[1]].max()), float(df[feature_names[1]].mean()))
    pl = st.slider("Petal length", float(df[feature_names[2]].min()), float(df[feature_names[2]].max()), float(df[feature_names[2]].mean()))
    pw = st.slider("Petal width", float(df[feature_names[3]].min()), float(df[feature_names[3]].max()), float(df[feature_names[3]].mean()))

    sample = np.array([[sl, sw, pl, pw]])
    sample_s = scaler.transform(sample)
    pred_label_idx = int(model.predict(sample_s)[0])
    pred_label = target_names[pred_label_idx]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_s).tolist()

    st.write("**Predicción:**", pred_label_idx, "-", pred_label)
    if proba:
        st.write("Probabilidades:", proba)

    # Guardar en session_state para exportación
    st.session_state["supervised"] = {
        "model_obj": pack_supervised(model, scaler),
        "metrics": metrics,
        "last_input": [float(sl), float(sw), float(pl), float(pw)],
        "last_output": {"class": pred_label_idx, "label": pred_label}
    }

# -----------------------
# Modo No Supervisado
# -----------------------
elif modo == "No Supervisado":
    st.header("🔴 Modo No Supervisado — Gaussian Mixture Model (GMM)")
    st.subheader("Vista previa del dataset (sin etiquetas)")
    st.dataframe(df.head())

    n_components = st.slider("Número de componentes (clusters) GMM", 2, 6, 3)

    scaler_unsup = StandardScaler()
    X_s = scaler_unsup.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels_gmm = gmm.fit_predict(X_s)

    # métricas
    if len(set(labels_gmm)) > 1:
        sil = float(silhouette_score(X_s, labels_gmm))
        db = float(davies_bouldin_score(X_s, labels_gmm))
    else:
        sil = None
        db = None

    st.subheader("Métricas de clustering")
    cc1, cc2 = st.columns(2)
    cc1.metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")
    cc2.metric("Davies-Bouldin", f"{db:.4f}" if db is not None else "N/A")

    # PCA 2D for plotting
    pca_unsup, Xp_unsup = pca_for_plot(X_s)

    st.subheader("📊 Visualización GMM (PCA 2D)")
    fig_u, ax_u = plt.subplots(figsize=(6,4))
    sc = ax_u.scatter(Xp_unsup[:,0], Xp_unsup[:,1], c=labels_gmm, cmap='tab10', s=50, edgecolor='k')
    ax_u.set_xlabel("PCA 1"); ax_u.set_ylabel("PCA 2")
    ax_u.set_title("GMM clusters (PCA 2D)")
    st.pyplot(fig_u)

    # Guardar en session_state
    st.session_state["unsupervised"] = {
        "model_obj": pack_unsupervised(gmm, scaler_unsup, pca_unsup),
        "metrics": {"silhouette_score": sil, "davies_bouldin": db},
        "cluster_labels": labels_gmm.tolist(),
        "n_components": int(n_components)
    }

# -----------------------
# Zona de Exportación
# -----------------------
elif modo == "Zona de Exportación":
    st.header("📦 Zona de Exportación — JSON y archivos .pkl")

    # Supervisado
    if "supervised" in st.session_state:
        sup = st.session_state["supervised"]
        st.subheader("🔵 Supervisado — Exportar (SVM)")

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
        st.download_button(
            label="Descargar JSON Supervisado",
            data=json.dumps(sup_json, indent=4),
            file_name="supervised_svm.json",
            mime="application/json"
        )
        # pickle bytes (model + scaler)
        pkl_sup = pickle.dumps(sup["model_obj"])
        st.download_button(
            label="Descargar .pkl Supervisado",
            data=pkl_sup,
            file_name="svm_supervised.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("Primero ejecuta el modo Supervisado para generar métricas y predicción.")

    st.write("---")

    # No supervisado
    if "unsupervised" in st.session_state:
        uns = st.session_state["unsupervised"]
        st.subheader("🔴 No Supervisado — Exportar (GMM)")

        uns_json = {
            "model_type": "Unsupervised",
            "algorithm": "GaussianMixture",
            "parameters": {"n_components": uns["n_components"]},
            "metrics": uns["metrics"],
            "cluster_labels": uns["cluster_labels"]
        }
        st.download_button(
            label="Descargar JSON No Supervisado",
            data=json.dumps(uns_json, indent=4),
            file_name="unsupervised_gmm.json",
            mime="application/json"
        )

        pkl_uns = pickle.dumps(uns["model_obj"])
        st.download_button(
            label="Descargar .pkl No Supervisado",
            data=pkl_uns,
            file_name="gmm_unsupervised.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("Primero ejecuta el modo No Supervisado para generar clusters y métricas.")
