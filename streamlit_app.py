import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.mixture import GaussianMixture

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA (solo una vez y primero)
# ------------------------------------------------------------
st.set_page_config(page_title="IA: SVM Lineal + GMM", layout="wide")

st.title("Tarea: Supervisado (SVM lineal) + No Supervisado (GMM)")

# ------------------------------------------------------------
# Cargar dataset Iris
# ------------------------------------------------------------
data = datasets.load_iris()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

# ------------------------------------------------------------
# SIDEBAR CONTROLES
# ------------------------------------------------------------
st.sidebar.title("Controles")

modo = st.sidebar.radio(
    "Selecciona modo:",
    ["Supervisado", "No Supervisado"]
)

st.sidebar.title("Zona de Exportación")
st.sidebar.write("Aquí puedes agregar futuras funciones de exportación.")


# ------------------------------------------------------------
# INSTRUCCIONES
# ------------------------------------------------------------
with st.expander("Instrucciones"):
    st.write("""
    - Este proyecto muestra **SVM lineal** como modelo supervisado.  
    - También incluye **Gaussian Mixture Model (GMM)** como modelo no supervisado.  
    - Puedes probar predicciones manuales y ver métricas completas.  
    """)

# ------------------------------------------------------------
# MOSTRAR DATASET
# ------------------------------------------------------------
st.subheader("Vista previa del dataset")
st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------
# MODO SUPERVISADO (SVM LINEAL)
# ------------------------------------------------------------
if modo == "Supervisado":
    st.markdown("## 🔵 Modo Supervisado — SVM (Kernel Lineal)")

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Entrenar SVM Lineal
    model_svm = SVC(kernel="linear", probability=True)
    model_svm.fit(X_train, y_train)

    # Predicciones
    y_pred = model_svm.predict(X_test)
    y_prob = model_svm.predict_proba(X_test)

    # Métricas
    st.subheader("Métricas (conjunto test)")

    st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision (macro)**: {precision_score(y_test, y_pred, average='macro'):.4f}")
    st.write(f"**Recall (macro)**: {recall_score(y_test, y_pred, average='macro'):.4f}")
    st.write(f"**F1-Score (macro)**: {f1_score(y_test, y_pred, average='macro'):.4f}")

    # ------------------------------------------------------------
    # GRAFICO — SVM (SE MANTIENE)
    # ------------------------------------------------------------
    st.subheader("📊 Visualizaciones Supervisado")

    fig_svm, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    st.pyplot(fig_svm)

    # ------------------------------------------------------------
    # Matriz de Confusión
    # ------------------------------------------------------------
    st.subheader("Matriz de confusión (conjunto test)")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    st.pyplot(fig_cm)

    # ------------------------------------------------------------
    # PREDICCIÓN MANUAL
    # ------------------------------------------------------------
    st.subheader("Probar una predicción manualmente")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal length", 4.3, 7.9, 5.1)
        sw = st.slider("Sepal width", 2.0, 4.4, 3.5)

    with col2:
        pl = st.slider("Petal length", 1.0, 6.9, 1.4)
        pw = st.slider("Petal width", 0.1, 2.5, 0.2)

    entrada = np.array([[sl, sw, pl, pw]])
    pred = model_svm.predict(entrada)[0]
    prob = model_svm.predict_proba(entrada)

    st.write(f"**Predicción:** {pred} - {data.target_names[pred]}")

    st.write("**Probabilidades:**")
    st.json(prob.tolist())


# ------------------------------------------------------------
# MODO NO SUPERVISADO (GMM)
# ------------------------------------------------------------
else:
    st.markdown("## 🔴 Modo No Supervisado — Gaussian Mixture Model (GMM)")

    gmm = GaussianMixture(n_components=3, random_state=42)
    clusters = gmm.fit_predict(X)

    # GRAFICO GMM
    st.subheader("📊 Gráfico No Supervisado (GMM)")

    fig_gmm, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap="rainbow")
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    st.pyplot(fig_gmm)

    st.write("Clusters asignados:")
    st.write(clusters)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.write("---")
st.write("Made with Streamlit")
