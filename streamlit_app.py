import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================
# CONFIGURACIÓN INICIAL (PRIMERA LÍNEA SÍ O SÍ)
# ============================================
st.set_page_config(page_title="IA: SVM (Lineal) + GMM", layout="wide")

# ============================================
# CARGA DEL DATASET
# ============================================
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

# ============================================
# ENTRENAMIENTO DE MODELOS
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM LINEAL
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# GMM
gmm_model = GaussianMixture(n_components=3, random_state=42)
gmm_model.fit(X)

# ============================================
# UI PRINCIPAL
# ============================================
st.title("Tarea: Supervisado (SVM lineal) + No Supervisado (GMM)")

st.sidebar.title("Controles")
modo = st.sidebar.radio("Selecciona modo:", ["Supervisado", "No Supervisado", "Exportación"])
st.sidebar.markdown("---")
st.sidebar.write("Zona de Exportación disponible en modo correspondiente.")

st.subheader("Instrucciones rápidas")
st.write("""
**Supervisado:** SVM Lineal → métricas, matriz de confusión, PCA, predicción manual.  
**No Supervisado:** GMM → clusters + PCA.  
**Exportación:** descarga JSON + modelos .pkl + gráficos.
""")

# ============================================
# MODO SUPERVISADO
# ============================================
if modo == "Supervisado":
    st.header("🔵 Modo Supervisado — SVM (Kernel Lineal)")

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    # ========= Métricas =========
    y_pred = svm_model.predict(X_test)

    st.subheader("Métricas (conjunto test)")
    st.write(f"**Accuracy: {accuracy_score(y_test, y_pred):.4f}**")
    st.write(f"**Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}**")
    st.write(f"**Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}**")
    st.write(f"**F1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}**")

    # ========= PCA para gráfico =========
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
    ax1.set_title("📊 PCA — SVM (Etiquetas reales)")
    st.pyplot(fig1)

    # ========= Matriz de confusión =========
    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots()
    ax2.imshow(cm, cmap="Blues")
    ax2.set_title("Matriz de confusión")
    ax2.set_xlabel("Predicción")
    ax2.set_ylabel("Real")
    st.pyplot(fig2)

    # ========= Predicción manual =========
    st.subheader("Probar una predicción manualmente")
    c1, c2, c3, c4 = st.columns(4)

    sl = c1.slider("Sepal length", 4.3, 7.9, 5.0)
    sw = c2.slider("Sepal width", 2.0, 4.4, 3.0)
    pl = c3.slider("Petal length", 1.0, 6.9, 4.0)
    pw = c4.slider("Petal width", 0.1, 2.5, 1.2)

    entrada = np.array([[sl, sw, pl, pw]])
    prediccion = svm_model.predict(entrada)[0]
    probs = svm_model.predict_proba(entrada)

    st.write(f"**Predicción:** {prediccion} - {iris.target_names[prediccion]}")
    st.write("**Probabilidades:**")
    st.json({i: float(probs[0][i]) for i in range(3)})

# ============================================
# MODO NO SUPERVISADO
# ============================================
elif modo == "No Supervisado":
    st.header("🔴 Modo No Supervisado — Gaussian Mixture Model")

    clusters = gmm_model.predict(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig3, ax3 = plt.subplots()
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="rainbow")
    ax3.set_title("📊 PCA — GMM (Clusters no supervisados)")
    st.pyplot(fig3)

# ============================================
# MODO EXPORTACIÓN
# ============================================
elif modo == "Exportación":
    st.header("📁 Zona de Exportación")

    st.write("Aquí puedes descargar modelos, dataset y gráficos.")

    # Guardar modelos
    pickle.dump(svm_model, open("svm_model.pkl", "wb"))
    pickle.dump(gmm_model, open("gmm_model.pkl", "wb"))

    # Export dataset
    json_data = df.to_json()

    # ===== Descargas =====
    st.download_button("📥 Descargar SVM (.pkl)", data=open("svm_model.pkl", "rb"), file_name="svm_model.pkl")
    st.download_button("📥 Descargar GMM (.pkl)", data=open("gmm_model.pkl", "rb"), file_name="gmm_model.pkl")
    st.download_button("📥 Descargar dataset JSON", data=json_data, file_name="dataset.json")
    st.download_button("📥 Descargar dataset CSV", data=df.to_csv(index=False), file_name="dataset.csv")

    
st.markdown("Made with Streamlit")
