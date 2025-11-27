import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

st.title("Tarea: Supervisado (SVM lineal) + No Supervisado (GMM)")
st.subheader("Vista previa del dataset")
st.dataframe(df)

st.header("🔵 Modo Supervisado — SVM (Kernel Lineal)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

st.markdown("### Métricas (conjunto test)")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.write("**Precision (macro):**", precision_score(y_test, y_pred, average="macro"))
st.write("**Recall (macro):**", recall_score(y_test, y_pred, average="macro"))
st.write("**F1-score (macro):**", f1_score(y_test, y_pred, average="macro"))

st.subheader("📊 Gráfico Supervisado (SVM)")
fig1, ax1 = plt.subplots()
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
ax1.set_xlabel("Sepal Length")
ax1.set_ylabel("Sepal Width")
ax1.set_title("Clasificación SVM - Proyección 2D")
st.pyplot(fig1)

st.subheader("Probar una predicción manualmente")
sl = st.slider("Sepal length", 4.3, 7.9, 5.1)
sw = st.slider("Sepal width", 2.0, 4.4, 3.5)
pl = st.slider("Petal length", 1.0, 6.9, 1.4)
pw = st.slider("Petal width", 0.1, 2.5, 0.2)

pred = svm.predict([[sl, sw, pl, pw]])[0]
st.write("Predicción:", pred, "-", iris.target_names[pred])

st.header("🔴 Modo No Supervisado — Gaussian Mixture Model")

gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X)

st.subheader("📊 Gráfico No Supervisado (GMM)")
fig2, ax2 = plt.subplots()
ax2.scatter(X[:, 0], X[:, 1], c=clusters, cmap="rainbow", edgecolor="k")
ax2.set_xlabel("Sepal Length")
ax2.set_ylabel("Sepal Width")
ax2.set_title("Clusters GMM - Proyección 2D")
st.pyplot(fig2)