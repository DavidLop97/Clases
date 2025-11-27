import pandas as pd
import pickle
import io
import json
import base64

st.subheader("📦 Zona de Exportación")

st.write("Aquí puedes descargar datos, métricas o tu modelo entrenado.")

# ---- 1. Exportar dataset ----
df_csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📄 Descargar Dataset (CSV)",
    df_csv,
    file_name="dataset_iris.csv",
    mime="text/csv"
)

# ---- 2. Exportar métricas ----
metrics_dict = {
    "accuracy": accuracy,
    "precision_macro": precision,
    "recall_macro": recall,
    "f1_macro": f1
}

metrics_json = json.dumps(metrics_dict, indent=4)
st.download_button(
    "📊 Descargar Métricas (JSON)",
    metrics_json,
    file_name="metricas_svm.json",
    mime="application/json"
)

# ---- 3. Exportar modelo SVM ----
model_bytes = pickle.dumps(model)
st.download_button(
    "🤖 Descargar Modelo SVM (PKL)",
    model_bytes,
    file_name="modelo_svm.pkl",
    mime="application/octet-stream"
)

# ---- 4. Exportar gráfico Supervisado ----
buffer = io.BytesIO()
fig_svm.savefig(buffer, format="png")
buffer.seek(0)

st.download_button(
    "📉 Descargar Gráfico SVM",
    buffer,
    file_name="grafico_svm.png",
    mime="image/png"
)

# ---- 5. Exportar gráfico GMM ----
buffer2 = io.BytesIO()
fig_gmm.savefig(buffer2, format="png")
buffer2.seek(0)

st.download_button(
    "📈 Descargar Gráfico GMM",
    buffer2,
    file_name="grafico_gmm.png",
    mime="image/png"
)
