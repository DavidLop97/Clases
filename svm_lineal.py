import pandas as pd
import pickle
import io
import json
import base64

# Zona de exportación
if modo == "Zona de Exportación":
    st.header("📦 Zona de Exportación")
    st.markdown("Primero ejecuta Supervisado y No Supervisado para habilitar descargas.")

    # Dataset
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
