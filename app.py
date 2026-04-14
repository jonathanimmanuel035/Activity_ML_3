# ==============================================================
# Aplikasi Streamlit: Prediksi Kanker Payudara (21 Fitur Terbaik)
# ==============================================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Judul Aplikasi
# -------------------------------------------------------------

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="рџ§¬", layout="wide")

st.title("Prediksi Diagnosis Kanker Payudara")
st.markdown("""
Aplikasi ini menggunakan **model Random Forest** hasil *GridSearchCV*
yang telah disimpan menggunakan **21 fitur terbaik** hasil seleksi otomatis.
Model ini memprediksi apakah sampel **jinak (0)** atau **ganas (1)**
berdasarkan nilai fitur statistik dari citra jaringan.
""")

# -------------------------------------------------------------
# Muat Model
# -------------------------------------------------------------
with open("best_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------------------
# 21 Fitur Terbaik yang Digunakan
# -------------------------------------------------------------
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'radius_se', 'perimeter_se', 'area_se',
    'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst'
]

# -------------------------------------------------------------
# Input Data Pengguna
# -------------------------------------------------------------
st.subheader("Masukkan Nilai untuk 21 Fitur")

cols = st.columns(3)
input_data = []

for i, feat in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"{feat}", min_value=0.0, format="%.4f")
        input_data.append(val)

# -------------------------------------------------------------
# Prediksi
# -------------------------------------------------------------
if st.button("Prediksi Diagnosis"):
    input_array = np.array(input_data).reshape(1,-1)

    # Prediksi kelas dan probabilitas
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    st.write("---")

    # Tampilkan hasil utama
    if prediction == 1:
        st.error(f"**Hasil Prediksi: Ganas (1)**")
    else:
        st.success(f"**Hasil Prediksi: Jinak (0)**")

    # Tampilkan probabilitas dengan bar chart
    st.markdown("### Probabilitas Prediksi")
    df_proba = pd.DataFrame({
        "Kelas": ["Jinak (0)", "Ganas (1)"],
        "Probabilitas": proba
    })

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(df_proba["Kelas"], df_proba["Probabilitas"], color=["green", "red"], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Distribusi Probabilitas Model")
    for i, v in enumerate(df_proba["Probabilitas"]):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10, fontweight="bold")
    st.pyplot(fig)

    st.write("---")

# -------------------------------------------------------------
# Info Tambahan
# -------------------------------------------------------------
with st.expander("Tentang Model"):
    st.markdown("""
        - **Model:** Random Forest (hasil *GridSearchCV*, 21 fitur terpilih)  
        - **Jumlah fitur input:** 21 fitur numerik hasil *feature selection*  
        - **Metrik evaluasi:** F1-score dengan 5-Fold Stratified Cross Validation  
        - **Label:**  
            - `0` Jinak (*Benign*)  
            - `1` Ganas (*Malignant*)  
        """)