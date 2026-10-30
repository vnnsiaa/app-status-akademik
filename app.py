
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Status Akademik",
    page_icon="🎓",
    layout="centered"
)

# Lokasi file model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_status_akademik.pkl"
INFO_PATH = BASE_DIR / "model_info.json"

# Pemetaan label sesuai LabelEncoder pada Modul 1A
LABEL_STATUS = {
    0: "Aman",
    1: "Berisiko",
    2: "Waspada"
}

# Memuat model dan informasi model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    return model, info


st.title("🎓 Prediksi Status Akademik Mahasiswa")
st.write(
    "Masukkan data akademik mahasiswa untuk memperoleh "
    "prediksi status akademik menggunakan model DNN."
)

# Periksa ketersediaan file
if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.error(
        "File model tidak ditemukan. Pastikan "
        "model_status_akademik.pkl dan model_info.json "
        "berada dalam folder yang sama dengan app.py."
    )
    st.stop()

try:
    model, info = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

# Form input data
st.subheader("Masukkan Data Mahasiswa")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        ipk = st.number_input(
            "IPK",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.01
        )

        kehadiran = st.number_input(
            "Kehadiran (%)",
            min_value=45,
            max_value=100,
            value=85
        )

        jam_belajar = st.number_input(
            "Jam Belajar per Minggu",
            min_value=0,
            max_value=30,
            value=8
        )

    with col2:
        tugas = st.number_input(
            "Tugas Terkumpul (%)",
            min_value=35,
            max_value=100,
            value=80
        )

        nilai_uts = st.number_input(
            "Nilai UTS",
            min_value=24,
            max_value=100,
            value=75
        )

        aktif_organisasi = st.selectbox(
            "Aktif Organisasi",
            options=["Ya", "Tidak"]
        )

    submitted = st.form_submit_button(
        "Prediksi Status Akademik",
        use_container_width=True
    )

# Proses prediksi
if submitted:
    data_mahasiswa = pd.DataFrame([{
        "ipk": ipk,
        "kehadiran": kehadiran,
        "jam_belajar": jam_belajar,
        "tugas_terkumpul": tugas,
        "nilai_uts": nilai_uts,
        "aktif_organisasi": aktif_organisasi
    }])

    try:
        prediksi = int(model.predict(data_mahasiswa)[0])
        nama_status = LABEL_STATUS.get(
            prediksi, "Status tidak dikenal"
        )

        st.subheader("Hasil Prediksi")

        if nama_status == "Aman":
            st.success(f"Status akademik: {nama_status}")
        elif nama_status == "Waspada":
            st.warning(f"Status akademik: {nama_status}")
        else:
            st.error(f"Status akademik: {nama_status}")

        st.caption(
            "Hasil ini merupakan prediksi model berdasarkan "
            "data yang dimasukkan, bukan keputusan akademik resmi."
        )

        if hasattr(model, "predict_proba"):
            probabilitas = model.predict_proba(data_mahasiswa)[0]

            st.subheader("Probabilitas Prediksi")

            hasil_probabilitas = pd.DataFrame({
                "Status": [
                    LABEL_STATUS.get(int(k), str(k))
                    for k in model.classes_
                ],
                "Probabilitas (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilitas
                ]
            })

            st.dataframe(
                hasil_probabilitas,
                hide_index=True,
                use_container_width=True
            )

        with st.expander("Lihat data yang dimasukkan"):
            st.dataframe(data_mahasiswa, hide_index=True)

    except Exception as e:
        st.error(f"Prediksi gagal dilakukan: {e}")

# Informasi model
with st.expander("Informasi Model DNN"):
    st.write("Nama model:", info.get("nama_model", "DNN"))
    st.write("Accuracy test:", info.get("metrik", {}).get("accuracy"))
    st.write("F1 macro test:", info.get("metrik", {}).get("f1_macro"))

# ==========================================
# PREDIKSI BANYAK MAHASISWA MELALUI FILE CSV
# ==========================================

st.divider()
st.subheader("📁 Prediksi dari File CSV")

st.write(
    "Unggah file CSV yang berisi data akademik mahasiswa. "
    "Sistem akan memprediksi status akademik setiap mahasiswa."
)

FITUR = info["fitur_numerik"] + info["fitur_kategori"]

uploaded_file = st.file_uploader(
    "Pilih file CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)

        st.write("### Data yang Diunggah")
        st.dataframe(df_csv, use_container_width=True)

        # Memeriksa kolom wajib
        kolom_kurang = [
            kolom for kolom in FITUR
            if kolom not in df_csv.columns
        ]

        if kolom_kurang:
            st.error(
                "Kolom berikut belum tersedia: "
                + ", ".join(kolom_kurang)
            )

        elif df_csv.empty:
            st.warning("File CSV tidak memiliki data.")

        else:
            if st.button("Prediksi Semua Data"):
                # Prediksi menggunakan model DNN
                hasil_prediksi = model.predict(df_csv[FITUR])

                # Mengubah label angka menjadi nama status
                df_csv["prediksi_status"] = [
                    LABEL_STATUS.get(int(label), "Tidak dikenal")
                    for label in hasil_prediksi
                ]

                st.success("Prediksi seluruh data berhasil!")

                st.write("### Hasil Prediksi")
                st.dataframe(
                    df_csv,
                    use_container_width=True
                )

                st.write("### Jumlah Mahasiswa per Status")
                st.write(
                    df_csv["prediksi_status"].value_counts()
                )

                # Download hasil sebagai CSV
                hasil_csv = df_csv.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Hasil Prediksi CSV",
                    data=hasil_csv,
                    file_name="hasil_prediksi_status_akademik.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")