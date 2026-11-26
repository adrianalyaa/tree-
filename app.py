"""
Prediksi Harga Mobil Bekas - Streamlit
Tiga cara prediksi sekaligus : Decision Tree, Random Forest, Gradient Boosting.

Menjalankan lokal :
    pip install -r requirements.txt
    streamlit run app.py

Dataset (quikr_car.csv) dicari dengan urutan :
    1. file quikr_car.csv di folder yang sama dengan app.py (atau folder data/)
    2. unduh otomatis dari Kaggle lewat kagglehub
    3. upload manual lewat sidebar
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="wide")

FITUR_KATEGORI = ["company", "model", "fuel_type"]
FITUR_ANGKA = ["year", "kms"]
TARGET = "Price"
MODEL_LAIN = "(model lain, ketik sendiri)"
NAMA_MODEL = ["Decision Tree", "Random Forest", "Gradient Boosting"]
WARNA = {"Decision Tree": "#1f77b4", "Random Forest": "#2ca02c", "Gradient Boosting": "#ff7f0e"}

LOKASI_CSV = ["quikr_car.csv", os.path.join("data", "quikr_car.csv")]


# ----------------------------------------------------------------------------
# Memuat & membersihkan data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def baca_csv(sumber):
    return pd.read_csv(sumber, dtype=str)


@st.cache_data(show_spinner="Mengunduh dataset dari Kaggle...")
def unduh_kaggle():
    import kagglehub

    path = kagglehub.dataset_download("muhammadmusharraf444/used-car-price-prediction-dataset")
    return pd.read_csv(os.path.join(path, "quikr_car.csv"), dtype=str)


def muat_data_mentah():
    for lokasi in LOKASI_CSV:
        if os.path.exists(lokasi):
            return baca_csv(lokasi)
    try:
        return unduh_kaggle()
    except Exception:
        pass

    st.warning(
        "Dataset `quikr_car.csv` tidak ditemukan dan tidak bisa diunduh otomatis. "
        "Upload file CSV-nya di bawah ini."
    )
    unggahan = st.file_uploader("Upload quikr_car.csv", type="csv")
    if unggahan is None:
        st.stop()
    return baca_csv(unggahan)


@st.cache_data(show_spinner=False)
def bersihkan_data(mentah):
    """Pembersihan sama persis dengan notebook (bagian persiapan data)."""
    data = mentah.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"].str.replace(",", "", regex=False), errors="coerce")
    data["kms"] = pd.to_numeric(
        data["kms_driven"].str.replace(",", "", regex=False).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    data = data.dropna(subset=["name", "company", "year", "Price", "kms", "fuel_type"]).drop_duplicates()
    data["model"] = data["name"].str.split().str[:3].str.join(" ")

    q1, q3 = data["Price"].quantile([0.25, 0.75])
    batas_atas = q3 + 1.5 * (q3 - q1)
    data = data[(data["Price"] <= batas_atas) & (data["kms"] <= 200_000)].reset_index(drop=True)
    return data, float(batas_atas)


# ----------------------------------------------------------------------------
# Training 3 model
# ----------------------------------------------------------------------------
def buat_pipeline(estimator):
    pre = ColumnTransformer(
        [("kategori", OneHotEncoder(handle_unknown="ignore"), FITUR_KATEGORI)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


@st.cache_resource(show_spinner=False)
def latih_semua_model(data):
    X = data[FITUR_KATEGORI + FITUR_ANGKA]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kandidat = {
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [6, 8, 10, 12, None], "min_samples_leaf": [2, 3, 5, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 2, 3],
                "max_features": [0.3, 0.5, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]},
        ),
    }

    model, metrik, param_terbaik, pred_test, penting = {}, [], {}, {}, {}
    for nama, (estimator, grid) in kandidat.items():
        pencarian = GridSearchCV(
            buat_pipeline(estimator),
            {f"model__{k}": v for k, v in grid.items()},
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        pencarian.fit(X_train, y_train)
        terbaik = pencarian.best_estimator_

        p_train, p_test = terbaik.predict(X_train), terbaik.predict(X_test)
        metrik.append(
            {
                "Model": nama,
                "R2 Training": r2_score(y_train, p_train),
                "R2 Testing": r2_score(y_test, p_test),
                "MAE Training": mean_absolute_error(y_train, p_train),
                "MAE Testing": mean_absolute_error(y_test, p_test),
                "RMSE Testing": float(np.sqrt(mean_squared_error(y_test, p_test))),
            }
        )
        model[nama] = terbaik
        param_terbaik[nama] = {k.replace("model__", ""): v for k, v in pencarian.best_params_.items()}
        pred_test[nama] = p_test

        nama_fitur = terbaik.named_steps["preprocess"].get_feature_names_out()
        nilai = pd.Series(terbaik.named_steps["model"].feature_importances_, index=nama_fitur)
        penting[nama] = nilai.groupby(_kolom_asal).sum()

    metrik = pd.DataFrame(metrik).set_index("Model")
    metrik["Selisih R2 (Train - Test)"] = metrik["R2 Training"] - metrik["R2 Testing"]

    return {
        "model": model,
        "metrik": metrik,
        "param": param_terbaik,
        "penting": pd.DataFrame(penting).fillna(0),
        "X_train": X_train,
        "y_train": y_train,
        "y_test": y_test,
        "pred_test": pred_test,
    }


def _kolom_asal(nama):
    """Gabungkan hasil One-Hot (mis. company_Hyundai) kembali ke kolom asalnya."""
    nama = nama.replace("kategori__", "").replace("remainder__", "")
    for kolom in FITUR_KATEGORI:
        if nama.startswith(kolom + "_"):
            return kolom
    return nama


# ----------------------------------------------------------------------------
# Prediksi
# ----------------------------------------------------------------------------
def prediksi_semua(hasil, merk, nama_mobil, tahun, kms, bbm):
    model_mobil = " ".join(nama_mobil.split()[:3])
    baris = pd.DataFrame(
        [{"company": merk, "model": model_mobil, "fuel_type": bbm, "year": float(tahun), "kms": float(kms)}]
    )

    keluaran = {}
    for nama, pipeline in hasil["model"].items():
        harga = max(float(pipeline.predict(baris)[0]), 0.0)
        rendah = tinggi = None
        pre, estimator = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]

        if nama == "Random Forest":
            # Sebaran prediksi dari tiap pohon
            angka = pre.transform(baris)
            per_pohon = np.array([p.predict(angka)[0] for p in estimator.estimators_])
            rendah, tinggi = np.percentile(per_pohon, [10, 90])
            keterangan = "Rentang prediksi antar pohon (P10 - P90)"
        elif nama == "Decision Tree":
            # Harga data training yang jatuh di daun yang sama
            daun = estimator.apply(pre.transform(baris))[0]
            daun_train = estimator.apply(pre.transform(hasil["X_train"]))
            mirip = hasil["y_train"][daun_train == daun]
            if len(mirip) > 1:
                rendah, tinggi = mirip.min(), mirip.max()
            keterangan = f"Rentang mobil serupa di data training ({len(mirip)} data)"
        else:
            keterangan = "Boosting tidak punya rentang bawaan"

        keluaran[nama] = {"harga": harga, "rendah": rendah, "tinggi": tinggi, "keterangan": keterangan}
    return keluaran


def rp(nilai):
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def inr(nilai):
    return f"INR {nilai:,.0f}"


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
st.title("🚗 Prediksi Harga Mobil Bekas")
st.caption("Tiga cara prediksi sekaligus : Decision Tree, Random Forest, dan Gradient Boosting.")

mentah = muat_data_mentah()
data, batas_harga = bersihkan_data(mentah)

with st.spinner("Melatih 3 model (hanya dilakukan sekali saat pertama dibuka)..."):
    hasil = latih_semua_model(data)

with st.sidebar:
    st.header("Pengaturan")
    kurs = st.number_input(
        "Kurs 1 INR ke Rupiah",
        min_value=1.0,
        value=195.0,
        step=1.0,
        help="Harga pada dataset dalam Rupee India (INR). Konversi bersifat perkiraan, sesuaikan dengan kurs terbaru.",
    )
    st.divider()
    st.metric("Data dipakai", f"{len(data):,} baris")
    st.metric("Merk", data["company"].nunique())
    st.metric("Model mobil", data["model"].nunique())
    st.caption(
        "Hasil prediksi adalah **perkiraan awal**, bukan harga pasti. Dataset tidak memuat kondisi mobil, "
        "tipe transmisi, atau kapasitas mesin."
    )

tab_prediksi, tab_performa, tab_data = st.tabs(["🔮 Prediksi", "📊 Performa Model", "🗂️ Tentang Data"])

# ---------------------------- Tab Prediksi ----------------------------------
with tab_prediksi:
    st.subheader("Masukkan data mobil")
    kiri, kanan = st.columns(2)

    with kiri:
        daftar_merk = sorted(data["company"].unique())
        merk = st.selectbox("Merk", daftar_merk, index=daftar_merk.index("Maruti") if "Maruti" in daftar_merk else 0)

        model_merk = data.loc[data["company"] == merk, "model"].value_counts().index.tolist()
        pilihan_model = st.selectbox("Model mobil", model_merk + [MODEL_LAIN])
        if pilihan_model == MODEL_LAIN:
            nama_ketik = st.text_input("Ketik nama/model mobil", placeholder=f"mis. {model_merk[0]}")
            nama_mobil = nama_ketik.strip()
            if nama_mobil and not nama_mobil.lower().startswith(merk.lower()):
                nama_mobil = f"{merk} {nama_mobil}"
        else:
            nama_mobil = pilihan_model

    with kanan:
        tahun_min, tahun_max = int(data["year"].min()), int(data["year"].max())
        tahun = st.number_input(
            "Tahun pembuatan", min_value=tahun_min, max_value=tahun_max,
            value=min(max(2012, tahun_min), tahun_max), step=1,
        )
        kms = st.number_input("Jarak tempuh (km)", min_value=0, max_value=1_000_000, value=40_000, step=1_000)
        daftar_bbm = sorted(data["fuel_type"].unique())
        bbm = st.selectbox("Jenis bahan bakar", daftar_bbm)

    if st.button("Prediksi Harga", type="primary", width="stretch"):
        if not nama_mobil:
            st.error("Nama/model mobil belum diisi.")
        else:
            prediksi = prediksi_semua(hasil, merk, nama_mobil, tahun, kms, bbm)
            model_dikenal = " ".join(nama_mobil.split()[:3]) in set(data["model"])

            st.divider()
            st.subheader(f"{nama_mobil} | {tahun} | {kms:,} km | {bbm}")
            if not model_dikenal:
                st.warning("Model mobil ini tidak ada di dataset, prediksi hanya memakai merk, tahun, jarak tempuh, dan bahan bakar sehingga kurang akurat.")

            kolom = st.columns(3)
            for kolom_i, nama in zip(kolom, NAMA_MODEL):
                p = prediksi[nama]
                with kolom_i.container(border=True):
                    st.markdown(f"**{nama}**")
                    st.markdown(f"### {rp(p['harga'] * kurs)}")
                    st.caption(inr(p["harga"]))
                    if p["rendah"] is not None:
                        st.write(f"{inr(p['rendah'])} - {inr(p['tinggi'])}")
                    st.caption(p["keterangan"])

            semua = np.array([prediksi[n]["harga"] for n in NAMA_MODEL])
            rata, selisih = semua.mean(), semua.max() - semua.min()

            ringkas = st.columns(3)
            ringkas[0].metric("Rata-rata 3 model", rp(rata * kurs), inr(rata), delta_color="off")
            ringkas[1].metric("Terendah - tertinggi", f"{rp(semua.min() * kurs)}", f"sampai {rp(semua.max() * kurs)}", delta_color="off")
            ringkas[2].metric("Selisih antar model", f"{selisih / rata * 100:.0f}%", "dari rata-rata", delta_color="off")
            if selisih / rata > 0.3:
                st.info("Ketiga model cukup berbeda pendapat untuk mobil ini, jadi anggap hasilnya sebagai perkiraan kasar.")

            grafik = pd.DataFrame({"Model": NAMA_MODEL, "Prediksi (Rp)": [prediksi[n]["harga"] * kurs for n in NAMA_MODEL]})
            st.bar_chart(grafik, x="Model", y="Prediksi (Rp)", color="#2ca02c")

# ---------------------------- Tab Performa ----------------------------------
with tab_performa:
    st.subheader("Perbandingan performa pada data test")
    st.write("Ketiga model dilatih dan diuji pada pembagian data yang sama (80% training, 20% testing).")

    metrik = hasil["metrik"]
    st.dataframe(
        metrik.style.format(
            {
                "R2 Training": "{:.3f}",
                "R2 Testing": "{:.3f}",
                "Selisih R2 (Train - Test)": "{:.3f}",
                "MAE Training": "{:,.0f}",
                "MAE Testing": "{:,.0f}",
                "RMSE Testing": "{:,.0f}",
            }
        ),
        width="stretch",
    )

    terbaik = metrik["R2 Testing"].idxmax()
    st.success(f"R2 testing tertinggi : **{terbaik}** ({metrik.loc[terbaik, 'R2 Testing']:.3f}), MAE testing {metrik.loc[terbaik, 'MAE Testing']:,.0f} INR.")
    st.caption("Selisih R2 training dan testing yang besar (di atas 0,1 - 0,2) menandakan model cenderung menghafal data (overfitting).")

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**R2 testing** (makin tinggi makin baik)")
        st.bar_chart(metrik[["R2 Testing"]])
    with kanan:
        st.markdown("**MAE testing, INR** (makin rendah makin baik)")
        st.bar_chart(metrik[["MAE Testing"]])

    st.markdown("**Fitur yang paling berpengaruh terhadap harga**")
    st.bar_chart(hasil["penting"])

    st.markdown("**Harga aktual vs prediksi (data test)**")
    pilih = st.radio("Model", NAMA_MODEL, horizontal=True, index=1)
    sebar = pd.DataFrame({"Harga aktual": hasil["y_test"].values, "Harga prediksi": hasil["pred_test"][pilih]})
    st.scatter_chart(sebar, x="Harga aktual", y="Harga prediksi", color=WARNA[pilih])

    with st.expander("Parameter terbaik hasil GridSearchCV"):
        for nama in NAMA_MODEL:
            st.write(f"**{nama}**", hasil["param"][nama])

# ---------------------------- Tab Data --------------------------------------
with tab_data:
    st.subheader("Data setelah dibersihkan")
    st.write(
        f"Dari **{len(mentah):,}** baris data mentah, tersisa **{len(data):,}** baris setelah membuang data kosong/tidak valid, "
        f"duplikat, harga di atas {batas_harga:,.0f} INR (outlier IQR), dan jarak tempuh di atas 200.000 km."
    )
    st.dataframe(data[["name", "company", "year", "Price", "kms", "fuel_type"]].head(50), width="stretch")

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Jumlah mobil per merk**")
        st.bar_chart(data["company"].value_counts())
    with kanan:
        st.markdown("**Jumlah mobil per jenis bahan bakar**")
        st.bar_chart(data["fuel_type"].value_counts())
