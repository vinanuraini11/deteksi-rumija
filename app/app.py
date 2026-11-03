"""
Aplikasi Deteksi Pemanfaatan Rumija
Dinas PU Bina Marga Provinsi Jawa Timur

Jalankan dari folder utama project:
    streamlit run app/app.py
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st
from ultralytics import YOLO

import pengolahan as pc

ROOT = Path(__file__).resolve().parent.parent
KANDIDAT_MODEL = [                                   # dipakai yang pertama ditemukan
    ROOT / "models" / "final_960" / "weights" / "best.pt",
    ROOT / "models" / "raw" / "weights" / "best.pt",
    Path(__file__).resolve().parent / "best.pt",     # untuk versi deploy
]
DB_PATH = ROOT / "hasil_deteksi.db"                  # database hasil deteksi (tahap Load)

st.set_page_config(page_title="Deteksi Pemanfaatan Rumija", page_icon="🛣️", layout="wide")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource
def muat_model(path):
    return YOLO(str(path))


lokasi_model = next((p for p in KANDIDAT_MODEL if p.exists()), None)
if lokasi_model is None:
    st.error("File model (best.pt) tidak ditemukan. Pastikan pelatihan sudah dijalankan.")
    st.stop()
model = muat_model(lokasi_model)


def simpan_ke_db(tabel, baris):
    """Tahap Load: menambahkan hasil deteksi ke database SQLite."""
    if not baris:
        return
    with sqlite3.connect(DB_PATH) as conn:
        pd.DataFrame(baris).to_sql(tabel, conn, if_exists="append", index=False)


def baca_db(tabel):
    """Membaca seluruh isi tabel dari database."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql(f"SELECT * FROM {tabel}", conn)
        except Exception:          # tabel belum ada
            return pd.DataFrame()
ukuran_default = 960 if "960" in str(lokasi_model) else 640

if "hasil" not in st.session_state:
    st.session_state.hasil = []     # ringkasan per foto
    st.session_state.objek = []     # detail per objek terdeteksi
    st.session_state.gambar = {}    # citra hasil deteksi per nama file


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛣️ Deteksi Rumija")
halaman = st.sidebar.radio("Menu", ["Deteksi", "Peta", "Riwayat", "Tentang Model"])
st.sidebar.divider()
conf = st.sidebar.slider("Ambang keyakinan (confidence)", 0.05, 0.90, 0.25, 0.05,
                         help="Objek dengan skor keyakinan di bawah nilai ini tidak ditampilkan.")
imgsz = st.sidebar.select_slider("Ukuran citra masukan model", [640, 960, 1280], value=ukuran_default)
st.sidebar.caption(f"Model: `{lokasi_model.parent.parent.name}`")
if not pc.OCR_TERSEDIA:
    st.sidebar.warning("Tesseract OCR tidak ditemukan, sehingga koordinat dari overlay foto tidak dapat dibaca.")
if st.sidebar.button("Hapus semua hasil"):
    st.session_state.hasil, st.session_state.objek, st.session_state.gambar = [], [], {}
    st.rerun()


# ---------------------------------------------------------------------------
# Halaman Deteksi
# ---------------------------------------------------------------------------
def proses_foto(nama, data_bytes):
    # Extract: membaca foto dan mengambil lokasi (EXIF / OCR teks overlay)
    img, exif = pc.baca_upload(data_bytes)
    lokasi = pc.cari_lokasi(img, exif, nama)

    # Transform: mengukur kualitas foto, mendeteksi objek, lalu menyusun hasilnya menjadi tabel
    kualitas = pc.ukur_kualitas(img)
    r = model.predict(img, conf=conf, imgsz=imgsz, iou=0.5, device="cpu", verbose=False)[0]
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nama_kelas = r.names
    jumlah = {n: 0 for n in nama_kelas.values()}
    objek = []
    for cls, skor, kotak in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        kelas = nama_kelas[int(cls)]
        jumlah[kelas] += 1
        objek.append({
            "waktu": waktu, "file": nama, "sta": lokasi["sta"], "kelas": kelas, "confidence": round(skor, 3),
            "x1": round(kotak[0]), "y1": round(kotak[1]), "x2": round(kotak[2]), "y2": round(kotak[3]),
            "lat": lokasi["lat"], "lon": lokasi["lon"],
        })
    ringkasan = {
        "waktu": waktu, "file": nama, "sta": lokasi["sta"],
        "jumlah_reklame": jumlah.get("Reklame", 0),
        "jumlah_tiang_utilitas": jumlah.get("Tiang_Utilitas", 0),
        "lat": lokasi["lat"], "lon": lokasi["lon"], "sumber_lokasi": lokasi["sumber_lokasi"],
        **kualitas,
    }

    st.session_state.objek.extend(objek)
    st.session_state.hasil.append(ringkasan)
    st.session_state.gambar[nama] = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

    # Load: menyimpan hasil ke database
    simpan_ke_db("ringkasan", [ringkasan])
    simpan_ke_db("objek", objek)


if halaman == "Deteksi":
    st.title("Deteksi Pemanfaatan Rumija")
    st.write("Unggah foto survei jalan untuk mendeteksi **reklame** dan **tiang utilitas** secara otomatis. "
             "Koordinat lokasi dibaca dari teks *overlay* pada foto atau dari data GPS kamera.")

    unggahan = st.file_uploader("Foto survei (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    sudah = {h["file"] for h in st.session_state.hasil}
    baru = [f for f in (unggahan or []) if f.name not in sudah]

    if baru and st.button(f"Proses {len(baru)} foto", type="primary"):
        bar = st.progress(0.0, text="Memproses foto...")
        for i, f in enumerate(baru, 1):
            proses_foto(f.name, f.getvalue())
            bar.progress(i / len(baru), text=f"Memproses {f.name} ({i}/{len(baru)})")
        bar.empty()
        st.success(f"{len(baru)} foto selesai diproses.")

    if st.session_state.hasil:
        df = pd.DataFrame(st.session_state.hasil)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Foto diproses", len(df))
        k2.metric("Reklame terdeteksi", int(df["jumlah_reklame"].sum()))
        k3.metric("Tiang utilitas terdeteksi", int(df["jumlah_tiang_utilitas"].sum()))
        k4.metric("Foto berlokasi", int(df["lat"].notna().sum()))

        st.subheader("Ringkasan per foto")
        st.dataframe(df, width="stretch", hide_index=True)

        bermasalah = df[df["peringatan_kualitas"] != "-"]
        if len(bermasalah):
            st.warning(f"{len(bermasalah)} foto tergolong gelap atau blur. Hasil deteksi pada foto tersebut "
                       "sebaiknya diperiksa ulang secara visual.")

        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Unduh ringkasan (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "ringkasan_deteksi_rumija.csv", "text/csv")
        if st.session_state.objek:
            c2.download_button("⬇️ Unduh detail objek (CSV)",
                               pd.DataFrame(st.session_state.objek).to_csv(index=False).encode("utf-8"),
                               "detail_objek_rumija.csv", "text/csv")

        st.subheader("Hasil deteksi")
        pilih = st.selectbox("Pilih foto", df["file"].tolist())
        baris = df[df["file"] == pilih].iloc[0]
        st.image(st.session_state.gambar[pilih], width="stretch",
                 caption=f"STA {baris['sta']} | Reklame: {baris['jumlah_reklame']} | "
                         f"Tiang utilitas: {baris['jumlah_tiang_utilitas']}")

    st.info("Hasil deteksi merupakan **kandidat** untuk diverifikasi petugas. Sistem tidak menentukan status izin "
            "maupun memastikan objek berada di dalam batas Rumija.", icon="ℹ️")


# ---------------------------------------------------------------------------
# Halaman Peta
# ---------------------------------------------------------------------------
elif halaman == "Peta":
    st.title("Peta Lokasi Temuan")
    df = pd.DataFrame(st.session_state.hasil)
    if df.empty or df["lat"].notna().sum() == 0:
        st.info("Belum ada foto yang memiliki koordinat. Proses foto terlebih dahulu di menu **Deteksi**.")
    else:
        peta = df.dropna(subset=["lat", "lon"]).copy()
        def warna_titik(r):
            if r["jumlah_reklame"] > 0:
                return [214, 39, 40]        # merah
            if r["jumlah_tiang_utilitas"] > 0:
                return [31, 119, 180]       # biru
            return [127, 127, 127]          # abu-abu
        peta["warna"] = peta.apply(warna_titik, axis=1)
        peta["radius"] = 15 + 5 * (peta["jumlah_reklame"] + peta["jumlah_tiang_utilitas"])

        lapisan = pdk.Layer(
            "ScatterplotLayer", data=peta, get_position="[lon, lat]",
            get_fill_color="warna", get_radius="radius",
            radius_min_pixels=7, radius_max_pixels=30,
            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2,
            pickable=True,
        )
        tampilan = pdk.ViewState(latitude=peta["lat"].mean(), longitude=peta["lon"].mean(),
                                 zoom=16 if len(peta) == 1 else 13)
        st.pydeck_chart(pdk.Deck(
            layers=[lapisan], initial_view_state=tampilan,
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            tooltip={"text": "{file}\nSTA {sta}\nReklame: {jumlah_reklame}\nTiang: {jumlah_tiang_utilitas}"},
        ))
        st.markdown("🔴 Ada reklame &nbsp;&nbsp; 🔵 Hanya tiang utilitas &nbsp;&nbsp; ⚪ Tidak ada objek "
                    "&nbsp;&nbsp;|&nbsp;&nbsp; Ukuran titik menunjukkan banyaknya objek.")
        st.dataframe(peta[["file", "sta", "jumlah_reklame", "jumlah_tiang_utilitas", "lat", "lon", "sumber_lokasi"]],
                     width="stretch", hide_index=True)
        tanpa = len(df) - len(peta)
        if tanpa:
            st.caption(f"{tanpa} foto tidak ditampilkan karena koordinatnya tidak tersedia.")


# ---------------------------------------------------------------------------
# Halaman Riwayat (isi database)
# ---------------------------------------------------------------------------
elif halaman == "Riwayat":
    st.title("Riwayat Deteksi")
    st.write("Seluruh hasil deteksi yang telah tersimpan di database.")
    ringkas = baca_db("ringkasan")
    if ringkas.empty:
        st.info("Database masih kosong. Proses foto terlebih dahulu di menu **Deteksi**.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total foto tersimpan", len(ringkas))
        k2.metric("Total reklame", int(ringkas["jumlah_reklame"].sum()))
        k3.metric("Total tiang utilitas", int(ringkas["jumlah_tiang_utilitas"].sum()))
        st.subheader("Ringkasan per foto")
        st.dataframe(ringkas, width="stretch", hide_index=True)
        st.download_button("⬇️ Unduh riwayat (CSV)", ringkas.to_csv(index=False).encode("utf-8"),
                           "riwayat_deteksi_rumija.csv", "text/csv")
        objek_db = baca_db("objek")
        if not objek_db.empty:
            with st.expander("Detail objek terdeteksi"):
                st.dataframe(objek_db, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Halaman Tentang Model
# ---------------------------------------------------------------------------
else:
    st.title("Tentang Model")
    st.write("Model deteksi menggunakan **YOLO11n** yang dilatih pada foto survei ruas Kasian–Balung (Jember), "
             "Pandaan–Tretes, dan Sampang–Omben dengan dua kelas objek pemanfaatan Rumija:")
    st.markdown("- **Reklame**: papan iklan dan media informasi di tepi jalan (Permen PU No. 20/PRT/M/2010).\n"
                "- **Tiang Utilitas**: tiang listrik dan tiang telekomunikasi.")
    evaluasi = ROOT / "data" / "output" / "hasil_evaluasi_uji.csv"
    if evaluasi.exists():
        st.subheader("Kinerja pada data uji (ruas Banyuwangi)")
        st.dataframe(pd.read_csv(evaluasi), width="stretch", hide_index=True)
    folder_model = lokasi_model.parent.parent
    for nama, judul in [("results.png", "Kurva pelatihan"), ("confusion_matrix_normalized.png", "Confusion matrix")]:
        if (folder_model / nama).exists():
            st.subheader(judul)
            st.image(str(folder_model / nama), width="stretch")
    st.caption("Keterbatasan: objek yang sangat kecil, jauh, atau tertutup sebagian dapat tidak terdeteksi. "
               "Koordinat merupakan posisi pengambilan foto, bukan posisi tepat objek.")