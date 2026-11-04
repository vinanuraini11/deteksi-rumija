"""
Aplikasi Deteksi Pemanfaatan Rumija
Dinas PU Bina Marga Provinsi Jawa Timur

Jalankan dari folder utama project:
    streamlit run app/app.py
"""
import base64
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import pengolahan as pc

ROOT = Path(__file__).resolve().parent.parent
KANDIDAT_MODEL = [                                   # dipakai yang pertama ditemukan
    ROOT / "models" / "final_960" / "weights" / "best.pt",
    ROOT / "models" / "raw" / "weights" / "best.pt",
    Path(__file__).resolve().parent / "best.pt",     # untuk versi deploy
]
APP_DIR = Path(__file__).resolve().parent
LOGO = APP_DIR / "logo_jatim.png"                    # logo Pemprov Jawa Timur
BANNER = APP_DIR / "banner.jpg"                       # foto survei untuk latar banner
DB_PATH = ROOT / "hasil_deteksi.db"                  # database hasil deteksi (tahap Load)

st.set_page_config(page_title="Deteksi Pemanfaatan Rumija", page_icon=str(LOGO) if LOGO.exists() else "🛣️", layout="wide")

# ---------------------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 3.5rem; max-width: 1200px;}
.hero {
    background: linear-gradient(120deg, #123A5E 0%, #1F5F8B 60%, #2E86AB 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 1.4rem; color: #FFFFFF;
}
.hero .ikon {font-size: 2.2rem; line-height: 1;}
.hero h1 {color: #FFFFFF; font-size: 2rem; margin: .4rem 0 .3rem 0; padding: 0;}
.hero p {color: #DCE9F5; font-size: 1.02rem; margin: 0; max-width: 820px;}
.hero .lencana {margin-top: .9rem;}
.hero .lencana span {
    display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
    border-radius: 999px; padding: 3px 12px; margin: 0 6px 6px 0; font-size: .82rem; color: #FFFFFF;
}
.langkah {
    background: rgba(31,95,139,.14); border: 1px solid rgba(31,95,139,.45); border-radius: 14px;
    padding: 18px 20px; height: 100%;
}
.langkah .no {
    display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
    border-radius: 50%; background: #F2B33D; color: #0F2D4A; font-weight: 700; margin-bottom: .5rem;
}
.langkah b {display: block; color: #2E86C1; margin-bottom: .2rem;}
.langkah span {color: #7F94A8; font-size: .92rem;}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16466F 0%, #1F5F8B 100%); border: none; border-left: 5px solid #F2B33D;
    border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 6px rgba(15,45,74,.25);
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color: #FFFFFF !important; font-weight: 700;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color: #BFD3E6 !important;}
/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0F2D4A 0%, #16466F 100%);}
[data-testid="stSidebar"] * {color: #E6EEF6;}
[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.15); margin: 1rem 0;}
.merek {display: flex; align-items: center; gap: 12px; padding: 4px 0 6px 0;}
.merek .logo {
    width: 46px; height: 46px; border-radius: 12px; background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.22); display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.merek .logo img {width: 34px; height: 34px; object-fit: contain;}
.merek b {display: block; font-size: 1.2rem; color: #FFFFFF; line-height: 1.2;}
.merek span {font-size: .78rem; color: #A9C2DA !important;}
.label-bagian {
    font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #8FB0CF !important;
    margin: .6rem 0 .3rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadio"] > div,
[data-testid="stSidebar"] .stButton, [data-testid="stSidebar"] [data-testid="stElementContainer"] {width: 100% !important;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: 4px; width: 100%;}
[data-testid="stSidebar"] [role="radiogroup"] > div {width: 100%;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex; width: 100%; padding: 9px 12px; border-radius: 10px; margin: 0; transition: background .15s;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255,255,255,.08);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,.16); box-shadow: inset 3px 0 0 #F2B33D;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]),
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {display: none;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size: .98rem; font-weight: 500;}
.kartu-model {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 12px;
    padding: 12px 14px; margin: .4rem 0 .8rem 0; font-size: .85rem;
}
.kartu-model b {color: #FFFFFF;}
.kartu-model .titik {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4ADE80;
    margin-right: 6px; box-shadow: 0 0 0 3px rgba(74,222,128,.25);
}
.kartu-model .baris {display: flex; justify-content: space-between; margin-top: 6px; color: #C7D8E8;}
[data-testid="stSidebar"] .stButton button {
    width: 100%; background: transparent; border: 1px solid rgba(255,255,255,.35); border-radius: 10px;
}
[data-testid="stSidebar"] .stButton button:hover {border-color: #F2B33D; color: #F2B33D;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def gambar_b64(path):
    """Mengubah file gambar menjadi teks base64 agar bisa ditampilkan di HTML."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".png":
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    img = Image.open(path).convert("RGB")
    img.thumbnail((1600, 1600))                      # diperkecil agar halaman tetap ringan
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def kepala(judul, deskripsi, lencana=()):
    """Menampilkan banner judul di bagian atas setiap halaman."""
    isi_lencana = "".join(f"<span>{x}</span>" for x in lencana)
    foto = gambar_b64(BANNER)
    latar = (f'style="background-image: linear-gradient(110deg, rgba(15,45,74,.95) 0%, rgba(22,70,111,.85) 45%, '
             f'rgba(22,70,111,.15) 100%), url({foto}); background-size: cover; background-position: center;"'
             if foto else "")
    st.markdown(f"""
    <div class="hero" {latar}>
        <h1>{judul}</h1>
        <p>{deskripsi}</p>
        {f'<div class="lencana">{isi_lencana}</div>' if lencana else ''}
    </div>""", unsafe_allow_html=True)


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


ukuran_default = 640 if "raw" in str(lokasi_model) else 960
nama_model = "final_960" if lokasi_model.parent.name == "app" else lokasi_model.parent.parent.name

if "hasil" not in st.session_state:
    st.session_state.hasil = []     # ringkasan per foto
    st.session_state.objek = []     # detail per objek terdeteksi
    st.session_state.gambar = {}    # citra hasil deteksi per nama file


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
IKON_MENU = {"Deteksi": ":material/document_scanner:", "Peta": ":material/map:",
             "Riwayat": ":material/database:", "Tentang Model": ":material/info:"}

logo = gambar_b64(LOGO)
isi_logo = f'<img src="{logo}" alt="Logo">' if logo else "🛣️"
st.sidebar.markdown(f"""
<div class="merek">
    <div class="logo">{isi_logo}</div>
    <div><b>Deteksi Rumija</b><span>Dinas PU Bina Marga Provinsi Jawa Timur</span></div>
</div>""", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.markdown('<div class="label-bagian">MENU</div>', unsafe_allow_html=True)
halaman = st.sidebar.radio("Menu", list(IKON_MENU), format_func=lambda x: f"{IKON_MENU[x]}  {x}",
                           label_visibility="collapsed")
st.sidebar.divider()

st.sidebar.markdown('<div class="label-bagian">PENGATURAN DETEKSI</div>', unsafe_allow_html=True)
conf = st.sidebar.slider("Ambang keyakinan (confidence)", 0.05, 0.90, 0.25, 0.05,
                         help="Objek dengan skor keyakinan di bawah nilai ini tidak ditampilkan.")
imgsz = st.sidebar.select_slider("Ukuran citra masukan model", [640, 960, 1280], value=ukuran_default)
st.sidebar.markdown(f"""
<div class="kartu-model">
    <span class="titik"></span><b>Model aktif</b>
    <div class="baris"><span>Nama</span><span>{nama_model}</span></div>
    <div class="baris"><span>Arsitektur</span><span>YOLO11n</span></div>
    <div class="baris"><span>Kelas</span><span>Reklame, Tiang</span></div>
</div>""", unsafe_allow_html=True)
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
    kepala("Deteksi Pemanfaatan Rumija",
           "Unggah foto survei jalan untuk mendeteksi reklame dan tiang utilitas secara otomatis. "
           "Koordinat lokasi dibaca dari teks pada foto atau dari data GPS kamera.",
           ["Model YOLO11n", "Pembacaan koordinat (OCR)", "Peta lokasi temuan", "Tersimpan di database"])

    if not st.session_state.hasil:
        l1, l2, l3 = st.columns(3)
        for kolom, no, judul, ket in [
            (l1, 1, "Unggah foto", "Pilih satu atau beberapa foto survei jalan (JPG/PNG)."),
            (l2, 2, "Proses deteksi", "Sistem menandai reklame dan tiang utilitas pada setiap foto."),
            (l3, 3, "Lihat dan unduh", "Cek lokasi temuan di menu Peta, lalu unduh hasilnya."),
        ]:
            kolom.markdown(f'<div class="langkah"><div class="no">{no}</div><b>{judul}</b>'
                           f'<span>{ket}</span></div>', unsafe_allow_html=True)
        st.write("")

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
        c1.download_button("Unduh ringkasan (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "ringkasan_deteksi_rumija.csv", "text/csv", icon=":material/download:")
        if st.session_state.objek:
            c2.download_button("Unduh detail objek (CSV)",
                               pd.DataFrame(st.session_state.objek).to_csv(index=False).encode("utf-8"),
                               "detail_objek_rumija.csv", "text/csv", icon=":material/download:")

        st.subheader("Hasil deteksi")
        pilih = st.selectbox("Pilih foto", df["file"].tolist())
        baris = df[df["file"] == pilih].iloc[0]
        st.image(st.session_state.gambar[pilih], width="stretch",
                 caption=f"STA {baris['sta']} | Reklame: {baris['jumlah_reklame']} | "
                         f"Tiang utilitas: {baris['jumlah_tiang_utilitas']}")

    st.info("Hasil deteksi merupakan **kandidat** untuk diverifikasi petugas. Sistem tidak menentukan status izin "
            "maupun memastikan objek berada di dalam batas Rumija.", icon=":material/info:")


# ---------------------------------------------------------------------------
# Halaman Peta
# ---------------------------------------------------------------------------
elif halaman == "Peta":
    kepala("Peta Lokasi Temuan",
           "Sebaran lokasi foto yang memuat reklame dan tiang utilitas berdasarkan koordinat hasil pembacaan.",
           ["Merah: ada reklame", "Biru: hanya tiang utilitas", "Ukuran titik: jumlah objek"])
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
    kepala("Riwayat Deteksi", "Seluruh hasil deteksi yang telah tersimpan di database.",
           ["Database SQLite", "Data tetap tersimpan", "Dapat diunduh (CSV)"])
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
        st.download_button("Unduh riwayat (CSV)", ringkas.to_csv(index=False).encode("utf-8"),
                           "riwayat_deteksi_rumija.csv", "text/csv", icon=":material/download:")
        objek_db = baca_db("objek")
        if not objek_db.empty:
            with st.expander("Detail objek terdeteksi"):
                st.dataframe(objek_db, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Halaman Tentang Model
# ---------------------------------------------------------------------------
else:
    kepala("Tentang Model", "Informasi model deteksi, data pelatihan, dan kinerjanya pada data uji.",
           ["YOLO11n", "2 kelas objek", "Data latih: 3 ruas", "Data uji: ruas Banyuwangi"])
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