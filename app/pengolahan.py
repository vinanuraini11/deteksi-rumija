"""
Fungsi pengolahan citra untuk aplikasi Deteksi Pemanfaatan Rumija.
Fungsi-fungsi ini sama dengan yang digunakan pada notebook 01 dan 03.
"""
import platform
import re
import shutil
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------
# 1. Menyiapkan Tesseract OCR
# ---------------------------------------------------------------
try:
    import pytesseract
    _lokasi_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if shutil.which("tesseract") is None and platform.system() == "Windows" and _lokasi_windows.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_lokasi_windows)
    OCR_TERSEDIA = shutil.which("tesseract") is not None or _lokasi_windows.exists()
except ImportError:
    pytesseract = None
    OCR_TERSEDIA = False


# ---------------------------------------------------------------
# 2. Membaca foto yang diunggah
# ---------------------------------------------------------------
def baca_upload(data_bytes):
    """Ubah file unggahan menjadi citra BGR (format OpenCV)."""
    pil = Image.open(BytesIO(data_bytes))
    exif = pil.getexif()
    pil = ImageOps.exif_transpose(pil).convert("RGB")   # koreksi orientasi foto
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr, exif


def perkecil(img, max_sisi=1280):
    """Resize citra agar sisi terpanjangnya maksimal 1280 piksel."""
    h, w = img.shape[:2]
    skala = max_sisi / max(h, w)
    if skala < 1:
        img = cv2.resize(img, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------
# 3. Mengukur kualitas citra (notebook 01)
# ---------------------------------------------------------------
AMBANG_GELAP = 90     # rata-rata intensitas grayscale
AMBANG_BLUR = 300     # varians Laplacian


def ukur_kualitas(img):
    g = cv2.cvtColor(perkecil(img), cv2.COLOR_BGR2GRAY)
    kecerahan = float(g.mean())
    ketajaman = float(cv2.Laplacian(g, cv2.CV_64F).var())
    catatan = []
    if kecerahan < AMBANG_GELAP:
        catatan.append("gelap")
    if ketajaman < AMBANG_BLUR:
        catatan.append("blur")
    return {"kecerahan": round(kecerahan, 1),
            "ketajaman": round(ketajaman, 1),
            "peringatan_kualitas": ", ".join(catatan) if catatan else "-"}


# ---------------------------------------------------------------
# 4. Membaca koordinat dari teks overlay (notebook 03)
# ---------------------------------------------------------------
_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


def segmentasi_teks(img, ambang_v=215):
    """Potong area overlay, pisahkan teks putih dengan HSV + morfologi."""
    h, w = img.shape[:2]
    roi = img[int(h * 0.62):, int(w * 0.50):]               # pojok kanan bawah
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, ambang_v), (180, 45, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    return 255 - cv2.resize(mask, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)


def ambil_koordinat(teks):
    """Cari bujur (11x,xxxx) dulu, lalu lintang dari angka di depannya."""
    for baris in teks.splitlines():
        b = baris.replace(",", ".")
        m = re.search(r"(?<!\d)(11[2-4])[.\s]?(\d{4,})", b)
        if not m:
            continue
        lon = float(f"{m[1]}.{m[2]}")
        depan = re.sub(r"\D", "", b[:m.start()])
        i = next((k for k, c in enumerate(depan) if c in "78"), None)
        if i is None:
            continue
        desimal = depan[i + 1:i + 1 + len(m[2])]
        if len(desimal) < 4:
            continue
        lat = -float(f"{depan[i]}.{desimal}")
        if -9 <= lat <= -7 and 111 <= lon <= 115:            # wilayah Jawa Timur
            return lat, lon
    return None, None


def baca_overlay(img, daftar_ambang=(215, 200, 185, 230)):
    """Baca koordinat dan STA dari overlay. Hasil: (lat, lon, sta)."""
    if not OCR_TERSEDIA:
        return None, None, None
    sta = None
    for v in daftar_ambang:
        teks = pytesseract.image_to_string(segmentasi_teks(img, v), config="--psm 6")
        s = re.search(r"(?:STA\s*)?(\d{1,2})\s*\+\s*(\d{3})", teks)
        sta = sta or (f"{int(s[1])}+{s[2]}" if s else None)
        lat, lon = ambil_koordinat(teks)
        if lat is not None:
            return lat, lon, sta
    return None, None, sta


def gps_dari_exif(exif):
    """Cadangan: ambil koordinat dari GPS kamera kalau ada."""
    try:
        gps = exif.get_ifd(0x8825)
        def ke_desimal(nilai, arah):
            d, m, s = [float(x) for x in nilai]
            hasil = d + m / 60 + s / 3600
            return -hasil if arah in ("S", "W") else hasil
        lat = ke_desimal(gps[2], gps[1])
        lon = ke_desimal(gps[4], gps[3])
        if np.isnan(lat) or np.isnan(lon) or (lat == 0 and lon == 0):
            return None, None
        return lat, lon
    except Exception:
        return None, None


def sta_dari_nama(nama):
    """Ambil STA dari nama file, misal '3+950.jpg'."""
    m = re.match(r"\s*(\d+)\s*\+\s*(\d{3})", nama)
    return f"{int(m[1])}+{m[2]}" if m else None


def cari_lokasi(img, exif, nama_file):
    """Urutan sumber lokasi: OCR overlay -> GPS EXIF -> tidak ada."""
    lat, lon, sta = baca_overlay(img)
    sumber = "OCR overlay" if lat is not None else None
    if lat is None:
        lat, lon = gps_dari_exif(exif)
        sumber = "GPS EXIF" if lat is not None else "tidak tersedia"
    sta = sta_dari_nama(nama_file) or sta
    return {"lat": lat, "lon": lon, "sta": sta or "-", "sumber_lokasi": sumber}