"""
sigdetect.py - Deteksi keberadaan tanda tangan dengan pengolahan citra klasik.

Pipeline:
    crop ROI -> grayscale -> blur -> thresholding -> morphology
    -> filter komponen kecil -> ekstraksi fitur -> aturan keputusan

Pemakaian CLI:
    python sigdetect.py gambar.png --method otsu
    python sigdetect.py gambar.png --method global --threshold 128 --debug out/
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------

# ROI relatif (x0, y0, x1, y1) terhadap halaman ijazah yang sudah TEGAK
# (landscape). Dipakai supaya tidak bergantung pada resolusi scan.
ROIS = {
    "rektor": (0.143, 0.711, 0.405, 0.825),   # tanda tangan Rektor (kiri bawah)
    "dekan": (0.631, 0.657, 0.875, 0.821),    # tanda tangan Dekan (kanan bawah)
}

METHODS = ("global", "otsu", "adaptive")


@dataclass
class Params:
    blur_ksize: int = 5            # Gaussian blur (ganjil)
    global_t: int = 128            # nilai threshold untuk metode global
    adaptive_block: int = 51       # ukuran jendela adaptive (ganjil)
    adaptive_c: int = 12           # konstanta pengurang adaptive
    open_ksize: int = 3            # opening: buang bintik/noise kecil
    close_ksize: int = 7           # closing: sambung goresan yang putus
    min_component_area: int = 40   # buang komponen lebih kecil dari ini (px)

    # --- aturan keputusan (lihat decide()) ---
    min_fg_ratio: float = 0.010    # minimal 1% piksel ROI adalah foreground
    max_fg_ratio: float = 0.400    # lebih dari 40% => threshold gagal / ROI kotor
    min_span_ratio: float = 0.35   # goresan harus menjangkau >= 35% lebar ROI
    min_largest_cc_ratio: float = 0.30  # komponen terbesar >= 30% dari foreground
    min_contrast: float = 40.0     # selisih rerata abu-abu kertas vs tinta (0-255)


# --------------------------------------------------------------------------
# Langkah-langkah pipeline
# --------------------------------------------------------------------------

def upright(page_bgr: np.ndarray) -> np.ndarray:
    """Pastikan halaman landscape. Hasil scan PDF ini portrait dengan isi
    terputar 90 derajat berlawanan arah jarum jam -> putar searah jarum jam."""
    h, w = page_bgr.shape[:2]
    if h > w:
        return cv2.rotate(page_bgr, cv2.ROTATE_90_CLOCKWISE)
    return page_bgr


def crop_roi(page_bgr: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = page_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    return page_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].copy()


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def threshold(gray: np.ndarray, method: str, p: Params = Params()) -> np.ndarray:
    """Hasil: biner dengan foreground (tinta) = 255, background (kertas) = 0."""
    k = p.blur_ksize | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    if method == "global":
        _, b = cv2.threshold(blur, p.global_t, 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif method == "adaptive":
        b = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            p.adaptive_block | 1, p.adaptive_c)
    else:
        raise ValueError(f"metode tidak dikenal: {method}")
    return b


def morphology(binary: np.ndarray, p: Params = Params()) -> np.ndarray:
    """Opening (hapus noise) lalu closing (sambung goresan putus)."""
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize, p.open_ksize))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
    out = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
    return out


def remove_small(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def extract_features(binary: np.ndarray, gray: np.ndarray | None = None) -> dict:
    """Karakteristik area hasil segmentasi."""
    h, w = binary.shape
    fg = int(np.count_nonzero(binary))
    feats = {
        "roi_pixels": h * w,
        "fg_pixels": fg,
        "fg_ratio": fg / (h * w),
        "n_components": 0,
        "largest_cc_area": 0,
        "largest_cc_ratio": 0.0,   # luas komponen terbesar / total foreground
        "span_ratio": 0.0,         # lebar bounding box seluruh foreground / lebar ROI
        "contrast": 0.0,           # rerata gray background - rerata gray foreground
    }
    if fg == 0:
        return feats
    if gray is not None and fg < h * w:
        feats["contrast"] = float(gray[binary == 0].mean() - gray[binary > 0].mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    feats["n_components"] = int(n - 1)
    feats["largest_cc_area"] = int(areas.max())
    feats["largest_cc_ratio"] = float(areas.max() / fg)
    xs = np.where(binary.any(axis=0))[0]
    feats["span_ratio"] = float((xs.max() - xs.min() + 1) / w)
    return feats


def decide(f: dict, p: Params = Params()) -> str:
    """Aturan sederhana:

    PRESENT jika
      (1) cukup banyak foreground          : min_fg_ratio <= fg_ratio <= max_fg_ratio
      (2) goresan menjangkau area lebar    : span_ratio >= min_span_ratio
      (3) ada goresan panjang tersambung   : largest_cc_ratio >= min_largest_cc_ratio
      (4) foreground benar-benar lebih gelap dari background : contrast >= min_contrast
    Selain itu ABSENT.

    (3) penting untuk membedakan tanda tangan (sedikit komponen besar yang
    tersambung) dari teks cetak (banyak komponen kecil-kecil: huruf).
    (4) mencegah Otsu "mengarang" foreground dari noise kertas kosong.
    """
    ok = (
        p.min_fg_ratio <= f["fg_ratio"] <= p.max_fg_ratio
        and f["span_ratio"] >= p.min_span_ratio
        and f["largest_cc_ratio"] >= p.min_largest_cc_ratio
        and f["contrast"] >= p.min_contrast
    )
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"


# --------------------------------------------------------------------------
# API tingkat atas
# --------------------------------------------------------------------------

def analyze(roi_bgr: np.ndarray, method: str = "otsu", p: Params = Params()) -> dict:
    gray = to_gray(roi_bgr)
    raw = threshold(gray, method, p)
    clean = remove_small(morphology(raw, p), p.min_component_area)
    feats = extract_features(clean, gray)
    return {
        "method": method,
        "gray": gray,
        "binary_raw": raw,
        "binary_clean": clean,
        "features": feats,
        "decision": decide(feats, p),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Deteksi tanda tangan: PRESENT / ABSENT")
    ap.add_argument("image", help="citra ROI (sudah di-crop) atau halaman penuh")
    ap.add_argument("--page", action="store_true",
                    help="input adalah halaman penuh ijazah: putar tegak + crop ROI")
    ap.add_argument("--roi", choices=list(ROIS), default="rektor",
                    help="ROI yang di-crop jika --page dipakai")
    ap.add_argument("--method", choices=METHODS, default="otsu")
    ap.add_argument("--threshold", type=int, default=Params.global_t,
                    help="nilai T untuk metode global")
    ap.add_argument("--debug", metavar="DIR", help="simpan citra antara ke folder ini")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"tidak bisa membaca {a.image}")
    if a.page:
        img = crop_roi(upright(img), ROIS[a.roi])

    p = Params(global_t=a.threshold)
    r = analyze(img, a.method, p)
    f = r["features"]
    print(f"metode          : {a.method}")
    print(f"fg_pixels       : {f['fg_pixels']} ({f['fg_ratio']*100:.2f}% dari ROI)")
    print(f"n_components    : {f['n_components']}")
    print(f"span_ratio      : {f['span_ratio']:.2f}")
    print(f"largest_cc_ratio: {f['largest_cc_ratio']:.2f}")
    print(f"contrast        : {f['contrast']:.1f}")
    print(f"KEPUTUSAN       : {r['decision']}")

    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
        cv2.imwrite(os.path.join(a.debug, "1_roi.png"), img)
        cv2.imwrite(os.path.join(a.debug, "2_gray.png"), r["gray"])
        cv2.imwrite(os.path.join(a.debug, "3_threshold.png"), r["binary_raw"])
        cv2.imwrite(os.path.join(a.debug, "4_morphology.png"), r["binary_clean"])


if __name__ == "__main__":
    main()
