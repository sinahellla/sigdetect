"""
build_dataset.py - Ubah PDF scan ijazah menjadi dataset uji kecil.

Setiap halaman PDF -> putar tegak -> crop:
  present/  : ROI tanda tangan Rektor & Dekan (label: ada tanda tangan)
  absent/   : (a) area kertas kosong,
              (b) area teks cetak saja (negatif "sulit"),
              (c) ROI tanda tangan yang tintanya dihapus dengan inpainting
                  (simulasi dokumen yang belum ditandatangani)

Pemakaian:
    python build_dataset.py --pdf lampiran.pdf --out data/samples
"""
import argparse
import os

import cv2
import numpy as np
from pdf2image import convert_from_path

from sigdetect import ROIS, crop_roi, upright, to_gray

# Negatif alami: area tanpa tanda tangan (koordinat relatif, halaman tegak)
BLANK_ROIS = {
    "blank_kiri_atas": (0.030, 0.050, 0.292, 0.164),
    "blank_kanan_atas": (0.685, 0.050, 0.947, 0.164),
}
TEXT_ROIS = {
    "text_nama_rektor": (0.119, 0.825, 0.381, 0.940),   # "Prof. Ari Kuncoro ..." + teks cetak
    "text_isi": (0.327, 0.236, 0.589, 0.349),           # baris teks cetak isi ijazah
}


def remove_ink(roi_bgr: np.ndarray) -> np.ndarray:
    """Hapus tinta dengan inpainting -> kertas 'bersih' dengan sisa artefak halus."""
    gray = to_gray(roi_bgr)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return cv2.inpaint(roi_bgr, mask, 5, cv2.INPAINT_TELEA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--dpi", type=int, default=72,
                    help="PDF ini sudah berukuran 2376x3360 pt, jadi 72 dpi cukup")
    a = ap.parse_args()

    for sub in ("present", "absent"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    pages = convert_from_path(a.pdf, dpi=a.dpi)
    print(f"{len(pages)} halaman dibaca dari {a.pdf}")

    n_pos = n_neg = 0
    for i, pil in enumerate(pages, start=1):
        page = upright(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))

        for name, roi in ROIS.items():
            crop = crop_roi(page, roi)
            cv2.imwrite(f"{a.out}/present/p{i}_{name}.png", crop)
            n_pos += 1
            cv2.imwrite(f"{a.out}/absent/p{i}_{name}_inpaint.png", remove_ink(crop))
            n_neg += 1

        for name, roi in {**BLANK_ROIS, **TEXT_ROIS}.items():
            cv2.imwrite(f"{a.out}/absent/p{i}_{name}.png", crop_roi(page, roi))
            n_neg += 1

    print(f"selesai: {n_pos} sampel PRESENT, {n_neg} sampel ABSENT -> {a.out}")


if __name__ == "__main__":
    main()
