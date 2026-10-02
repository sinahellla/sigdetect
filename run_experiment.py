"""
run_experiment.py - Jalankan seluruh eksperimen mini project.

Keluaran (folder results/):
  results.csv              fitur + keputusan setiap sampel x metode
  summary.md               confusion matrix, akurasi, daftar kesalahan
  fig1_pipeline.png        tahapan: crop -> gray -> threshold -> morphology
  fig2_threshold_compare.png  global vs Otsu vs adaptive (halaman gelap & terang)
  fig3_morphology.png      sebelum/sesudah opening & closing
  fig4_threshold_sweep.png analisis threshold terlalu rendah / terlalu tinggi
  fig5_sweep_curves.png    kurva fg_ratio, TPR, TNR terhadap nilai threshold

Pemakaian:
    python run_experiment.py                 # dataset sudah ada di data/samples
    python run_experiment.py --pdf lampiran.pdf   # bangun dataset dulu
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigdetect import METHODS, Params, analyze, morphology, remove_small, threshold, to_gray

SAMPLES = "data/samples"
OUT = "results"


def load_dataset():
    items = []
    for label in ("present", "absent"):
        for f in sorted(glob.glob(f"{SAMPLES}/{label}/*.png")):
            items.append((label, os.path.basename(f), cv2.imread(f)))
    return items


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get(items, name):
    return next(im for _, n, im in items if n == name)


# --------------------------------------------------------------------------
# 1. Evaluasi seluruh sampel
# --------------------------------------------------------------------------

def evaluate(items, p):
    rows, summary = [], []
    for m in METHODS:
        tp = fp = tn = fn = 0
        errors = []
        for label, name, img in items:
            r = analyze(img, m, p)
            f = r["features"]
            pred_present = r["decision"].endswith("PRESENT")
            truth_present = label == "present"
            tp += pred_present and truth_present
            fn += (not pred_present) and truth_present
            fp += pred_present and (not truth_present)
            tn += (not pred_present) and (not truth_present)
            if pred_present != truth_present:
                errors.append((name, label, r["decision"]))
            rows.append({
                "sample": name, "truth": label.upper(), "method": m,
                "fg_pixels": f["fg_pixels"], "fg_ratio": round(f["fg_ratio"], 4),
                "n_components": f["n_components"],
                "span_ratio": round(f["span_ratio"], 3),
                "largest_cc_ratio": round(f["largest_cc_ratio"], 3),
                "contrast": round(f["contrast"], 1),
                "decision": r["decision"],
            })
        summary.append((m, tp, fn, fp, tn, errors))

    with open(f"{OUT}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(1 for l, _, _ in items if l == "present")
    n_neg = len(items) - n_pos
    lines = [f"# Ringkasan evaluasi\n",
             f"Dataset: {n_pos} sampel PRESENT, {n_neg} sampel ABSENT.\n"]
    for m, tp, fn, fp, tn, errors in summary:
        acc = (tp + tn) / len(items)
        lines += [
            f"## Metode: {m}\n",
            "| | Prediksi PRESENT | Prediksi ABSENT |",
            "|---|---|---|",
            f"| **Sebenarnya PRESENT** | {tp} (TP) | {fn} (FN) |",
            f"| **Sebenarnya ABSENT** | {fp} (FP) | {tn} (TN) |",
            f"\nAkurasi = {acc*100:.1f}%  |  Recall = {tp/max(tp+fn,1)*100:.1f}%"
            f"  |  Specificity = {tn/max(tn+fp,1)*100:.1f}%\n",
        ]
        if errors:
            lines.append("Kesalahan:")
            lines += [f"- `{n}` (seharusnya {l.upper()}) -> {d}" for n, l, d in errors]
            lines.append("")
    open(f"{OUT}/summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))


# --------------------------------------------------------------------------
# 2. Figur
# --------------------------------------------------------------------------

def fig_pipeline(items, p):
    img = get(items, "p1_dekan.png")
    r = analyze(img, "otsu", p)
    raw = r["binary_raw"]
    panels = [("1. Crop ROI (BGR->RGB)", rgb(img), None),
              ("2. Grayscale", r["gray"], "gray"),
              ("3. Otsu threshold", raw, "gray"),
              ("4. Opening + closing\n+ buang komponen kecil", r["binary_clean"], "gray")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.6), dpi=150)
    for a, (t, im, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm, vmin=0 if cm else None, vmax=255 if cm else None)
        a.set_title(t, fontsize=11)
        a.axis("off")
    f = r["features"]
    fig.suptitle(f"Pipeline pada p1_dekan.png - foreground {f['fg_pixels']} px "
                 f"({f['fg_ratio']*100:.2f}%), {f['n_components']} komponen -> {r['decision']}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_threshold_compare(items, p):
    names = ["p1_rektor.png", "p5_dekan.png"]
    fig, ax = plt.subplots(len(names), 4, figsize=(20, 3.4 * len(names)), dpi=150)
    for i, n in enumerate(names):
        img = get(items, n)
        ax[i, 0].imshow(rgb(img))
        ax[i, 0].set_title(f"{n}\n(rerata gray ROI = {to_gray(img).mean():.0f})", fontsize=10)
        for j, m in enumerate(METHODS, start=1):
            r = analyze(img, m, p)
            f = r["features"]
            ax[i, j].imshow(r["binary_raw"], cmap="gray", vmin=0, vmax=255)
            lbl = {"global": f"Global T={p.global_t}", "otsu": "Otsu", "adaptive": "Adaptive Gaussian"}[m]
            ax[i, j].set_title(f"{lbl}\nfg={f['fg_pixels']} px ({f['fg_ratio']*100:.1f}%) -> "
                               f"{r['decision'][10:]}", fontsize=10)
        for a in ax[i]:
            a.axis("off")
    fig.suptitle("Perbandingan metode thresholding (sebelum morphology)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_threshold_compare.png", bbox_inches="tight")
    plt.close(fig)


def fig_morphology(items, p):
    img = get(items, "p1_dekan.png")
    gray = to_gray(img)
    raw = threshold(gray, "adaptive", p)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize, p.open_ksize))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ko)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)
    final = remove_small(closed, p.min_component_area)
    steps = [("Hasil threshold (adaptive)", raw), ("Opening", opened),
             ("Opening + Closing", closed), ("+ buang komponen kecil", final)]
    fig, ax = plt.subplots(2, 4, figsize=(20, 7), dpi=150)
    h, w = raw.shape
    zoom = (slice(int(h * 0.55), int(h * 0.95)), slice(int(w * 0.05), int(w * 0.40)))
    for j, (t, im) in enumerate(steps):
        n = cv2.connectedComponents(im, connectivity=8)[0] - 1
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(f"{t}\nfg={np.count_nonzero(im)} px, {n} komponen", fontsize=10)
        ax[1, j].imshow(im[zoom], cmap="gray", vmin=0, vmax=255)
        ax[1, j].set_title("zoom", fontsize=9)
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle("Efek operasi morfologi", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_morphology.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweep(items, p):
    # (a) citra hasil threshold pada beberapa nilai T
    name = "p5_dekan.png"
    img = get(items, name)
    gray = to_gray(img)
    Ts = [30, 100, 150, 200, 235]
    fig, ax = plt.subplots(1, len(Ts) + 1, figsize=(22, 3.2), dpi=150)
    ax[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(f"{name} (gray)", fontsize=10)
    for a, T in zip(ax[1:], Ts):
        q = Params(global_t=T)
        b = threshold(gray, "global", q)
        f_ratio = np.count_nonzero(b) / b.size
        a.imshow(b, cmap="gray", vmin=0, vmax=255)
        tag = "TERLALU RENDAH" if T <= 50 else ("TERLALU TINGGI" if T >= 200 else "")
        a.set_title(f"T = {T} {tag}\nfg = {f_ratio*100:.1f}%", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.suptitle("Pengaruh nilai global threshold pada satu ROI tanda tangan", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # (b) kurva terhadap seluruh dataset
    grid = list(range(10, 251, 10))
    pos = [im for l, _, im in items if l == "present"]
    neg = [im for l, _, im in items if l == "absent"]
    fg_pos, tpr, tnr = [], [], []
    for T in grid:
        q = Params(global_t=T)
        rp = [analyze(im, "global", q) for im in pos]
        rn = [analyze(im, "global", q) for im in neg]
        fg_pos.append(np.mean([r["features"]["fg_ratio"] for r in rp]) * 100)
        tpr.append(np.mean([r["decision"].endswith("PRESENT") for r in rp]) * 100)
        tnr.append(np.mean([r["decision"].endswith("ABSENT") for r in rn]) * 100)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    ax[0].plot(grid, fg_pos, "o-")
    ax[0].axhline(p.max_fg_ratio * 100, color="r", ls="--", label="batas atas fg_ratio pada aturan")
    ax[0].axhline(p.min_fg_ratio * 100, color="g", ls="--", label="batas bawah fg_ratio pada aturan")
    ax[0].set_xlabel("global threshold T")
    ax[0].set_ylabel("rerata fg_ratio sampel PRESENT (%)")
    ax[0].set_title("Jumlah foreground vs T")
    ax[0].legend(fontsize=8)
    ax[1].plot(grid, tpr, "o-", label="TPR (PRESENT terdeteksi)")
    ax[1].plot(grid, tnr, "s-", label="TNR (ABSENT terdeteksi)")
    ax[1].set_xlabel("global threshold T")
    ax[1].set_ylabel("%")
    ax[1].set_title("Kinerja aturan vs T (seluruh dataset)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_sweep_curves.png", bbox_inches="tight")
    plt.close(fig)
    return grid, tpr, tnr


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="bangun dataset dari PDF ini sebelum eksperimen")
    a = ap.parse_args()

    if a.pdf:
        subprocess.check_call([sys.executable, "build_dataset.py", "--pdf", a.pdf, "--out", SAMPLES])
    if not glob.glob(f"{SAMPLES}/present/*.png"):
        raise SystemExit("Dataset kosong. Jalankan dengan --pdf lampiran.pdf terlebih dahulu.")

    os.makedirs(OUT, exist_ok=True)
    p = Params()
    items = load_dataset()
    evaluate(items, p)
    fig_pipeline(items, p)
    fig_threshold_compare(items, p)
    fig_morphology(items, p)
    grid, tpr, tnr = fig_sweep(items, p)
    print("\nSweep global threshold (T, TPR%, TNR%):")
    for T, a_, b_ in zip(grid, tpr, tnr):
        print(f"  T={T:3d}  TPR={a_:5.1f}  TNR={b_:5.1f}")
    print(f"\nSemua keluaran ada di folder '{OUT}/'.")


if __name__ == "__main__":
    main()
