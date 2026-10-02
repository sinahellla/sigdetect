# Signature Presence Detector (Mini Project Pengolahan Citra)

Mendeteksi apakah sebuah area tanda tangan pada scan ijazah **ada** atau **tidak ada**
memakai pengolahan citra klasik (tanpa machine learning):

`crop ROI → grayscale → thresholding → morphology → fitur area → aturan → SIGNATURE PRESENT / ABSENT`

Input: `data/lampiran.pdf` (scan ijazah, 8 halaman). File ini **tidak** ikut di repo karena berisi data pribadi; letakkan sendiri di folder `data/`.
Tanda tangan yang di-crop: **Rektor** (kiri bawah) dan **Dekan** (kanan bawah).

## 1. Instalasi

Butuh Python 3.9+ dan **poppler** (dipakai `pdf2image` untuk membaca PDF).

```bash
# poppler
#   Ubuntu/Debian : sudo apt install poppler-utils
#   macOS         : brew install poppler
#   Windows       : unduh poppler, tambahkan folder bin ke PATH

git clone <URL-REPO-ANDA>
cd sigdetect
mkdir -p data && cp /path/ke/lampiran.pdf data/lampiran.pdf
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Cara menjalankan

### a) Seluruh eksperimen (disarankan)

```bash
python run_experiment.py --pdf data/lampiran.pdf
```

Perintah ini: membangun dataset uji dari PDF → mengevaluasi 3 metode threshold →
membuat semua figur dan tabel di folder `results/`.
Run berikutnya tidak perlu `--pdf` (dataset sudah ada di `data/samples/`).

### b) Satu citra saja

```bash
# ROI yang sudah di-crop
python sigdetect.py data/samples/present/p1_rektor.png --method otsu

# Halaman penuh (diputar tegak + crop otomatis)
python sigdetect.py halaman.png --page --roi dekan --method adaptive --debug out/

# Global threshold dengan T manual
python sigdetect.py data/samples/present/p1_rektor.png --method global --threshold 140
```

Contoh keluaran:

```
metode          : otsu
fg_pixels       : 9383 (3.93% dari ROI)
n_components    : 2
span_ratio      : 0.88
largest_cc_ratio: 0.72
contrast        : 71.3
KEPUTUSAN       : SIGNATURE PRESENT
```

`--debug DIR` menyimpan citra tiap tahap (`1_roi`, `2_gray`, `3_threshold`, `4_morphology`).

## 3. Struktur repo

```
sigdetect.py         pipeline + aturan keputusan + CLI
build_dataset.py     PDF -> sampel PRESENT / ABSENT
run_experiment.py    evaluasi + figur + analisis threshold
data/lampiran.pdf    input (taruh sendiri, tidak di-commit)
data/samples/        dataset hasil crop (dibuat otomatis, tidak di-commit)
results/             results.csv, summary.md, fig1..fig5
```

## 4. Cara kerja

1. **Crop ROI.** Halaman PDF diputar 90° searah jarum jam (isi scan terputar), lalu ROI
   diambil dengan koordinat *relatif* (persen lebar/tinggi) agar tidak tergantung resolusi.
2. **Grayscale.** `cv2.cvtColor(..., BGR2GRAY)`. Warna kertas berbeda-beda antarhalaman
   (hijau, kuning, abu), jadi hanya intensitas yang dipakai.
3. **Thresholding** (setelah Gaussian blur 5×5). Tiga metode dibandingkan:
   - **Global** `T = 128` (tetap),
   - **Otsu** (T dipilih otomatis dari histogram),
   - **Adaptive Gaussian** (T berbeda untuk tiap piksel, jendela 51×51, C = 12).
   Hasil: tinta = foreground (putih), kertas = background (hitam).
4. **Morphology.** *Opening* (elips 3×3) membuang bintik kecil; *closing* (elips 7×7)
   menyambung goresan yang putus. Lalu komponen < 40 piksel dibuang.
5. **Fitur area** pada hasil segmentasi:

   | Fitur | Arti |
   |---|---|
   | `fg_pixels`, `fg_ratio` | jumlah dan persentase piksel foreground |
   | `n_components` | jumlah komponen terhubung |
   | `largest_cc_ratio` | luas komponen terbesar / total foreground |
   | `span_ratio` | lebar bounding box foreground / lebar ROI |
   | `contrast` | rerata gray background − rerata gray foreground |

6. **Aturan keputusan** (parameter di `Params`):

   ```
   SIGNATURE PRESENT jika SEMUA benar:
     0.010 <= fg_ratio <= 0.400     # cukup tinta, tapi bukan "seluruh ROI hitam"
     span_ratio        >= 0.35      # goresan menjangkau area lebar
     largest_cc_ratio  >= 0.30      # ada goresan panjang yang tersambung
     contrast          >= 40        # foreground memang jauh lebih gelap dari kertas
   selain itu: SIGNATURE ABSENT
   ```

   `largest_cc_ratio` membedakan tanda tangan (sedikit komponen besar) dari **teks cetak**
   (puluhan komponen kecil berupa huruf). `contrast` mencegah Otsu "mengarang" foreground
   dari tekstur kertas kosong.

## 5. Dataset uji

Dibuat otomatis dari 8 halaman PDF (`build_dataset.py`):

| Kelas | Isi | Jumlah |
|---|---|---|
| PRESENT | ROI Rektor + ROI Dekan × 8 halaman | 16 |
| ABSENT | kertas kosong (2 area) × 8 halaman | 16 |
| ABSENT | teks cetak saja (2 area, negatif "sulit") × 8 halaman | 16 |
| ABSENT | ROI tanda tangan yang tintanya dihapus (inpainting) × 8 halaman | 16 |

## 6. Hasil

| Metode | TP | FN | FP | TN | Akurasi |
|---|---|---|---|---|---|
| Global (T=128) | 15 | 1 | 0 | 48 | 98.4% |
| Otsu | 16 | 0 | 0 | 48 | 100% |
| Adaptive | 16 | 0 | 0 | 48 | 100% |

Detail per sampel: `results/results.csv` dan `results/summary.md`.

**Keterbatasan (penting).** 8 halaman PDF adalah **satu ijazah yang sama** dengan warna/kecerahan
scan berbeda, dan sampel ABSENT dibuat dari dokumen yang sama (kertas kosong, teks cetak,
inpainting). Jadi akurasi 100% **tidak** membuktikan sistem akan bekerja pada ijazah lain.
Untuk pengujian yang lebih kuat, taruh citra dokumen lain di `data/samples/present/` dan
`data/samples/absent/` lalu jalankan `run_experiment.py` tanpa `--pdf`.
Selain itu, ROI memakai koordinat tetap sehingga bergantung pada tata letak ijazah ini.

## 7. Analisis

### Mengapa thresholding diperlukan sebelum analisis tanda tangan?

Citra grayscale hanya berisi nilai 0–255; komputer belum "tahu" mana piksel tinta dan mana
kertas. Fitur seperti jumlah piksel foreground, jumlah komponen, atau lebar goresan hanya
bisa dihitung pada citra **biner**. Thresholding memisahkan objek (tanda tangan) dari latar,
mengubah pertanyaan "apakah ada tanda tangan?" menjadi pertanyaan terukur: "apakah ada
cukup piksel foreground yang membentuk goresan panjang?". Thresholding juga menghilangkan
pengaruh warna dan gradasi kertas. Tanpa langkah ini, nilai piksel kertas hijau, kuning,
dan abu tidak bisa dibandingkan, dan morphology serta connected components tidak dapat
dipakai.

### Apa masalahnya jika threshold terlalu tinggi atau terlalu rendah?

Tinta lebih gelap dari kertas, sehingga piksel dengan gray ≤ T dijadikan foreground.
Eksperimen pada `results/fig4_threshold_sweep.png` dan `fig5_sweep_curves.png`:

| | Terlalu rendah (T kecil, mis. 30–60) | Terlalu tinggi (T besar, mis. ≥ 200) |
|---|---|---|
| Efek | Hanya tinta tergelap yang lolos; goresan tipis atau pudar hilang atau terputus-putus | Kertas, bayangan, dan teks cetak ikut menjadi foreground; goresan menebal dan menyatu |
| Hasil | `fg_ratio` ≈ 0, tanda tangan nyata dianggap ABSENT (**false negative**) | Foreground membanjiri ROI, melewati batas `fg_ratio`, atau menyatu dengan teks cetak (**false positive / false negative**) |
| Pada data ini | TPR 0% pada T ≤ 50 | TPR turun ke 75% pada T 190–230 dan 0% pada T = 250 |

Pada data ini rentang T yang bekerja baik sempit (T = 140–150 mencapai TPR 100%), dan
T = 128 sudah gagal pada `p5_dekan.png` karena halaman itu sangat terang (rerata gray 235)
dengan tinta tipis. Itu alasan metode **Otsu** (T dari histogram) dan **adaptive**
(T lokal) lebih stabil daripada global T tetap: keduanya menyesuaikan diri dengan
kecerahan halaman. Kekurangan Otsu: pada ROI kertas kosong ia tetap memaksa membelah
histogram sehingga menghasilkan foreground dari noise (hingga puluhan persen). Itu sebabnya
aturan keputusan perlu fitur `contrast`, `largest_cc_ratio`, dan `span_ratio`, tidak
cukup hanya jumlah piksel.

### Perbandingan metode

- **Global:** paling sederhana dan cepat, tetapi sensitif terhadap kecerahan halaman (gagal di p5_dekan).
- **Otsu:** otomatis dan akurat pada ROI bimodal (ada tinta dan kertas), tetapi berbahaya pada ROI kosong (membuat foreground palsu).
- **Adaptive:** tahan terhadap iluminasi tidak merata dan tidak membuat foreground dari kertas kosong; sedikit lebih bergantung pada ukuran jendela dan C.
