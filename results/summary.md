# Ringkasan evaluasi

Dataset: 16 sampel PRESENT, 48 sampel ABSENT.

## Metode: global

| | Prediksi PRESENT | Prediksi ABSENT |
|---|---|---|
| **Sebenarnya PRESENT** | 15 (TP) | 1 (FN) |
| **Sebenarnya ABSENT** | 0 (FP) | 48 (TN) |

Akurasi = 98.4%  |  Recall = 93.8%  |  Specificity = 100.0%

Kesalahan:
- `p5_dekan.png` (seharusnya PRESENT) -> SIGNATURE ABSENT

## Metode: otsu

| | Prediksi PRESENT | Prediksi ABSENT |
|---|---|---|
| **Sebenarnya PRESENT** | 16 (TP) | 0 (FN) |
| **Sebenarnya ABSENT** | 0 (FP) | 48 (TN) |

Akurasi = 100.0%  |  Recall = 100.0%  |  Specificity = 100.0%

## Metode: adaptive

| | Prediksi PRESENT | Prediksi ABSENT |
|---|---|---|
| **Sebenarnya PRESENT** | 16 (TP) | 0 (FN) |
| **Sebenarnya ABSENT** | 0 (FP) | 48 (TN) |

Akurasi = 100.0%  |  Recall = 100.0%  |  Specificity = 100.0%
