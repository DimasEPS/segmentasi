## Segmentasi (Roberts, Prewitt, Sobel, Frei-Chen)

Script `main.py` menjalankan segmentasi tepi berbasis discontinuity untuk empat kondisi citra: warna asli, grayscale bersih, grayscale dengan derau salt & pepper, dan grayscale dengan derau Gaussian. Hasilnya mencakup peta tepi, nilai MSE (noisy vs grayscale bersih), ringkasan rata-rata, grafik batang, serta salinan hasil terbaik per operator.

### Struktur input/output

- Input: `../Pengcit_Noise-And-Filter/images/original/*.jpg|png` (dipakai apa adanya).
- Output: `output/`
  - `sources/` contoh citra yang dipakai.
  - `edges/<variant>/<operator>.png` hasil segmentasi.
  - `metrics.csv` detail MSE per citra/variant/operator.
  - `summary.csv` rata-rata MSE per operator (saltpepper vs gaussian).
  - `mse_chart.png` grafik batang MSE (diperbaharui dengan grid, legend, dan info lebih jelas).
  - `best_noise/` salinan hasil dengan MSE terendah per operator.
  - `resegmented_best/` re-segmentasi citra terbaik Prewitt dengan semua operator.
  - `resegmented_metrics.csv` detail MSE hasil re-segmentasi.
  - `resegmented_comparison_chart.png` grafik perbandingan hasil re-segmentasi.
  - `comparison_table.csv` tabel perbandingan MSE dalam format yang mudah dibaca.

### Setup Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan

```bash
# Pastikan venv sudah aktif
source venv/bin/activate

# Run program
python main.py
```

Output tersimpan di folder `output` (dibuat otomatis). Nilai MSE dihitung relatif terhadap hasil segmentasi grayscale bersih, sehingga makin kecil makin mirip citra tanpa derau.

### Dependencies

- `opencv-python==4.12.0.88`
- `numpy==2.2.6`
