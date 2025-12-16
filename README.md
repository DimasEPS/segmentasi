## Segmentasi (Roberts, Prewitt, Sobel, Frei-Chen)

Script `main.py` menjalankan segmentasi tepi berbasis discontinuity untuk empat kondisi citra: warna asli, grayscale bersih, grayscale dengan derau salt & pepper, dan grayscale dengan derau Gaussian. Hasilnya mencakup peta tepi, nilai MSE (noisy vs grayscale bersih), ringkasan rata-rata, grafik batang, serta salinan hasil terbaik per operator.

### Struktur input/output

- Input: `../Pengcit_Noise-And-Filter/images/original/*.jpg|png` (dipakai apa adanya).
- Output: `output/`
  - `sources/` contoh citra yang dipakai.
  - `edges/<variant>/<operator>.png` hasil segmentasi.
  - `metrics.csv` detail MSE per citra/variant/operator.
  - `summary.csv` rata-rata MSE per operator (saltpepper vs gaussian).
  - `mse_chart.png` grafik batang MSE (1400x800px dengan grid, legend, shadow, axis labels).
  - `best_noise/` salinan hasil dengan MSE terendah per operator.
  - `resegmented_best/` re-segmentasi **1 citra terbaik** (MSE Prewitt terendah) dengan semua operator untuk kedua noise.
  - `resegmented_metrics.csv` detail MSE hasil re-segmentasi citra terbaik.
  - `resegmented_comparison_chart.png` grafik visual perbandingan hasil re-segmentasi (1200x700px).
  - `comparison_table.csv` tabel perbandingan lengkap dengan metadata, ranking, analisis, stabilitas, dan kesimpulan.

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

### Fitur Analisis

1. **Segmentasi Multi-Operator**: 4 operator (Roberts, Prewitt, Sobel, Frei-Chen) pada 4 kondisi citra
2. **MSE Comparison**: Evaluasi ketahanan operator terhadap noise
3. **Best Image Selection**: Otomatis memilih citra dengan hasil Prewitt terbaik
4. **Re-segmentation**: Citra terbaik di-segmentasi ulang dengan semua operator untuk analisis mendalam
5. **Visual Reports**: 2 chart informatif (ukuran besar dengan grid, legend, labels)
6. **Comprehensive Table**: Tabel CSV lengkap dengan ranking, persentase, stabilitas, dan kesimpulan

### Hasil Analisis

Berdasarkan eksperimen:

- **Terbaik untuk Gaussian**: Prewitt (MSE: 213.19)
- **Terbaik untuk Salt & Pepper**: Frei-Chen (MSE: 978.13)
- **Terbaik Overall**: Frei-Chen (paling balanced)
- **Terburuk**: Roberts (98% lebih buruk dari terbaik)

### Dependencies

- `opencv-python==4.12.0.88`
- `numpy==2.2.6`
