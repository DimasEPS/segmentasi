# Analisis Hasil Segmentasi

## 1. Perbandingan Operator pada Noise

Berdasarkan data di `comparison_table.csv`:

| Operator  | Gaussian | Salt & Pepper |
|-----------|----------|---------------|
| PREWITT   | 213.19   | 987.50        |
| FREI-CHEN | 219.09   | 978.13        |
| SOBEL     | 235.92   | 1025.78       |
| ROBERTS   | 422.32   | 1599.55       |

### Temuan Utama:

#### Gaussian Noise:
- **Terbaik**: Prewitt (MSE: 213.19)
- **Terburuk**: Roberts (MSE: 422.32)
- Prewitt dan Frei-Chen menunjukkan performa sangat baik dan mirip (~213-219)
- Roberts paling sensitif terhadap Gaussian noise (hampir 2x lipat MSE)

#### Salt & Pepper Noise:
- **Terbaik**: Frei-Chen (MSE: 978.13)
- **Terburuk**: Roberts (MSE: 1599.55)
- Frei-Chen sedikit lebih baik dari Prewitt untuk noise impulsif
- Semua operator memiliki MSE 4-5x lebih tinggi dibanding Gaussian

## 2. Analisis Per Operator

### Prewitt
- **Kelebihan**: Konsisten baik untuk Gaussian (terbaik) dan Salt & Pepper
- **MSE Ratio**: 987.50 / 213.19 = 4.63x (degradasi sedang pada noise impulsif)
- **Rekomendasi**: Pilihan terbaik untuk Gaussian noise

### Frei-Chen
- **Kelebihan**: Terbaik untuk Salt & Pepper, runner-up untuk Gaussian
- **MSE Ratio**: 978.13 / 219.09 = 4.47x (paling robust terhadap perubahan tipe noise)
- **Rekomendasi**: Pilihan terbaik untuk Salt & Pepper, paling balanced overall

### Sobel
- **Kelebihan**: Performa tengah untuk kedua tipe noise
- **MSE Ratio**: 1025.78 / 235.92 = 4.35x
- **Rekomendasi**: Pilihan aman untuk kedua tipe noise

### Roberts
- **Kelebihan**: Kernel kecil (2x2), komputasi cepat
- **Kelemahan**: Paling sensitif terhadap noise (MSE tertinggi untuk kedua tipe)
- **MSE Ratio**: 1599.55 / 422.32 = 3.79x (paling sensitif ke noise impulsif)
- **Rekomendasi**: Hindari untuk citra noisy

## 3. Kesimpulan

1. **Gaussian noise lebih mudah ditangani** (MSE 4-5x lebih kecil) dibanding Salt & Pepper
   - Gaussian: noise aditif, terdistribusi merata
   - Salt & Pepper: noise impulsif, merusak informasi piksel secara drastis

2. **Ranking operator berdasarkan robustness**:
   1. **Frei-Chen**: Paling balanced, terbaik untuk Salt & Pepper
   2. **Prewitt**: Terbaik untuk Gaussian, konsisten
   3. **Sobel**: Performa tengah, reliable
   4. **Roberts**: Paling sensitif, hindari untuk citra noisy

3. **Ukuran kernel matters**:
   - Kernel 3x3 (Prewitt, Sobel, Frei-Chen) lebih robust
   - Kernel 2x2 (Roberts) terlalu sensitif terhadap noise

4. **Rekomendasi praktis**:
   - **Untuk Gaussian**: Gunakan Prewitt
   - **Untuk Salt & Pepper**: Gunakan Frei-Chen
   - **Tidak tahu tipe noise**: Gunakan Frei-Chen (paling balanced)
   - **Butuh kecepatan**: Pre-processing dengan filter (median/gaussian) dahulu, baru Roberts

## 4. Visualisasi

- **Chart utama**: `output/mse_chart.png` - Perbandingan semua operator pada semua noise
- **Chart re-segmentasi**: `output/resegmented_comparison_chart.png` - Fokus pada citra terbaik Prewitt
- **Tabel perbandingan**: `output/comparison_table.csv` - Format mudah dibaca untuk laporan

## 5. Data Mentah

- **Semua hasil**: `output/metrics.csv`
- **Ringkasan**: `output/summary.csv`
- **Re-segmentasi**: `output/resegmented_metrics.csv`
- **Gambar hasil terbaik**: `output/best_noise/` dan `output/resegmented_best/`
