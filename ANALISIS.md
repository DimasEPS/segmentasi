# Analisis Hasil Segmentasi

## Informasi Citra Re-Segmentasi

**Citra terbaik yang dipilih**: `landscape`  
**Variant terbaik Prewitt**: Gaussian  
**MSE Prewitt terbaik**: 213.19  
**Catatan**: Citra `landscape` memiliki hasil segmentasi Prewitt dengan MSE terendah untuk noise Gaussian. Citra ini kemudian di-segmentasi ulang dengan semua 4 operator (Roberts, Prewitt, Sobel, Frei-Chen) untuk kedua tipe noise (Gaussian & Salt&Pepper) untuk analisis perbandingan mendalam.

## 1. Perbandingan Operator pada Noise

Berdasarkan data di `comparison_table.csv` (hasil re-segmentasi citra `landscape`):

| Operator  | Gaussian | Salt & Pepper |
| --------- | -------- | ------------- |
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

### Prewitt (#1 Gaussian, #2 Overall)

- **Kelebihan**: Konsisten baik untuk Gaussian (terbaik) dan Salt & Pepper
- **MSE Ratio**: 987.50 / 213.19 = 4.63x (degradasi sedang pada noise impulsif)
- **Peningkatan vs terbaik**: Gaussian +0%, Salt&Pepper +1.0%
- **Stabilitas**: #2 (selisih 774.31)
- **Rekomendasi**: Pilihan terbaik untuk Gaussian noise, sangat stabil

### Frei-Chen (#1 Salt&Pepper, #1 Overall)

- **Kelebihan**: Terbaik untuk Salt & Pepper, runner-up untuk Gaussian
- **MSE Ratio**: 978.13 / 219.09 = 4.47x (paling robust terhadap perubahan tipe noise)
- **Peningkatan vs terbaik**: Gaussian +2.8%, Salt&Pepper +0%
- **Stabilitas**: #1 (selisih 759.04 - paling stabil)
- **Rata-rata MSE**: 598.61 (terendah)
- **Rekomendasi**: **Pilihan terbaik overall**, paling balanced untuk kedua tipe noise

### Sobel (#3 Overall)

- **Kelebihan**: Performa tengah untuk kedua tipe noise, reliable
- **MSE Ratio**: 1025.78 / 235.92 = 4.35x
- **Peningkatan vs terbaik**: Gaussian +10.7%, Salt&Pepper +4.9%
- **Stabilitas**: #3 (selisih 789.86)
- **Rekomendasi**: Pilihan aman ketika tidak tahu tipe noise

### Roberts (#4 Overall - Terburuk)

- **Kelebihan**: Kernel kecil (2x2), komputasi cepat
- **Kelemahan**: Paling sensitif terhadap noise (MSE tertinggi untuk kedua tipe)
- **MSE Ratio**: 1599.55 / 422.32 = 3.79x
- **Peningkatan vs terbaik**: Gaussian +98.1%, Salt&Pepper +63.5% (**jauh tertinggal!**)
- **Stabilitas**: #4 (selisih 1177.23 - paling tidak stabil)
- **Rata-rata MSE**: 1010.93 (1.7x lipat dari terbaik)
- **Rekomendasi**: Hindari untuk citra noisy, gunakan pre-filtering jika terpaksa pakai Roberts

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

- **Chart utama** (`mse_chart.png` - 1400x800px):

  - Perbandingan rata-rata MSE semua operator pada kedua tipe noise
  - Grid horizontal untuk kemudahan baca
  - Legend dengan warna jelas (Salt&Pepper: merah, Gaussian: hijau)
  - Shadow effect pada bar, nilai MSE di atas setiap bar
  - Cocok untuk presentasi/laporan

- **Chart re-segmentasi** (`resegmented_comparison_chart.png` - 1200x700px):

  - Fokus pada hasil re-segmentasi citra terbaik (`landscape`)
  - Format sama dengan chart utama untuk konsistensi
  - Menampilkan performa semua operator pada citra yang sama

- **Tabel perbandingan lengkap** (`comparison_table.csv`):
  - Metadata: Info citra terbaik yang di-resegmentasi
  - Tabel utama: MSE per operator dengan ranking
  - Ranking overall: Urutan dari terbaik ke terburuk
  - Analisis: Terbaik per noise type, overall, dan terburuk
  - Stabilitas: Ranking resistance terhadap noise
  - Persentase: Peningkatan relatif vs operator terbaik
  - Kesimpulan: 5 poin insight praktis

## 5. Data Mentah

- **Semua hasil** (`metrics.csv`): MSE detail per citra/variant/operator (semua citra input)
- **Ringkasan** (`summary.csv`): Rata-rata MSE per operator untuk saltpepper vs gaussian
- **Re-segmentasi** (`resegmented_metrics.csv`): MSE hasil re-segmentasi citra `landscape` dengan semua operator
- **Gambar hasil terbaik**:
  - `best_noise/<operator>/`: Hasil dengan MSE terendah per operator (dari semua citra)
  - `resegmented_best/gaussian/`: 8 gambar (4 operator × 2 untuk citra `landscape` gaussian + baseline)
  - `resegmented_best/saltpepper/`: 8 gambar (4 operator × 2 untuk citra `landscape` saltpepper + baseline)

## 6. Cara Menggunakan Hasil untuk Laporan

1. **Slide presentasi**: Gunakan `mse_chart.png` dan `resegmented_comparison_chart.png`
2. **Tabel di laporan**: Copy dari `comparison_table.csv` (sudah format rapi)
3. **Contoh gambar**: Ambil dari `resegmented_best/` (semua dari citra yang sama untuk konsistensi)
4. **Analisis tertulis**: Gunakan section 2 & 3 dari file ini
5. **Kesimpulan**: Ambil dari bagian akhir `comparison_table.csv`
