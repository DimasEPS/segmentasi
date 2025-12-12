## Segmentasi (Roberts, Prewitt, Sobel, Frei-Chen)

Script `main.py` menjalankan segmentasi tepi berbasis discontinuity untuk empat kondisi citra: warna asli, grayscale bersih, grayscale dengan derau salt & pepper, dan grayscale dengan derau Gaussian. Hasilnya mencakup peta tepi, nilai MSE (noisy vs grayscale bersih), ringkasan rata-rata, grafik batang, serta salinan hasil terbaik per operator.

### Struktur input/output
- Input: `../Pengcit_Noise-And-Filter/images/original/*.jpg|png` (dipakai apa adanya).
- Output: `output/`
  - `sources/` contoh citra yang dipakai.
  - `edges/<variant>/<operator>.png` hasil segmentasi.
  - `metrics.csv` detail MSE per citra/variant/operator.
  - `summary.csv` rata-rata MSE per operator (saltpepper vs gaussian).
  - `mse_chart.png` grafik batang MSE.
  - `best_noise/` salinan hasil dengan MSE terendah per operator.

### Menjalankan
Aktifkan terlebih dahulu virtualenv bawaan tugas noise & filter (berisi `opencv-python` dan `numpy`):
```bash
source ../Pengcit_Noise-And-Filter/venv/bin/activate
# atau langsung:
../Pengcit_Noise-And-Filter/venv/bin/python main.py
```
Output tersimpan di folder `output` (dibuat otomatis). Nilai MSE dihitung relatif terhadap hasil segmentasi grayscale bersih, sehingga makin kecil makin mirip citra tanpa derau.
