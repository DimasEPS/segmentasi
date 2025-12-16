# Dokumentasi Fungsi Segmentasi

Dokumentasi lengkap untuk setiap fungsi yang digunakan dalam implementasi segmentasi tepi berbasis discontinuity.

---

## 1. Kernel Operator Discontinuity

### Definisi Kernel

```python
import numpy as np

SQRT2 = np.sqrt(2, dtype=np.float32)

OPERATORS = {
    "roberts": (
        np.array([[1, 0], [0, -1]], dtype=np.float32),
        np.array([[0, 1], [-1, 0]], dtype=np.float32),
    ),
    "prewitt": (
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
        np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32),
    ),
    "sobel": (
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
    ),
    "frei-chen": (
        np.array([[1, SQRT2, 1], [0, 0, 0], [-1, -SQRT2, -1]], dtype=np.float32),
        np.array([[1, 0, -1], [SQRT2, 0, -SQRT2], [1, 0, -1]], dtype=np.float32),
    ),
}
```

**Penjelasan:**

- Setiap operator memiliki 2 kernel: horizontal (Gx) dan vertikal (Gy)
- **Roberts (2×2)**: Kernel terkecil, komputasi cepat tapi sensitif noise
- **Prewitt (3×3)**: Bobot merata, mudah diimplementasi
- **Sobel (3×3)**: Bobot lebih besar di tengah (2), lebih robust
- **Frei-Chen (3×3)**: Menggunakan √2 untuk diagonal, optimal secara matematis

**Output:** Dictionary berisi pasangan kernel (Gx, Gy) untuk setiap operator

---

## 1. Roberts Operator

### Kernel

```python
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
```

### Segmentasi

```python
edges = edge_magnitude(gray, (roberts_x, roberts_y))
```

---

## 2. Prewitt Operator

### Kernel

```python
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
```

### Segmentasi

```python
edges = edge_magnitude(gray, (prewitt_x, prewitt_y))
```

---

## 3. Sobel Operator

### Kernel

```python
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
```

### Segmentasi

```python
edges = edge_magnitude(gray, (sobel_x, sobel_y))
```

---

## 4. Frei-Chen Operator

### Kernel

```python
SQRT2 = np.sqrt(2, dtype=np.float32)
frei_chen_x = np.array([[1, SQRT2, 1], [0, 0, 0], [-1, -SQRT2, -1]], dtype=np.float32)
frei_chen_y = np.array([[1, 0, -1], [SQRT2, 0, -SQRT2], [1, 0, -1]], dtype=np.float32)
```

### Segmentasi

```python
edges = edge_magnitude(gray, (frei_chen_x, frei_chen_y))
```

---

## Fungsi Universal Segmentasi

```python
def edge_magnitude(gray: np.ndarray, kernels: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    kx, ky = kernels
    gx = cv2.filter2D(gray, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0
    return mag.astype(np.uint8)
```

---

## 2. Konversi Grayscale

### Fungsi: `to_gray()`

```python
def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Konversi citra berwarna ke grayscale.

    Args:
        img: Citra input (BGR color atau grayscale)

    Returns:
        Citra grayscale (1 channel)
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Penjelasan:**

- Jika citra sudah grayscale (2D), langsung return
- Jika berwarna (3D), konversi ke grayscale dengan formula weighted:
  - `Gray = 0.299*R + 0.587*G + 0.114*B`
- OpenCV menggunakan `COLOR_BGR2GRAY` karena format BGR

**Input:** Citra BGR (H×W×3) atau Grayscale (H×W)  
**Output:** Citra Grayscale (H×W)

---

## 3. Penambahan Noise

### Fungsi: `add_salt_and_pepper()`

```python
def add_salt_and_pepper(gray: np.ndarray, prob: float) -> np.ndarray:
    """
    Menambahkan noise Salt & Pepper (impulse noise).

    Args:
        gray: Citra grayscale input
        prob: Probabilitas noise (0.0 - 1.0)

    Returns:
        Citra grayscale dengan noise salt & pepper
    """
    noisy = gray.copy()
    rand = RNG.random(gray.shape)
    noisy[rand < prob / 2] = 0        # Salt (hitam)
    noisy[(rand >= prob / 2) & (rand < prob)] = 255  # Pepper (putih)
    return noisy
```

**Penjelasan:**

- **Salt**: Piksel acak diset ke 0 (hitam)
- **Pepper**: Piksel acak diset ke 255 (putih)
- Probabilitas dibagi 2: setengah untuk salt, setengah untuk pepper
- Menggunakan RNG dengan seed 42 untuk reproducibility

**Input:** Grayscale (H×W), probabilitas (misal: 0.05 = 5%)  
**Output:** Grayscale dengan noise impulsif

---

### Fungsi: `add_gaussian()`

```python
def add_gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Menambahkan noise Gaussian (additive noise).

    Args:
        gray: Citra grayscale input
        sigma: Standard deviasi noise

    Returns:
        Citra grayscale dengan noise gaussian
    """
    noise = RNG.normal(0, sigma, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)
```

**Penjelasan:**

- Generate noise dari distribusi normal: mean=0, std=sigma
- Noise bersifat aditif (ditambahkan ke nilai piksel)
- `np.clip()` memastikan nilai tetap dalam range [0, 255]
- Konversi kembali ke uint8 untuk format gambar standar

**Input:** Grayscale (H×W), sigma (misal: 15)  
**Output:** Grayscale dengan noise aditif terdistribusi normal

---

## 4. Deteksi Tepi (Edge Detection)

### Fungsi: `edge_magnitude()`

```python
def edge_magnitude(gray: np.ndarray, kernels: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Menghitung magnitude gradien untuk deteksi tepi.

    Args:
        gray: Citra grayscale input
        kernels: Tuple (kernel_x, kernel_y) untuk operator

    Returns:
        Peta tepi (edge map) dengan nilai 0-255
    """
    kx, ky = kernels
    # Konvolusi dengan kernel X dan Y
    gx = cv2.filter2D(gray, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)

    # Hitung magnitude: sqrt(Gx² + Gy²)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Normalisasi ke range 0-255
    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0
    return mag.astype(np.uint8)
```

**Penjelasan:**

1. **Konvolusi**: Terapkan kernel Gx dan Gy pada citra

   - `cv2.filter2D()`: Operasi konvolusi 2D
   - `cv2.CV_32F`: Gunakan float32 untuk presisi
   - `BORDER_REFLECT`: Refleksikan piksel di tepi untuk menghindari border effect

2. **Magnitude Gradien**:

   ```
   Magnitude = √(Gx² + Gy²)
   ```

   - Gx: Gradien arah horizontal (perubahan intensitas kiri-kanan)
   - Gy: Gradien arah vertikal (perubahan intensitas atas-bawah)
   - Magnitude: Kekuatan perubahan intensitas (edge strength)

3. **Normalisasi**: Scale magnitude ke range 0-255 untuk visualisasi

**Input:** Grayscale (H×W), tuple kernel (Gx, Gy)  
**Output:** Edge map (H×W) dengan nilai 0-255

**Contoh Proses:**

```
Citra Original (3×3):        Setelah Konvolusi Gx:    Setelah Konvolusi Gy:
  10  20  30                    -30  -30  -30              30   60   30
  40  50  60          →         -30  -30  -30      dan     0    0    0
  70  80  90                    -30  -30  -30             -30  -60  -30

Magnitude = √((-30)² + (30)²) = √(900 + 900) = 42.43
```

---

## 5. Evaluasi Kualitas (MSE)

### Fungsi: `mse()`

```python
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Menghitung Mean Squared Error antara dua citra.

    Args:
        a: Citra pertama (baseline/referensi)
        b: Citra kedua (hasil segmentasi yang dievaluasi)

    Returns:
        Nilai MSE (semakin kecil semakin mirip)
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff ** 2))
```

**Penjelasan:**

Formula MSE:

```
MSE = (1 / N) × Σ(A[i] - B[i])²
```

Dimana:

- N = jumlah total piksel (H × W)
- A[i] = nilai piksel ke-i dari citra referensi
- B[i] = nilai piksel ke-i dari citra hasil

**Interpretasi:**

- **MSE = 0**: Kedua citra identik
- **MSE kecil (< 500)**: Sangat mirip, operator robust terhadap noise
- **MSE sedang (500-1000)**: Cukup mirip, degradasi moderate
- **MSE besar (> 1000)**: Sangat berbeda, operator sensitif terhadap noise

**Input:** Dua citra dengan ukuran sama (H×W)  
**Output:** Float (nilai MSE)

---

## 6. Visualisasi Grafik

### Fungsi: `draw_bar_chart()`

```python
def draw_bar_chart(summary: dict[tuple[str, str], float], save_path: Path) -> None:
    """
    Membuat grafik batang perbandingan MSE antar operator.

    Args:
        summary: Dict dengan key (operator, noise_type) dan value MSE
        save_path: Path untuk menyimpan grafik PNG
    """
    operators = sorted({key[0] for key in summary})
    noises = sorted({key[1] for key in summary})

    if not operators or not noises:
        return

    # Konfigurasi canvas
    width, height = 1400, 800
    margin_left, margin_right = 100, 80
    margin_top, margin_bottom = 120, 150

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    max_mse = max(summary.values())
    max_mse = max_mse if max_mse > 0 else 1.0

    colors = {
        "saltpepper": (220, 120, 60),   # Orange-red
        "gaussian": (60, 180, 75),       # Green
    }

    # 1. Gambar grid horizontal
    grid_lines = 5
    for i in range(grid_lines + 1):
        y = margin_top + int(i * plot_height / grid_lines)
        cv2.line(canvas, (margin_left, y),
                (margin_left + plot_width, y), (200, 200, 200), 1)

        # Label MSE value
        mse_val = max_mse * (1 - i / grid_lines)
        cv2.putText(canvas, f"{mse_val:.0f}",
                   (margin_left - 70, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1,
                   lineType=cv2.LINE_AA)

    # 2. Gambar bars untuk setiap operator
    group_width = plot_width // len(operators)
    bar_width = int(group_width / (len(noises) + 1) * 0.7)

    for i, op in enumerate(operators):
        base_x = margin_left + i * group_width + group_width // 2 - \
                 (len(noises) * bar_width) // 2

        for j, noise in enumerate(noises):
            val = summary.get((op, noise), 0.0)
            bar_h = int((val / max_mse) * plot_height)

            x1 = base_x + j * (bar_width + 10)
            y1 = margin_top + plot_height - bar_h
            x2 = x1 + bar_width
            y2 = margin_top + plot_height

            # Shadow effect
            cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2),
                         (180, 180, 180), -1)
            # Bar
            cv2.rectangle(canvas, (x1, y1), (x2, y2),
                         colors.get(noise, (60, 60, 60)), -1)
            # Border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), 2)

            # Nilai MSE di atas bar
            text = f"{val:.1f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            cv2.putText(canvas, text,
                       (x1 + (bar_width - tw) // 2,
                        max(y1 - 10, margin_top + 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2,
                       lineType=cv2.LINE_AA)

        # Label operator
        cv2.putText(canvas, op.upper(),
                   (margin_left + i * group_width + group_width // 2 - 40,
                    margin_top + plot_height + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                   lineType=cv2.LINE_AA)

    # 3. Title dan legend
    cv2.putText(canvas, "Perbandingan MSE Hasil Segmentasi",
               (width // 2 - 320, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
               lineType=cv2.LINE_AA)

    # Legend
    legend_x = width - margin_right - 180
    legend_y = margin_top + 30
    noise_labels = {"saltpepper": "Salt & Pepper", "gaussian": "Gaussian"}

    for idx, (noise, label) in enumerate(noise_labels.items()):
        y_pos = legend_y + idx * 40
        cv2.rectangle(canvas, (legend_x, y_pos),
                     (legend_x + 30, y_pos + 25), colors[noise], -1)
        cv2.rectangle(canvas, (legend_x, y_pos),
                     (legend_x + 30, y_pos + 25), (40, 40, 40), 2)
        cv2.putText(canvas, label, (legend_x + 40, y_pos + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                   lineType=cv2.LINE_AA)

    # 4. Simpan grafik
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
```

**Penjelasan:**

Grafik dibuat dengan OpenCV tanpa library plotting eksternal:

1. **Canvas**: Buat background 1400×800 warna abu muda (250)
2. **Grid**: 5 garis horizontal dengan label nilai MSE
3. **Bars**:
   - Shadow hitam untuk efek depth
   - Warna berbeda per noise type (orange=saltpepper, hijau=gaussian)
   - Border hitam untuk outline
   - Nilai MSE ditampilkan di atas setiap bar
4. **Labels**: Nama operator di bawah, title di atas
5. **Legend**: Kotak warna dengan label di kanan atas

**Input:** Dictionary {(operator, noise_type): mse_value}  
**Output:** File PNG grafik perbandingan

---

## 7. Pipeline Utama

### Fungsi: `process_image()`

```python
def process_image(img_path: Path) -> list[dict]:
    """
    Pipeline lengkap: load → noise → segmentasi → evaluasi.

    Args:
        img_path: Path ke citra input

    Returns:
        List of dict berisi metadata hasil untuk setiap variant/operator
    """
    # 1. Load citra
    color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if color is None:
        return []

    base = img_path.stem
    base_dir = OUTPUT_DIR / base
    records = []

    # 2. Konversi grayscale
    gray = to_gray(color)

    # 3. Buat variants (original, gray, noisy)
    variants = {
        "original_color": color,
        "gray": gray,
        "saltpepper": add_salt_and_pepper(gray, SALT_PEPPER_PROB),
        "gaussian": add_gaussian(gray, GAUSSIAN_SIGMA),
    }

    # Simpan source images
    for name, img in variants.items():
        save_image(base_dir / "sources" / f"{base}_{name}.png", img)

    # 4. Segmentasi grayscale (baseline/reference)
    reference_edges = {}
    for op_name, kernels in OPERATORS.items():
        ref_edges = edge_magnitude(gray, kernels)
        reference_edges[op_name] = ref_edges
        save_image(base_dir / "edges" / "gray" / f"{base}_{op_name}.png",
                  ref_edges)

        records.append({
            "image": base,
            "variant": "gray",
            "operator": op_name,
            "mse_to_gray": 0.0,  # Baseline, MSE = 0
            "edge_path": str(...),
        })

    # 5. Segmentasi noisy images
    for variant, img in variants.items():
        if variant == "gray":
            continue

        img_gray = to_gray(img)

        for op_name, kernels in OPERATORS.items():
            # Segmentasi
            edges = edge_magnitude(img_gray, kernels)
            edge_path = base_dir / "edges" / variant / f"{base}_{variant}_{op_name}.png"
            save_image(edge_path, edges)

            # Evaluasi: bandingkan dengan baseline
            mse_val = mse(reference_edges[op_name], edges)

            records.append({
                "image": base,
                "variant": variant,
                "operator": op_name,
                "mse_to_gray": mse_val,
                "edge_path": str(...),
            })

    return records
```

**Pipeline:**

```
Input Image
    ↓
Grayscale Conversion
    ↓
├── Original (no noise)
├── Grayscale Baseline ← reference untuk evaluasi
├── Salt & Pepper Noise (5%)
└── Gaussian Noise (σ=15)
    ↓
Edge Detection (4 operators × 4 variants = 16 hasil)
    ↓
MSE Evaluation (bandingkan dengan baseline)
    ↓
Save Results (images + metrics CSV)
```

**Input:** Path ke citra sumber  
**Output:** List dict berisi metadata (image name, variant, operator, MSE, path)

---

## 8. Re-Segmentasi Citra Terbaik

### Fungsi: `resegment_best_prewitt()`

```python
def resegment_best_prewitt(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Pilih citra dengan hasil Prewitt terbaik, lalu re-segmentasi
    dengan semua operator untuk analisis mendalam.

    Args:
        records: List hasil dari process_image() untuk semua citra

    Returns:
        Tuple (reseg_records, comparison_table)
    """
    # 1. Filter hanya hasil Prewitt dengan noise
    prewitt_records = [
        r for r in records
        if r["operator"] == "prewitt" and
           r["variant"] in ("saltpepper", "gaussian")
    ]

    if not prewitt_records:
        return [], {}

    # 2. Pilih citra dengan MSE Prewitt terendah
    best_rec = min(prewitt_records, key=lambda x: x["mse_to_gray"])
    img_name = best_rec["image"]
    best_variant = best_rec["variant"]

    # 3. Load grayscale baseline
    base_dir = OUTPUT_DIR / img_name
    gray_path = base_dir / "sources" / f"{img_name}_gray.png"
    gray_img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)

    # 4. Segmentasi baseline dengan semua operator (sekali saja)
    baseline_edges = {}
    for op_name, kernels in OPERATORS.items():
        baseline_edges[op_name] = edge_magnitude(gray_img, kernels)

    reseg_records = []
    comparison_table = {}

    # 5. Proses kedua noise variant dari citra terbaik
    for noise_type in ("saltpepper", "gaussian"):
        noisy_path = base_dir / "sources" / f"{img_name}_{noise_type}.png"
        noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)

        reseg_dir = OUTPUT_DIR / "resegmented_best" / noise_type

        # 6. Segmentasi dengan semua operator
        for op_name, kernels in OPERATORS.items():
            # Segmentasi noisy
            noisy_edges = edge_magnitude(noisy_img, kernels)
            noisy_out = reseg_dir / f"{img_name}_{noise_type}_{op_name}.png"
            save_image(noisy_out, noisy_edges)

            # Simpan baseline
            gray_out = reseg_dir / f"{img_name}_gray_{op_name}.png"
            save_image(gray_out, baseline_edges[op_name])

            # Hitung MSE
            mse_val = mse(baseline_edges[op_name], noisy_edges)

            reseg_records.append({
                "image": img_name,
                "noise_type": noise_type,
                "operator": op_name,
                "mse": mse_val,
                "noisy_path": str(noisy_out.relative_to(OUTPUT_DIR)),
                "baseline_path": str(gray_out.relative_to(OUTPUT_DIR)),
            })

            comparison_table[(noise_type, op_name)] = mse_val

    return reseg_records, comparison_table
```

**Penjelasan:**

1. **Seleksi Best Image**: Pilih 1 citra dengan hasil Prewitt terbaik (MSE terendah)
2. **Fokus Analisis**: Re-segmentasi citra ini dengan semua 4 operator
3. **Konsistensi**: Semua perbandingan menggunakan citra yang sama
4. **Output**: 16 gambar (4 operator × 2 noise × 2 untuk baseline/noisy)

**Keuntungan Approach Ini:**

- Analisis adil (semua operator ditest pada citra yang sama)
- Tidak bias oleh karakteristik citra berbeda
- Mudah untuk visualisasi dan perbandingan

**Input:** List records dari semua citra  
**Output:** Records re-segmentasi + comparison table

---

## 9. Workflow Lengkap

```
main.py
  ↓
1. Load semua citra dari folder input
  ↓
2. Untuk setiap citra:
   - Konversi grayscale
   - Tambah noise (salt&pepper, gaussian)
   - Segmentasi dengan 4 operator
   - Hitung MSE vs baseline
   - Simpan hasil (images + CSV)
  ↓
3. Analisis hasil:
   - Rata-rata MSE per operator
   - Pilih best result per operator
   - Generate grafik perbandingan
  ↓
4. Re-segmentasi citra terbaik:
   - Pilih 1 citra (Prewitt MSE terendah)
   - Segmentasi dengan semua operator
   - Generate grafik & tabel comparison
  ↓
5. Output:
   - metrics.csv (detail semua)
   - summary.csv (rata-rata)
   - resegmented_metrics.csv (fokus best)
   - comparison_table.csv (analisis lengkap)
   - mse_chart.png (grafik utama)
   - resegmented_comparison_chart.png (grafik fokus)
   - best_noise/ (gambar terbaik per operator)
   - resegmented_best/ (16 gambar fokus)
```

---

## 10. Contoh Penggunaan

### Edge Detection Roberts

```python
# Load citra
img = cv2.imread("landscape.jpg", cv2.IMREAD_GRAYSCALE)

# Kernel Roberts
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

# Deteksi tepi
edges = edge_magnitude(img, (roberts_x, roberts_y))

# Simpan hasil
cv2.imwrite("landscape_roberts.png", edges)
```

### Evaluasi dengan MSE

```python
# Baseline (grayscale tanpa noise)
baseline = edge_magnitude(gray_clean, operators["prewitt"])

# Hasil dengan noise gaussian
noisy = add_gaussian(gray_clean, sigma=15)
result = edge_magnitude(noisy, operators["prewitt"])

# Hitung MSE
error = mse(baseline, result)
print(f"MSE: {error:.2f}")  # Contoh: MSE: 213.19

# Interpretasi:
# MSE < 500 → Excellent (operator robust)
# MSE 500-1000 → Good (degradasi moderate)
# MSE > 1000 → Poor (sangat sensitif noise)
```

---

## 11. Tips Implementasi

### Optimasi Performa

```python
# 1. Gunakan vectorized operations (NumPy)
# SLOW:
for i in range(height):
    for j in range(width):
        result[i, j] = np.sqrt(gx[i, j]**2 + gy[i, j]**2)

# FAST:
result = np.sqrt(gx**2 + gy**2)

# 2. Gunakan tipe data yang tepat
img = img.astype(np.float32)  # Untuk komputasi
result = result.astype(np.uint8)  # Untuk save image

# 3. Pre-allocate arrays
edges = np.zeros_like(img, dtype=np.float32)
```

### Error Handling

```python
def edge_magnitude_safe(gray: np.ndarray, kernels: tuple) -> np.ndarray:
    """Versi dengan error handling"""
    if gray.ndim != 2:
        raise ValueError("Input harus grayscale (2D)")

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    kx, ky = kernels
    if kx.shape != ky.shape:
        raise ValueError("Kernel X dan Y harus ukuran sama")

    # ... proses normal ...
    return edges
```

---

## Referensi

1. **Roberts Cross**: L. G. Roberts, "Machine Perception of Three-Dimensional Solids", 1963
2. **Prewitt Operator**: J. M. S. Prewitt, "Object Enhancement and Extraction", 1970
3. **Sobel Operator**: I. Sobel, "An Isotropic 3×3 Image Gradient Operator", 1968
4. **Frei-Chen Operator**: W. Frei and C. C. Chen, "Fast Boundary Detection", 1977

---

## Appendix: Formula Matematis

### Gradient Magnitude

```
G = √(Gx² + Gy²)
Dimana:
  Gx = I ⊗ Kx  (konvolusi dengan kernel X)
  Gy = I ⊗ Ky  (konvolusi dengan kernel Y)
```

### Mean Squared Error

```
MSE = (1/N) × Σᵢ₌₁ᴺ (Aᵢ - Bᵢ)²
```

### Gradient Direction (optional)

```
θ = arctan(Gy / Gx)
```

### Normalisasi

```
I_normalized = (I - I_min) / (I_max - I_min) × 255
```
