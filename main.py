import csv
import os
from pathlib import Path

import cv2
import numpy as np

# Direktori sumber dari tugas noise & filter sebelumnya
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "Pengcit_Noise-And-Filter"
ORIGINAL_DIR = REFERENCE_DIR / "images" / "original"

# Direktori keluaran baru untuk tugas segmentasi
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Konfigurasi noise
SALT_PEPPER_PROB = 0.05
GAUSSIAN_SIGMA = 15

RNG = np.random.default_rng(42)

# Kernel operator discontinuity
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


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def add_salt_and_pepper(gray: np.ndarray, prob: float) -> np.ndarray:
    noisy = gray.copy()
    rand = RNG.random(gray.shape)
    noisy[rand < prob / 2] = 0
    noisy[(rand >= prob / 2) & (rand < prob)] = 255
    return noisy


def add_gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    noise = RNG.normal(0, sigma, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def edge_magnitude(gray: np.ndarray, kernels: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    kx, ky = kernels
    gx = cv2.filter2D(gray, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0
    return mag.astype(np.uint8)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff ** 2))


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def draw_bar_chart(summary: dict[tuple[str, str], float], save_path: Path) -> None:
    operators = sorted({key[0] for key in summary})
    noises = sorted({key[1] for key in summary})
    if not operators or not noises:
        return

    width, height = 1400, 800
    margin_left, margin_right = 100, 80
    margin_top, margin_bottom = 120, 150
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    max_mse = max(summary.values())
    max_mse = max_mse if max_mse > 0 else 1.0
    
    colors = {
        "saltpepper": (220, 120, 60),
        "gaussian": (60, 180, 75),
    }
    
    noise_labels = {
        "saltpepper": "Salt & Pepper",
        "gaussian": "Gaussian"
    }

    # Grid horizontal
    grid_lines = 5
    for i in range(grid_lines + 1):
        y = margin_top + int(i * plot_height / grid_lines)
        cv2.line(canvas, (margin_left, y), (margin_left + plot_width, y), (200, 200, 200), 1)
        mse_val = max_mse * (1 - i / grid_lines)
        cv2.putText(
            canvas,
            f"{mse_val:.0f}",
            (margin_left - 70, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            lineType=cv2.LINE_AA,
        )

    # Bars
    group_width = plot_width // len(operators)
    bar_width = int(group_width / (len(noises) + 1) * 0.7)
    
    for i, op in enumerate(operators):
        base_x = margin_left + i * group_width + group_width // 2 - (len(noises) * bar_width) // 2
        
        for j, noise in enumerate(noises):
            val = summary.get((op, noise), 0.0)
            bar_h = int((val / max_mse) * plot_height)
            
            x1 = base_x + j * (bar_width + 10)
            y1 = margin_top + plot_height - bar_h
            x2 = x1 + bar_width
            y2 = margin_top + plot_height
            
            # Bar dengan shadow
            cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (180, 180, 180), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colors.get(noise, (60, 60, 60)), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), 2)
            
            # Nilai MSE di atas bar
            text = f"{val:.1f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                canvas,
                text,
                (x1 + (bar_width - tw) // 2, max(y1 - 10, margin_top + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                2,
                lineType=cv2.LINE_AA,
            )
        
        # Label operator
        cv2.putText(
            canvas,
            op.upper(),
            (margin_left + i * group_width + group_width // 2 - 40, margin_top + plot_height + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    # Title
    cv2.putText(
        canvas,
        "Perbandingan MSE Hasil Segmentasi (vs Grayscale Bersih)",
        (width // 2 - 320, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    
    # Subtitle
    cv2.putText(
        canvas,
        "Semakin kecil MSE, semakin mirip dengan baseline (grayscale tanpa noise)",
        (width // 2 - 280, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 80, 80),
        1,
        lineType=cv2.LINE_AA,
    )

    # Y-axis label
    cv2.putText(
        canvas,
        "MSE",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    # Legend
    legend_x = width - margin_right - 180
    legend_y = margin_top + 30
    for idx, (noise, label) in enumerate(noise_labels.items()):
        y_pos = legend_y + idx * 40
        cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 30, y_pos + 25), colors[noise], -1)
        cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 30, y_pos + 25), (40, 40, 40), 2)
        cv2.putText(
            canvas,
            label,
            (legend_x + 40, y_pos + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    save_image(save_path, canvas)


def process_image(img_path: Path) -> list[dict]:
    color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if color is None:
        return []

    base = img_path.stem
    base_dir = OUTPUT_DIR / base
    records: list[dict] = []

    gray = to_gray(color)
    variants = {
        "original_color": color,
        "gray": gray,
        "saltpepper": add_salt_and_pepper(gray, SALT_PEPPER_PROB),
        "gaussian": add_gaussian(gray, GAUSSIAN_SIGMA),
    }

    for name, img in variants.items():
        save_image(base_dir / "sources" / f"{base}_{name}.png", img)

    reference_edges: dict[str, np.ndarray] = {}
    for op_name, kernels in OPERATORS.items():
        ref_edges = edge_magnitude(gray, kernels)
        reference_edges[op_name] = ref_edges
        save_image(base_dir / "edges" / "gray" / f"{base}_{op_name}.png", ref_edges)
        records.append(
            {
                "image": base,
                "variant": "gray",
                "operator": op_name,
                "mse_to_gray": 0.0,
                "edge_path": os.path.relpath(
                    base_dir / "edges" / "gray" / f"{base}_{op_name}.png",
                    OUTPUT_DIR,
                ),
            }
        )

    for variant, img in variants.items():
        if variant == "gray":
            continue
        img_gray = to_gray(img)
        for op_name, kernels in OPERATORS.items():
            edges = edge_magnitude(img_gray, kernels)
            edge_path = base_dir / "edges" / variant / f"{base}_{variant}_{op_name}.png"
            save_image(edge_path, edges)
            records.append(
                {
                    "image": base,
                    "variant": variant,
                    "operator": op_name,
                    "mse_to_gray": mse(reference_edges[op_name], edges),
                    "edge_path": os.path.relpath(edge_path, OUTPUT_DIR),
                }
            )

    return records


def write_csv(records: list[dict], path: Path, fieldnames: list[str] = None) -> None:
    if not records:
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict]) -> tuple[dict[tuple[str, str], float], list[dict]]:
    summary: dict[tuple[str, str], list[float]] = {}
    best_choices: list[dict] = []
    for rec in records:
        variant = rec["variant"]
        op = rec["operator"]
        if variant not in ("saltpepper", "gaussian"):
            continue
        summary.setdefault((op, variant), []).append(rec["mse_to_gray"])

    averaged = {k: float(np.mean(v)) for k, v in summary.items()}

    for op in OPERATORS:
        candidates = [
            r
            for r in records
            if r["operator"] == op and r["variant"] in ("saltpepper", "gaussian")
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda item: item["mse_to_gray"])
        best_choices.append(best)

    return averaged, best_choices


def save_best_copies(best_choices: list[dict]) -> None:
    for rec in best_choices:
        src = OUTPUT_DIR / rec["edge_path"]
        target = OUTPUT_DIR / "best_noise" / rec["operator"] / (
            Path(rec["edge_path"]).name
        )
        if src.exists():
            data = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            save_image(target, data)


def resegment_best_prewitt(records: list[dict]) -> tuple[list[dict], dict]:
    """Segmentasi ulang citra terbaik Prewitt dengan semua operator"""
    prewitt_records = [
        r for r in records
        if r["operator"] == "prewitt" and r["variant"] in ("saltpepper", "gaussian")
    ]
    
    if not prewitt_records:
        return [], {}
    
    # Pilih citra terbaik per noise type untuk Prewitt
    best_by_noise = {}
    for noise_type in ("saltpepper", "gaussian"):
        candidates = [r for r in prewitt_records if r["variant"] == noise_type]
        if candidates:
            best = min(candidates, key=lambda x: x["mse_to_gray"])
            best_by_noise[noise_type] = best
    
    reseg_records = []
    comparison_table = {}
    
    for noise_type, best_rec in best_by_noise.items():
        img_name = best_rec["image"]
        variant = best_rec["variant"]
        
        # Load citra noisy dan grayscale baseline
        base_dir = OUTPUT_DIR / img_name
        noisy_path = base_dir / "sources" / f"{img_name}_{variant}.png"
        gray_path = base_dir / "sources" / f"{img_name}_gray.png"
        
        if not noisy_path.exists() or not gray_path.exists():
            continue
        
        noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        
        if noisy_img is None or gray_img is None:
            continue
        
        # Segmentasi dengan semua operator
        reseg_dir = OUTPUT_DIR / "resegmented_best" / noise_type
        
        for op_name, kernels in OPERATORS.items():
            # Segmentasi noisy
            noisy_edges = edge_magnitude(noisy_img, kernels)
            noisy_out = reseg_dir / f"{img_name}_{variant}_{op_name}.png"
            save_image(noisy_out, noisy_edges)
            
            # Segmentasi grayscale baseline
            gray_edges = edge_magnitude(gray_img, kernels)
            gray_out = reseg_dir / f"{img_name}_gray_{op_name}.png"
            save_image(gray_out, gray_edges)
            
            # Hitung MSE
            mse_val = mse(gray_edges, noisy_edges)
            
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


def draw_comparison_chart(comparison: dict[tuple[str, str], float], save_path: Path) -> None:
    """Grafik perbandingan untuk hasil re-segmentasi citra terbaik"""
    if not comparison:
        return
    
    operators = sorted({key[1] for key in comparison})
    noises = sorted({key[0] for key in comparison})
    
    width, height = 1200, 700
    margin_left, margin_right = 100, 80
    margin_top, margin_bottom = 120, 150
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    
    max_mse = max(comparison.values()) if comparison else 1.0
    max_mse = max_mse if max_mse > 0 else 1.0
    
    colors = {
        "saltpepper": (220, 120, 60),
        "gaussian": (60, 180, 75),
    }
    
    noise_labels = {
        "saltpepper": "Salt & Pepper (Best)",
        "gaussian": "Gaussian (Best)"
    }
    
    # Grid
    grid_lines = 5
    for i in range(grid_lines + 1):
        y = margin_top + int(i * plot_height / grid_lines)
        cv2.line(canvas, (margin_left, y), (margin_left + plot_width, y), (200, 200, 200), 1)
        mse_val = max_mse * (1 - i / grid_lines)
        cv2.putText(
            canvas,
            f"{mse_val:.0f}",
            (margin_left - 70, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            lineType=cv2.LINE_AA,
        )
    
    # Bars
    group_width = plot_width // len(operators)
    bar_width = int(group_width / (len(noises) + 1) * 0.7)
    
    for i, op in enumerate(operators):
        base_x = margin_left + i * group_width + group_width // 2 - (len(noises) * bar_width) // 2
        
        for j, noise in enumerate(noises):
            val = comparison.get((noise, op), 0.0)
            bar_h = int((val / max_mse) * plot_height)
            
            x1 = base_x + j * (bar_width + 10)
            y1 = margin_top + plot_height - bar_h
            x2 = x1 + bar_width
            y2 = margin_top + plot_height
            
            cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (180, 180, 180), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colors.get(noise, (60, 60, 60)), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), 2)
            
            text = f"{val:.1f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                canvas,
                text,
                (x1 + (bar_width - tw) // 2, max(y1 - 10, margin_top + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                2,
                lineType=cv2.LINE_AA,
            )
        
        cv2.putText(
            canvas,
            op.upper(),
            (margin_left + i * group_width + group_width // 2 - 40, margin_top + plot_height + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    
    # Title
    cv2.putText(
        canvas,
        "Re-Segmentasi Citra Terbaik Prewitt dengan Semua Operator",
        (width // 2 - 340, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    
    cv2.putText(
        canvas,
        "MSE hasil segmentasi citra noisy terbaik vs grayscale baseline",
        (width // 2 - 260, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 80, 80),
        1,
        lineType=cv2.LINE_AA,
    )
    
    # Y-axis label
    cv2.putText(
        canvas,
        "MSE",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    
    # Legend
    legend_x = width - margin_right - 220
    legend_y = margin_top + 30
    for idx, (noise, label) in enumerate(noise_labels.items()):
        y_pos = legend_y + idx * 40
        cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 30, y_pos + 25), colors[noise], -1)
        cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 30, y_pos + 25), (40, 40, 40), 2)
        cv2.putText(
            canvas,
            label,
            (legend_x + 40, y_pos + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    
    save_image(save_path, canvas)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")
    image_paths = [
        ORIGINAL_DIR / f for f in sorted(os.listdir(ORIGINAL_DIR))
        if f.lower().endswith(valid_ext)
    ]

    if not image_paths:
        raise FileNotFoundError(f"Tidak ada citra di {ORIGINAL_DIR}")

    all_records: list[dict] = []
    for path in image_paths:
        all_records.extend(process_image(path))

    csv_path = OUTPUT_DIR / "metrics.csv"
    write_csv(all_records, csv_path, ["image", "variant", "operator", "mse_to_gray", "edge_path"])

    averaged, best_choices = summarize(all_records)
    summary_path = OUTPUT_DIR / "summary.csv"
    write_csv(
        [
            {
                "image": "rata-rata",
                "variant": variant,
                "operator": op,
                "mse_to_gray": mse_val,
                "edge_path": "",
            }
            for (op, variant), mse_val in averaged.items()
        ],
        summary_path,
    )

    chart_path = OUTPUT_DIR / "mse_chart.png"
    draw_bar_chart(averaged, chart_path)
    save_best_copies(best_choices)
    
    # Re-segmentasi citra terbaik Prewitt dengan semua operator
    reseg_records, comparison_table = resegment_best_prewitt(all_records)
    
    if reseg_records:
        # Simpan CSV re-segmentasi
        reseg_csv_path = OUTPUT_DIR / "resegmented_metrics.csv"
        write_csv(reseg_records, reseg_csv_path)
        
        # Buat grafik perbandingan
        comparison_chart_path = OUTPUT_DIR / "resegmented_comparison_chart.png"
        draw_comparison_chart(comparison_table, comparison_chart_path)
        
        # Buat tabel perbandingan lengkap dengan analisis
        comparison_table_path = OUTPUT_DIR / "comparison_table.csv"
        operators = sorted({key[1] for key in comparison_table})
        noises = sorted({key[0] for key in comparison_table})
        
        # Hitung statistik untuk setiap operator
        operator_stats = {}
        for op in operators:
            mse_values = [comparison_table.get((n, op), 0.0) for n in noises]
            operator_stats[op] = {
                "gaussian": comparison_table.get(("gaussian", op), 0.0),
                "saltpepper": comparison_table.get(("saltpepper", op), 0.0),
                "avg": np.mean(mse_values),
                "diff": abs(mse_values[0] - mse_values[1]) if len(mse_values) == 2 else 0,
            }
        
        # Ranking per noise type
        gaussian_ranking = sorted(operators, key=lambda x: operator_stats[x]["gaussian"])
        saltpepper_ranking = sorted(operators, key=lambda x: operator_stats[x]["saltpepper"])
        avg_ranking = sorted(operators, key=lambda x: operator_stats[x]["avg"])
        
        with comparison_table_path.open("w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["=== PERBANDINGAN MSE HASIL SEGMENTASI ==="])
            writer.writerow(["Semakin kecil MSE, semakin mirip dengan baseline (grayscale tanpa noise)"])
            writer.writerow([])
            
            # Tabel utama
            writer.writerow(["Operator", "Gaussian MSE", "Rank", "Salt&Pepper MSE", "Rank", "Rata-rata", "Selisih (S&P - Gaussian)"])
            for op in operators:
                stats = operator_stats[op]
                g_rank = gaussian_ranking.index(op) + 1
                sp_rank = saltpepper_ranking.index(op) + 1
                writer.writerow([
                    op.upper(),
                    f"{stats['gaussian']:.2f}",
                    f"#{g_rank}",
                    f"{stats['saltpepper']:.2f}",
                    f"#{sp_rank}",
                    f"{stats['avg']:.2f}",
                    f"{stats['saltpepper'] - stats['gaussian']:.2f}"
                ])
            
            writer.writerow([])
            writer.writerow(["=== RANKING OVERALL (berdasarkan rata-rata MSE) ==="])
            for i, op in enumerate(avg_ranking, 1):
                stats = operator_stats[op]
                writer.writerow([f"#{i}", op.upper(), f"Rata-rata MSE: {stats['avg']:.2f}"])
            
            writer.writerow([])
            writer.writerow(["=== ANALISIS ==="])
            
            # Best per noise
            best_gaussian = gaussian_ranking[0]
            best_saltpepper = saltpepper_ranking[0]
            best_overall = avg_ranking[0]
            worst_overall = avg_ranking[-1]
            
            writer.writerow(["Terbaik untuk Gaussian:", best_gaussian.upper(), f"MSE: {operator_stats[best_gaussian]['gaussian']:.2f}"])
            writer.writerow(["Terbaik untuk Salt&Pepper:", best_saltpepper.upper(), f"MSE: {operator_stats[best_saltpepper]['saltpepper']:.2f}"])
            writer.writerow(["Terbaik Overall:", best_overall.upper(), f"MSE rata-rata: {operator_stats[best_overall]['avg']:.2f}"])
            writer.writerow(["Terburuk Overall:", worst_overall.upper(), f"MSE rata-rata: {operator_stats[worst_overall]['avg']:.2f}"])
            
            writer.writerow([])
            writer.writerow(["=== STABILITAS (resistance terhadap noise) ==="])
            stability_ranking = sorted(operators, key=lambda x: operator_stats[x]["diff"])
            for i, op in enumerate(stability_ranking, 1):
                stats = operator_stats[op]
                writer.writerow([
                    f"#{i}",
                    op.upper(),
                    f"Selisih: {stats['diff']:.2f}",
                    "(semakin kecil = lebih stabil)"
                ])
            
            writer.writerow([])
            writer.writerow(["=== PERSENTASE PENINGKATAN (relatif terhadap terbaik) ==="])
            best_g_mse = operator_stats[gaussian_ranking[0]]["gaussian"]
            best_sp_mse = operator_stats[saltpepper_ranking[0]]["saltpepper"]
            
            for op in operators:
                stats = operator_stats[op]
                g_increase = ((stats["gaussian"] - best_g_mse) / best_g_mse * 100) if best_g_mse > 0 else 0
                sp_increase = ((stats["saltpepper"] - best_sp_mse) / best_sp_mse * 100) if best_sp_mse > 0 else 0
                writer.writerow([
                    op.upper(),
                    f"Gaussian: +{g_increase:.1f}%",
                    f"Salt&Pepper: +{sp_increase:.1f}%"
                ])
            
            writer.writerow([])
            writer.writerow(["=== KESIMPULAN ==="])
            writer.writerow(["1. Noise Gaussian lebih mudah ditangani (MSE lebih rendah) dibanding Salt&Pepper untuk semua operator"])
            writer.writerow([f"2. {best_overall.upper()} memberikan hasil terbaik secara keseluruhan (MSE rata-rata terendah)"])
            writer.writerow([f"3. {worst_overall.upper()} paling sensitif terhadap noise (MSE hingga {operator_stats[worst_overall]['avg']/operator_stats[best_overall]['avg']:.1f}x lipat)"])
            writer.writerow([f"4. {stability_ranking[0].upper()} paling stabil (selisih performa antar noise paling kecil)"])
            
            # Pilih rekomendasi yang berbeda jika best_overall sama dengan stability_ranking[0]
            second_rec = stability_ranking[1] if best_overall == stability_ranking[0] else stability_ranking[0]
            writer.writerow([f"5. Untuk aplikasi real-world, rekomendasikan {best_overall.upper()} (optimal) atau {second_rec.upper()} (stabil)"])


if __name__ == "__main__":
    main()
