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

    width, height = 960, 600
    margin, bottom = 80, 120
    group_width = (width - 2 * margin) // len(operators)
    bar_width = int(group_width / max(len(noises), 1) * 0.8)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    max_mse = max(summary.values())
    max_mse = max_mse if max_mse > 0 else 1.0
    colors = {
        "saltpepper": (50, 102, 204),
        "gaussian": (30, 150, 30),
    }

    for i, op in enumerate(operators):
        base_x = margin + i * group_width
        for j, noise in enumerate(noises):
            val = summary.get((op, noise), 0.0)
            bar_h = int((val / max_mse) * (height - margin - bottom))
            x1 = base_x + j * (bar_width + 6)
            y1 = height - bottom - bar_h
            x2 = x1 + bar_width
            y2 = height - bottom
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colors.get(noise, (60, 60, 60)), -1)
            cv2.putText(
                canvas,
                f"{val:.1f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (40, 40, 40),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            op,
            (base_x, height - bottom + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Rata-rata MSE vs grayscale bersih (lebih kecil lebih baik)",
        (margin, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
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


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "variant", "operator", "mse_to_gray", "edge_path"]
        )
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
    write_csv(all_records, csv_path)

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


if __name__ == "__main__":
    main()
