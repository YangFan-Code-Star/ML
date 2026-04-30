import csv
import math
import random
from pathlib import Path


FEATURE_NAMES = [
    "intensity_main",
    "light_count",
    "ambient_intensity",
    "main_height",
    "main_angle",
    "fill_ratio",
    "median_gray",
]


def simulate_row():
    intensity_main = random.uniform(0.5, 4.0)
    light_count = random.uniform(1.0, 8.0)
    ambient_intensity = random.uniform(0.0, 1.2)
    main_height = random.uniform(1.0, 9.0)
    main_angle = random.uniform(10.0, 90.0)
    fill_ratio = random.uniform(0.0, 0.9)

    angle_factor = max(0.0, math.cos(math.radians(main_angle)))
    height_decay = 1.0 / (1.0 + 0.18 * (main_height - 2.5) ** 2)
    count_gain = math.log1p(light_count) / math.log(9.0)
    interaction = intensity_main * fill_ratio

    median_gray = (
        0.08
        + 0.26 / (1.0 + math.exp(-1.15 * intensity_main))
        + 0.18 / (1.0 + math.exp(-2.8 * ambient_intensity))
        + 0.14 * count_gain
        + 0.18 * angle_factor
        + 0.10 * height_decay
        + 0.10 / (1.0 + math.exp(-1.35 * interaction))
        + 0.02 * math.sin(intensity_main * 1.4 + fill_ratio * 2.2)
        + random.gauss(0.0, 0.01)
    )
    median_gray = min(1.0, max(0.0, median_gray))

    return [
        round(intensity_main, 4),
        round(light_count, 4),
        round(ambient_intensity, 4),
        round(main_height, 4),
        round(main_angle, 4),
        round(fill_ratio, 4),
        round(median_gray, 4),
    ]


def write_csv(num_examples=5000):
    output_path = Path(__file__).resolve().parents[1] / "data" / "lighting_regression" / "simulated_lighting_samples.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES)
        for _ in range(num_examples):
            writer.writerow(simulate_row())

    print(output_path)


if __name__ == "__main__":
    write_csv()
