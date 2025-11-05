import subprocess
import random
import math
import time
import re
import json
from pathlib import Path

TRAIN_SCRIPT = "train_model.py"
MODEL = "cnn"

batch_sizes = [256, 512, 1024]
learning_rates = [0.005, 0.01, 0.02, 0.05]
steps = [50, 100, 150, 200, 500]
learning_rate_decays = [0.85, 0.9, 0.95]
patience_values = [10, 25, 50]

log_file = Path(__file__).parent.parent / f"HYPERPARAMETERS-{MODEL.upper()}.md"


def run_training(
    batch: int, lr: float, steps: int, learning_rate_decays: float, patience: int
) -> tuple[dict[str, float], float]:
    cmd = [
        "python",
        TRAIN_SCRIPT,
        "--model",
        MODEL,
        "--batch",
        str(batch),
        "--lr",
        str(lr),
        "--steps",
        str(steps),
        "--lr_decay",
        str(learning_rate_decays),
        "--patience",
        str(patience),
    ]

    print(f"Running: {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    elapsed = t1 - t0

    output = result.stdout

    metrics = None
    m = re.search(r"FINAL_METRICS:\s*(\{.*\})", output)
    if m:
        try:
            metrics = json.loads(m.group(1))
        except Exception:
            metrics = None

    return metrics, elapsed


def append_markdown_header():
    if not log_file.exists():
        log_file.write_text(
            f"# Hyperparameter Search Results\n\n"
            f"This report summarizes all hyperparameter runs."
            f"\n\n"
            f"## Summary for {MODEL.upper()}\n\n"
            f"| Batch | LR | Steps | LR Decay | Patience | Time | Accuracy | Precision | Recall | F1 |\n"
            f"|------:|---:|------:|---------:|---------:|:----:|---------:|----------:|-------:|---:|\n"
        )


def _format_time(seconds: float) -> str:
    # Format seconds as H:MM:SS or MM:SS if short
    if seconds is None or math.isnan(seconds):
        return "-"
    s = int(seconds)
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    else:
        return f"{mins:02d}:{secs:02d}"


def log_markdown_row(
    batch: int,
    lr: float,
    steps: int,
    lr_decay: float,
    patience: int,
    metrics: dict[str, float],
    run_time: float,
) -> None:
    acc = metrics.get("accuracy", float("nan"))
    prec = metrics.get("precision", float("nan"))
    rec = metrics.get("recall", float("nan"))
    f1 = metrics.get("f1", float("nan"))
    t_str = _format_time(run_time)
    with open(log_file, "a") as f:
        f.write(
            f"| {batch} | {lr} | {steps} | {lr_decay} | {patience} | {t_str} | **{acc:.2f}%** | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"
        )


def main(total_config=15):
    append_markdown_header()

    mandatory_configs = [
        (512, 0.01, 100, 0.9, 25),
        (256, 0.005, 500, 0.9, 50),
        (2048, 0.02, 200, 0.85, 25),
    ]

    def random_unique_configs(total_config):
        configs = set(mandatory_configs)
        while len(configs) < total_config:
            cfg = (
                random.choice(batch_sizes),
                random.choice(learning_rates),
                random.choice(steps),
                random.choice(learning_rate_decays),
                random.choice(patience_values),
            )
            configs.add(cfg)
        return list(configs)

    combos = random_unique_configs(total_config)
    print(f"Starting hyperparameter sweep: {len(combos)} runs\n")

    for batch, lr, s, lr_decay, pat in combos:
        metrics, elapsed = run_training(batch, lr, s, lr_decay, pat)
        metrics = metrics or {}
        acc = metrics.get("accuracy", float("nan"))
        run_time = metrics.get("time", elapsed)
        print(
            f"b{batch}_lr{lr}_s{s}_d{lr_decay}_p{pat} -> Accuracy={acc:.2f}%  Time={run_time:.1f}s"
        )

        log_markdown_row(batch, lr, s, lr_decay, pat, metrics, run_time)


if __name__ == "__main__":
    main()
