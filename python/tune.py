import subprocess
import itertools
import math
import time
import re
import json
from pathlib import Path

TRAIN_SCRIPT = "train_model.py"
MODEL = "cnn"

batch_sizes = [512, 1024, 2048]
learning_rates = [0.01, 0.02, 0.05]
steps = [80, 100, 150]
learning_rate_decays = [0.8, 0.9]

log_file = Path(__file__).parent.parent / f"HYPERPARAMETERS-{MODEL.upper()}.md"


def run_training(
    batch: int, lr: float, steps: int, learning_rate_decays: float
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
    ]

    print(f"Running: {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    elapsed = t1 - t0

    output = result.stdout

    # Try to parse machine-readable metrics printed by train_model.py
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
            f"| Run | Batch | LR | Steps | LR Decay | Time | Accuracy | Precision | Recall | F1 |\n"
            f"|-----|------:|----:|------:|---------:|:----:|---------:|----------:|-------:|---:|\n"
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
            f"| b{batch}_lr{lr}_s{steps}_a{lr_decay} | {batch} | {lr} | {steps} | {lr_decay} | {t_str} | {acc:.2f}% | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"
        )


def main():
    append_markdown_header()
    best = (0, None)

    combos = list(
        itertools.product(batch_sizes, learning_rates, steps, learning_rate_decays)
    )
    print(f"Starting hyperparameter sweep: {len(combos)} runs\n")

    for batch, lr, s, ang in combos:
        metrics, elapsed = run_training(batch, lr, s, ang)
        metrics = metrics or {}
        acc = metrics.get("accuracy", float("nan"))
        # prefer the time reported by the training script (FINAL_METRICS) when available
        run_time = metrics.get("time", elapsed)
        print(
            f"b{batch}_lr{lr}_s{s}_a{ang} -> Accuracy={acc:.2f}%  Time={run_time:.1f}s"
        )

        log_markdown_row(batch, lr, s, ang, metrics, run_time)

        if not math.isnan(acc) and acc > best[0]:
            best = (acc, f"b{batch}_lr{lr}_s{s}_a{ang}")

    print(f"\nBEST MODEL: {best[1]} with {best[0]:.2f}% accuracy")

    with open(log_file, "a") as f:
        f.write(
            f"\n---\n\n"
            f"## Best Model\n\n"
            f"**Run:** `{best[1]}`  \n"
            f"**Accuracy:** {best[0]:.2f}%\n\n"
        )


if __name__ == "__main__":
    main()
