from enum import Enum
from pathlib import Path
from typing import Callable, Literal
from tinygrad import Tensor, TinyJit, nn
from tinygrad.device import Device
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from export_model import export_model
import json
import time

import argparse
import math


class SamplingMod(Enum):
    BILINEAR = 0
    NEAREST = 1


def geometric_transform(
    X: Tensor,
    angle_deg: Tensor,
    scale: Tensor,
    shift_x: Tensor,
    shift_y: Tensor,
    sampling: SamplingMod,
) -> Tensor:
    B, C, H, W = X.shape

    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = Tensor.cos(angle), Tensor.sin(angle)
    R11, R12, T13 = cos_a * scale, -sin_a * scale, shift_x
    R21, R22, T23 = sin_a * scale, cos_a * scale, shift_y
    row1 = Tensor.cat(
        R11.reshape(B, 1), R12.reshape(B, 1), T13.reshape(B, 1), dim=1
    ).reshape(B, 1, 3)
    row2 = Tensor.cat(
        R21.reshape(B, 1), R22.reshape(B, 1), T23.reshape(B, 1), dim=1
    ).reshape(B, 1, 3)
    row3 = Tensor([[0.0, 0.0, 1.0]]).expand(B, 1, 3)
    affine_matrix = Tensor.cat(row1, row2, row3, dim=1)

    x_idx, y_idx = Tensor.arange(W).float(), Tensor.arange(H).float()
    grid_y, grid_x = (
        y_idx.reshape(-1, 1).expand(H, W),
        x_idx.reshape(1, -1).expand(H, W),
    )
    coords = Tensor.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords_homo = (
        Tensor.cat(coords, Tensor.ones(H * W, 1), dim=1)
        .reshape(1, H * W, 3)
        .expand(B, H * W, 3)
    )
    transformed_coords = coords_homo.matmul(affine_matrix.permute(0, 2, 1))

    match sampling:
        case SamplingMod.NEAREST:
            x_idx = transformed_coords[:, :, 0].round().clip(0, W - 1).int()
            y_idx = transformed_coords[:, :, 1].round().clip(0, H - 1).int()
            return (
                X.reshape(B, C * H * W).gather(1, y_idx * W + x_idx).reshape(B, C, H, W)
            )

        case SamplingMod.BILINEAR:
            x_prime, y_prime = transformed_coords[:, :, 0], transformed_coords[:, :, 1]
            x0, y0 = x_prime.floor().int(), y_prime.floor().int()
            dx, dy = x_prime - x0.float(), y_prime - y0.float()

            x1, y1 = x0 + 1, y0 + 1
            x0, y0 = x0.clip(0, W - 1), y0.clip(0, H - 1)
            x1, y1 = x1.clip(0, W - 1), y1.clip(0, H - 1)

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            X_flat = X.reshape(B, C * H * W)
            v00 = X_flat.gather(1, y0 * W + x0)
            v10 = X_flat.gather(1, y0 * W + x1)
            v01 = X_flat.gather(1, y1 * W + x0)
            v11 = X_flat.gather(1, y1 * W + x1)

            return ((w00 * v00) + (w10 * v10) + (w01 * v01) + (w11 * v11)).reshape(
                B, C, H, W
            )


def normalize(X: Tensor) -> Tensor:
    return X * 2 / 255 - 1


class MLPModel:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            lambda x: x.flatten(1),
            nn.Linear(784, 512),
            Tensor.silu,
            nn.Linear(512, 512),
            Tensor.silu,
            nn.Linear(512, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class CNNModel:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.silu,
            nn.Conv2d(32, 32, 5),
            Tensor.silu,
            nn.BatchNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.silu,
            nn.Conv2d(64, 64, 3),
            Tensor.silu,
            nn.BatchNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def train(
    model_type: Literal["mlp", "cnn"],
    save_model: bool,
    B: int,
    LR: float,
    LR_DECAY: float,
    PATIENCE: int,
    STEPS: int,
    ANGLE: float,
    SCALE: float,
    SHIFT: float,
    SAMPLING: int,
) -> None:
    SAMPLING = SamplingMod(SAMPLING)

    model_name = model_type
    dir_name = Path(__file__).parent.parent / "public" / model_name
    if save_model:
        dir_name.mkdir(exist_ok=True)

    model = MLPModel() if model_type == "mlp" else CNNModel()

    X_train, Y_train, X_test, Y_test = mnist()
    opt = nn.optim.Muon(nn.state.get_parameters(model), lr=LR)

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        samples = Tensor.randint(B, high=int(X_train.shape[0]))
        angle_deg = Tensor.rand(B) * 2 * ANGLE - ANGLE
        scale = 1.0 + (Tensor.rand(B) * 2 * SCALE - SCALE)
        shift_x = Tensor.rand(B) * 2 * SHIFT - SHIFT
        shift_y = Tensor.rand(B) * 2 * SHIFT - SHIFT

        opt.zero_grad()
        input = normalize(
            geometric_transform(
                X_train[samples], angle_deg, scale, shift_x, shift_y, SAMPLING
            )
        )
        loss = model(input).sparse_categorical_crossentropy(Y_train[samples]).backward()
        return loss.realize(*opt.schedule_step())

    @TinyJit
    def get_test_acc() -> Tensor:
        return (model(normalize(X_test)).argmax(axis=1) == Y_test).mean() * 100

    test_acc, best_acc, best_since = float("nan"), 0, 0
    t0 = time.time()
    for i in (t := trange(STEPS)):
        loss = train_step()

        if (i % 10 == 9) and (test_acc := get_test_acc().item()) > best_acc:
            best_since = 0
            best_acc = test_acc
            if save_model:
                state_dict = get_state_dict(model)
                safe_save(state_dict, dir_name / f"{model_name}.safetensors")
            continue

        if (best_since := best_since + 1) % PATIENCE == PATIENCE - 1:
            best_since = 0
            opt.lr *= LR_DECAY
            if save_model:
                state_dict = safe_load(dir_name / f"{model_name}.safetensors")
                load_state_dict(model, state_dict)

        t.set_description(
            f"lr: {opt.lr.item():2.2e}  loss: {loss.item():2.2f}  accuracy: {best_acc:2.2f}%"
        )

    t1 = time.time()
    elapsed = t1 - t0

    if save_model:
        Device.DEFAULT = "WEBGPU"
        model = MLPModel() if model_type == "mlp" else CNNModel()
        state_dict = safe_load(dir_name / f"{model_name}.safetensors")
        load_state_dict(model, state_dict)
        input = Tensor.randn(1, 1, 28, 28)
        prg, *_, state = export_model(
            model, Device.DEFAULT.lower(), input, model_name=model_name
        )
        safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")
        with open(dir_name / f"{model_name}.js", "w") as text_file:
            text_file.write(prg)
        print(f"Model exported to {dir_name}")

    else:
        # --- compute basic classification metrics on test set and print them ---
        # Use the best model if we saved and reloaded above, otherwise use current model
        preds = model(normalize(X_test)).argmax(axis=1)
        truths = Y_test

        # accuracy in percent
        accuracy = (preds == truths).mean().item() * 100

        # compute per-class precision/recall/f1 and macro averages (basic)
        num_classes = 10
        precisions = []
        recalls = []
        f1s = []

        for c in range(num_classes):
            tp = ((preds == c) & (truths == c)).sum().item()
            pp = (preds == c).sum().item()
            ap = (truths == c).sum().item()

            prec = (tp / pp) if pp > 0 else 0.0
            rec = (tp / ap) if ap > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        precision_macro = sum(precisions) / num_classes
        recall_macro = sum(recalls) / num_classes
        f1_macro = sum(f1s) / num_classes

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
            "time": float(elapsed),
        }

        # Machine-parseable line (JSON) for external tools like tune.py
        print("FINAL_METRICS: " + json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP or CNN model on MNIST")
    parser.add_argument("-s", "--save", action="store_true", help="Save model outputs")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        required=True,
        help="Select which model to train",
    )
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Patience for LR decay"
    )
    parser.add_argument("--steps", type=int, default=70, help="Training steps")
    parser.add_argument(
        "--angle", type=float, default=15, help="Max rotation angle for augmentation"
    )
    parser.add_argument(
        "--scale", type=float, default=0.1, help="Max scale jitter for augmentation"
    )
    parser.add_argument(
        "--shift", type=float, default=0.1, help="Max shift for augmentation"
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=SamplingMod.NEAREST.value,
        help="Sampling mode (0 bilinear, 1 nearest)",
    )
    args = parser.parse_args()

    model = args.model

    print()
    print(f"Training {model.upper()} model")
    print(f"Model saving is {'enabled' if args.save else 'disabled'}")
    print()

    train(
        model_type=args.model,
        save_model=args.save,
        B=args.batch,
        LR=args.lr,
        LR_DECAY=args.lr_decay,
        PATIENCE=args.patience,
        STEPS=args.steps,
        ANGLE=args.angle,
        SCALE=args.scale,
        SHIFT=args.shift,
        SAMPLING=args.sampling,
    )

    print()
    print("Training complete")
    print()
