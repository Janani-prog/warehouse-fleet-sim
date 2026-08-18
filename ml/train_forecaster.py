"""Train the anomaly LSTM classifier, calibrate it on a held-out validation
split, choose an operating threshold, and report precision/recall on a
separately-seeded held-out test set (never touched during training or
calibration).

    python -m ml.train_forecaster --train data/datasets/forecast_train.parquet \
        --test data/datasets/forecast_test.parquet --out data/models/forecaster \
        --epochs 15 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

from ml.calibration import apply_temperature, choose_threshold_for_precision, fit_temperature
from ml.forecast_features import FEATURE_DIM
from ml.forecaster_model import AnomalyLSTM
from ml.generate_forecast_dataset import WINDOW

HEADS = ["congestion", "collision"]
# See CLAUDE.md's M4 note for the full story: the model turns out to be
# extremely confident once a trend is real (congestion is close to a
# deterministic function of the current active_orders trajectory), so its
# calibrated probability climbs through the achievable precision/recall
# range over a very narrow probability band rather than gradually across
# [0, 1]. 0.9 sits in that band where precision is already high (~0.9+)
# but recall stays ~0.98 - a better operating point than either extreme
# (threshold~0 predicts positive on ~75% of ticks; threshold in the
# isotonic version's old 0.33+ plateau only fired once the event had
# already started). Chosen by inspecting the precision-recall curve
# directly, not guessed.
TARGET_PRECISION = 0.9


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    X = np.stack(df["window"].to_numpy()).reshape(-1, WINDOW, FEATURE_DIM).astype(np.float32)
    y_congestion = df["label_congestion"].to_numpy(dtype=np.float32)
    y_collision = df["label_collision"].to_numpy(dtype=np.float32)
    return X, y_congestion, y_collision


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=str, default="data/datasets/forecast_train.parquet")
    parser.add_argument("--test", type=str, default="data/datasets/forecast_test.parquet")
    parser.add_argument("--out", type=str, default="data/models/forecaster")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X, y_cong, y_coll = load_dataset(args.train)
    y = np.stack([y_cong, y_coll], axis=1)

    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    mean = X[train_idx].reshape(-1, FEATURE_DIM).mean(axis=0)
    std = X[train_idx].reshape(-1, FEATURE_DIM).std(axis=0) + 1e-6

    X_train = torch.tensor(normalize(X[train_idx], mean, std))
    y_train = torch.tensor(y[train_idx])
    X_val = torch.tensor(normalize(X[val_idx], mean, std))
    y_val = torch.tensor(y[val_idx])

    pos_weight = torch.tensor(
        [(len(y_train) - y_train[:, i].sum()) / max(y_train[:, i].sum(), 1) for i in range(2)]
    )

    model = AnomalyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_train = len(X_train)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.tensor(rng.permutation(n_train))
        total_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val)).numpy()
        val_auc_proxy = [
            f1_score(y_val[:, i].numpy(), (val_probs[:, i] > 0.5).astype(float)) for i in range(2)
        ]
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss / n_train:.4f}  "
            f"val_f1(congestion)={val_auc_proxy[0]:.4f}  val_f1(collision)={val_auc_proxy[1]:.4f}"
        )

    # --- calibration (temperature scaling) + threshold selection on the
    # validation split ---
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).numpy()

    temperatures, thresholds = {}, {}
    for i, head in enumerate(HEADS):
        temperatures[head] = fit_temperature(val_logits[:, i], y_val[:, i].numpy())
        calibrated = apply_temperature(val_logits[:, i], temperatures[head])
        thresholds[head] = choose_threshold_for_precision(calibrated, y_val[:, i].numpy(), TARGET_PRECISION)

    # --- final report on the held-out, never-touched test set ---
    X_test, y_test_cong, y_test_coll = load_dataset(args.test)
    y_test = np.stack([y_test_cong, y_test_coll], axis=1)
    X_test_t = torch.tensor(normalize(X_test, mean, std))
    with torch.no_grad():
        test_logits = model(X_test_t).numpy()

    report = {"target_precision": TARGET_PRECISION, "temperatures": temperatures, "heads": {}}
    for i, head in enumerate(HEADS):
        calibrated = apply_temperature(test_logits[:, i], temperatures[head])
        pred = (calibrated >= thresholds[head]).astype(float)
        labels = y_test[:, i]
        report["heads"][head] = {
            "threshold": thresholds[head],
            "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred, zero_division=0),
            "f1": f1_score(labels, pred, zero_division=0),
            "positive_rate_actual": float(labels.mean()),
            "positive_rate_predicted": float(pred.mean()),
            "n_test_examples": len(labels),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "normalization.npz", mean=mean, std=std)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out_dir / "temperatures.json", "w") as f:
        json.dump(temperatures, f, indent=2)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    test_raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
    plot_calibration_curves(test_raw_probs, test_logits, temperatures, y_test, out_dir / "calibration_curves.png")

    print()
    print(json.dumps(report, indent=2))
    print(f"Saved model + calibrators + report to {out_dir}/")


def plot_calibration_curves(
    raw_probs: np.ndarray, logits: np.ndarray, temperatures: dict, y_test: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, head in enumerate(HEADS):
        ax = axes[i]
        raw = raw_probs[:, i]
        calibrated = apply_temperature(logits[:, i], temperatures[head])
        labels = y_test[:, i]

        for probs, style, label in [(raw, "o--", "raw"), (calibrated, "s-", "calibrated")]:
            bins = np.linspace(0, 1, 11)
            bin_ids = np.digitize(probs, bins) - 1
            bin_ids = np.clip(bin_ids, 0, 9)
            observed = [labels[bin_ids == b].mean() if (bin_ids == b).any() else np.nan for b in range(10)]
            predicted_mean = [probs[bin_ids == b].mean() if (bin_ids == b).any() else np.nan for b in range(10)]
            ax.plot(predicted_mean, observed, style, label=label)

        ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="perfect calibration")
        ax.set_title(f"{head} calibration")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)


if __name__ == "__main__":
    main()
