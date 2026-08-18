"""Train the learned avoidance policy on the classical-resolver-labeled
dataset.

    python -m ml.train_traffic --dataset data/datasets/traffic_train.parquet \
        --out data/models/traffic_mlp.pt --epochs 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.traffic_model import TrafficMLP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/datasets/traffic_train.parquet")
    parser.add_argument("--out", type=str, default="data/models/traffic_mlp.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.dataset)
    X = np.stack(df["features"].to_numpy())
    y = df["label"].to_numpy(dtype=np.float32)

    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx])
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx])

    model = TrafficMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

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
            val_pred = (torch.sigmoid(model(X_val)) > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss / n_train:.4f}  val_acc={val_acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    with torch.no_grad():
        final_val_acc = ((torch.sigmoid(model(X_val)) > 0.5).float() == y_val).float().mean().item()
    meta = {
        "seed": args.seed,
        "epochs": args.epochs,
        "n_train_examples": int(n_train),
        "n_val_examples": int(n_val),
        "final_val_accuracy": final_val_acc,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {out_path} (val accuracy {final_val_acc:.4f})")


if __name__ == "__main__":
    main()
