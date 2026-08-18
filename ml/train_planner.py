"""Train the imitation-learned planner on the A*-solved dataset.

    python -m ml.train_planner --dataset data/datasets/astar_train.parquet \
        --out data/models/planner_mlp.pt --epochs 30 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.planner_features import action_index, encode
from ml.planner_model import PlannerMLP
from sim.grid import generate_default_layout


def build_training_examples(df: pd.DataFrame, warehouse) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for path in df["path"]:
        cells = [(int(c[0]), int(c[1])) for c in path]
        goal = cells[-1]
        for pos, nxt in zip(cells, cells[1:]):
            delta = (nxt[0] - pos[0], nxt[1] - pos[1])
            features.append(encode(warehouse, pos, goal))
            labels.append(action_index(delta))
    return np.stack(features), np.array(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/datasets/astar_train.parquet")
    parser.add_argument("--out", type=str, default="data/models/planner_mlp.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    warehouse = generate_default_layout()
    df = pd.read_parquet(args.dataset)
    X, y = build_training_examples(df, warehouse)

    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_val = torch.tensor(X[val_idx])
    y_val = torch.tensor(y[val_idx])

    model = PlannerMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

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
            val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={total_loss / n_train:.4f}  val_acc={val_acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    with torch.no_grad():
        final_val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
    meta = {
        "seed": args.seed,
        "epochs": args.epochs,
        "n_train_examples": int(n_train),
        "n_val_examples": int(n_val),
        "final_val_action_accuracy": final_val_acc,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {out_path} (val action accuracy {final_val_acc:.4f})")


if __name__ == "__main__":
    main()
