"""CLI: run a scripted scenario and save a replayable telemetry log.

    python -m sim.run --seed 0 --ticks 500 --robots 6 --order-rate 0.4 --out data/runs/demo
"""

from __future__ import annotations

import argparse

from sim.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--order-rate", type=float, default=0.4)
    parser.add_argument("--out", type=str, default="data/runs/demo")
    args = parser.parse_args()

    world = World(seed=args.seed, num_robots=args.robots, order_rate=args.order_rate)
    world.run(args.ticks)
    world.telemetry.save(args.out, list(world.orders.values()), world.manifest())
    print(f"Saved {args.ticks} ticks, {len(world.orders)} orders to {args.out}")


if __name__ == "__main__":
    main()
