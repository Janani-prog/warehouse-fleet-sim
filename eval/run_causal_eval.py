"""M9 done-when: a reproducible statistical report from a single command.

    python -m eval.run_causal_eval --n-pairs 30 --seed-start 5000 \
        --ticks 140 --robots 8 --forecaster-model data/models/forecaster \
        --out data/results/causal_eval

Default scenario is a compressed version of M4's congestion-spike scenario
(spike at ticks 30-70 of a 140-tick trial, vs. the full demo's 200-260 of
500) specifically so N pairs is tractable: CPU-only llama3.1:8b inference
is the bottleneck (each triggered tick during a loop-ON trial is one LLM
call), and only loop-ON trials pay that cost - see CLAUDE.md's M9 note for
the wall-clock tradeoffs this was tuned against.
"""

from __future__ import annotations

import argparse
import json

from eval.causal_harness import run_and_report
from ml.scripted_scenarios import BASELINE_RATE, SPIKE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--ticks", type=int, default=140)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--baseline-rate", type=float, default=BASELINE_RATE)
    parser.add_argument("--spike-rate", type=float, default=SPIKE_RATE)
    parser.add_argument("--spike-start", type=int, default=30)
    parser.add_argument("--spike-end", type=int, default=70)
    parser.add_argument("--forecaster-model", type=str, default="data/models/forecaster")
    parser.add_argument("--agent-model", type=str, default="llama3.1:8b")
    parser.add_argument("--out", type=str, default="data/results/causal_eval")
    args = parser.parse_args()

    report = run_and_report(
        n_pairs=args.n_pairs, seed_start=args.seed_start, ticks=args.ticks, robots=args.robots,
        baseline_rate=args.baseline_rate, spike_rate=args.spike_rate,
        spike_start=args.spike_start, spike_end=args.spike_end,
        forecaster_model_dir=args.forecaster_model, out_dir=args.out, agent_model=args.agent_model,
    )

    print(f"Saved {args.n_pairs} paired trials to {args.out}")
    for metric, result in report["metrics"].items():
        print(f"\n{result['interpretation']}")
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
