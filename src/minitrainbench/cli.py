from __future__ import annotations

import argparse

from .communication import communication_benchmark
from .report import write_report
from .training import train


def _add_common_distributed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minitrainbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="run DDP or FSDP training benchmark")
    _add_common_distributed_arguments(train_parser)
    train_parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    train_parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    train_parser.add_argument("--activation-checkpointing", action="store_true")
    train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--seq-length", type=int, default=256)
    train_parser.add_argument("--vocab-size", type=int, default=16_384)
    train_parser.add_argument("--d-model", type=int, default=512)
    train_parser.add_argument("--n-heads", type=int, default=8)
    train_parser.add_argument("--n-layers", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--steps", type=int, default=20)
    train_parser.add_argument("--warmup-steps", type=int, default=5)
    train_parser.add_argument("--repeat", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--output", default=None)

    comm_parser = subparsers.add_parser("comm", help="run collective communication benchmark")
    _add_common_distributed_arguments(comm_parser)
    comm_parser.add_argument("--sizes", default="1024,1048576,16777216")
    comm_parser.add_argument("--warmup", type=int, default=10)
    comm_parser.add_argument("--iters", type=int, default=50)
    comm_parser.add_argument("--output", default=None)

    report_parser = subparsers.add_parser("report", help="render JSON benchmark results as Markdown")
    report_parser.add_argument("--input", nargs="+", required=True)
    report_parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if args.grad_accum_steps < 1 or args.steps < 1 or args.repeat < 1 or args.warmup_steps < 0:
            raise SystemExit(
                "steps, repeat, and grad-accum-steps must be positive; "
                "warmup cannot be negative"
            )
        train(args)
    elif args.command == "comm":
        if args.iters < 1 or args.warmup < 0:
            raise SystemExit("iters must be positive and warmup cannot be negative")
        communication_benchmark(args)
    elif args.command == "report":
        print(write_report(args.input, args.output), end="")
