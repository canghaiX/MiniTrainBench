from __future__ import annotations

import argparse

from .communication import communication_benchmark
from .deepspeed_benchmark import deepspeed_benchmark
from .profiler import profile_training
from .report import write_report
from .training import train
from .verification import verify_checkpoints


def _add_common_distributed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    _add_common_model_arguments(parser)
    parser.add_argument(
        "--gradient-sync-mode",
        choices=["auto", "every", "last"],
        default="auto",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)


def _validate_training_shape(args: argparse.Namespace) -> None:
    if (
        args.grad_accum_steps < 1
        or args.steps < 1
        or args.repeat < 1
        or args.warmup_steps < 0
    ):
        raise SystemExit(
            "steps、repeat 和 grad-accum-steps 必须为正数；warmup-steps 不能为负数"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minitrainbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="运行 DDP 或 FSDP 训练 benchmark")
    _add_common_distributed_arguments(train_parser)
    _add_training_arguments(train_parser)
    train_parser.add_argument("--checkpoint-dir", default=None)
    train_parser.add_argument("--save-every", type=int, default=0)
    train_parser.add_argument("--keep-last", type=int, default=3)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--output", default=None)

    profile_parser = subparsers.add_parser(
        "profile",
        help="运行 PyTorch Profiler 并导出 Chrome trace",
    )
    _add_common_distributed_arguments(profile_parser)
    profile_parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    profile_parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    _add_common_model_arguments(profile_parser)
    profile_parser.add_argument(
        "--gradient-sync-mode",
        choices=["auto", "every", "last"],
        default="auto",
    )
    profile_parser.add_argument("--trace-dir", required=True)
    profile_parser.add_argument("--profile-wait", type=int, default=1)
    profile_parser.add_argument("--profile-warmup", type=int, default=1)
    profile_parser.add_argument("--profile-active", type=int, default=3)
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.add_argument("--with-stack", action="store_true")
    profile_parser.add_argument("--output", default=None)

    deepspeed_parser = subparsers.add_parser(
        "deepspeed",
        help="运行可选 DeepSpeed ZeRO-2/ZeRO-3 benchmark",
    )
    _add_common_distributed_arguments(deepspeed_parser)
    deepspeed_parser.add_argument("--zero-stage", type=int, choices=[2, 3], default=2)
    deepspeed_parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    _add_common_model_arguments(deepspeed_parser)
    deepspeed_parser.add_argument("--steps", type=int, default=20)
    deepspeed_parser.add_argument("--warmup-steps", type=int, default=5)
    deepspeed_parser.add_argument("--repeat", type=int, default=1)
    deepspeed_parser.add_argument("--output", default=None)

    comm_parser = subparsers.add_parser("comm", help="运行 collective 通信 benchmark")
    _add_common_distributed_arguments(comm_parser)
    comm_parser.add_argument("--sizes", default="1024,1048576,16777216")
    comm_parser.add_argument("--warmup", type=int, default=10)
    comm_parser.add_argument("--iters", type=int, default=50)
    comm_parser.add_argument("--output", default=None)

    report_parser = subparsers.add_parser("report", help="将 JSON benchmark 结果渲染为 Markdown")
    report_parser.add_argument("--input", nargs="+", required=True)
    report_parser.add_argument("--output", default=None)

    checkpoint_parser = subparsers.add_parser(
        "checkpoint",
        help="检查 checkpoint 的可恢复性和一致性",
    )
    checkpoint_subparsers = checkpoint_parser.add_subparsers(
        dest="checkpoint_command",
        required=True,
    )
    verify_parser = checkpoint_subparsers.add_parser(
        "verify",
        help="比较两份同配置 checkpoint 是否精确一致",
    )
    _add_common_distributed_arguments(verify_parser)
    verify_parser.add_argument("--left", required=True)
    verify_parser.add_argument("--right", required=True)
    verify_parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _validate_training_shape(args)
        if args.save_every < 0 or args.keep_last < 0:
            raise SystemExit("save-every 和 keep-last 不能为负数")
        if args.resume and not args.checkpoint_dir:
            raise SystemExit("--resume 必须与 --checkpoint-dir 一起使用")
        if args.save_every > 0 and not args.checkpoint_dir:
            raise SystemExit("--save-every 大于 0 时必须提供 --checkpoint-dir")
        if args.repeat > 1 and (
            args.resume or args.checkpoint_dir or args.save_every > 0
        ):
            raise SystemExit(
                "--repeat 大于 1 时表示独立 benchmark trial，不能同时使用 "
                "--checkpoint-dir、--save-every 或 --resume"
            )
        try:
            train(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "profile":
        if (
            args.grad_accum_steps < 1
            or args.profile_wait < 0
            or args.profile_warmup < 0
            or args.profile_active < 1
        ):
            raise SystemExit(
                "grad-accum-steps 和 profile-active 必须为正数；"
                "profile-wait 和 profile-warmup 不能为负数"
            )
        try:
            profile_training(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "deepspeed":
        _validate_training_shape(args)
        try:
            deepspeed_benchmark(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "comm":
        if args.iters < 1 or args.warmup < 0:
            raise SystemExit("iters 必须为正数，warmup 不能为负数")
        communication_benchmark(args)
    elif args.command == "report":
        print(write_report(args.input, args.output), end="")
    elif args.command == "checkpoint" and args.checkpoint_command == "verify":
        try:
            result = verify_checkpoints(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if result is not None and not result["exact_match"]:
            raise SystemExit("checkpoint 不一致，详见校验 JSON")
