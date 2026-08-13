from __future__ import annotations

import argparse

from .communication import communication_benchmark
from .deepspeed_benchmark import deepspeed_benchmark
from .doctor import run_doctor
from .fault_tolerance import fault_tolerance_smoke
from .moe_routing import moe_routing_benchmark
from .profiler import profile_training
from .report import write_report
from .tensor_parallel import (
    tensor_parallel_check,
    tensor_parallel_mlp_check,
    tensor_parallel_sequence_check,
)
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


def _add_runtime_stability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr-scheduler",
        choices=["constant", "cosine"],
        default="constant",
        help="按 optimizer step 推进的学习率调度器",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="全局梯度范数裁剪阈值；0 表示只记录范数、不裁剪",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    _add_common_model_arguments(parser)
    _add_runtime_stability_arguments(parser)
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


def _validate_runtime_stability(args: argparse.Namespace) -> None:
    if args.max_grad_norm < 0:
        raise SystemExit("max-grad-norm 不能为负数；0 表示关闭梯度裁剪")
    if args.lr_warmup_steps < 0 or args.lr_decay_steps < 0:
        raise SystemExit("lr-warmup-steps 和 lr-decay-steps 不能为负数")
    if args.min_learning_rate < 0 or args.learning_rate <= 0:
        raise SystemExit("learning-rate 必须大于 0，min-learning-rate 不能为负数")
    if args.lr_scheduler == "constant" and (
        args.lr_warmup_steps or args.lr_decay_steps or args.min_learning_rate
    ):
        raise SystemExit(
            "constant scheduler 要求 lr-warmup-steps、lr-decay-steps "
            "和 min-learning-rate 均为 0"
        )
    if args.lr_scheduler == "cosine" and args.lr_decay_steps <= args.lr_warmup_steps:
        raise SystemExit(
            "cosine scheduler 要求 lr-decay-steps 大于 lr-warmup-steps"
        )
    if args.min_learning_rate > args.learning_rate:
        raise SystemExit("min-learning-rate 不能大于 learning-rate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minitrainbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="检查多机/NCCL 运行环境")
    _add_common_distributed_arguments(doctor_parser)
    doctor_parser.add_argument("--master-addr", default=None)
    doctor_parser.add_argument("--master-port", type=int, default=None)
    doctor_parser.add_argument("--timeout", type=float, default=2.0)
    doctor_parser.add_argument("--skip-connectivity", action="store_true")
    doctor_parser.add_argument("--expected-world-size", type=int, default=0)
    doctor_parser.add_argument("--expected-gpus", type=int, default=0)
    doctor_parser.add_argument("--output", default=None)

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
    _add_runtime_stability_arguments(profile_parser)
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
    comm_parser.add_argument(
        "--operations",
        default="all_reduce,all_gather,reduce_scatter,all_to_all",
        help="逗号分隔的 collective 列表",
    )
    comm_parser.add_argument(
        "--all-to-all-mode",
        choices=["equal", "uneven", "both"],
        default="both",
        help="all-to-all 的 split 模式",
    )
    comm_parser.add_argument("--warmup", type=int, default=10)
    comm_parser.add_argument("--iters", type=int, default=50)
    comm_parser.add_argument("--output", default=None)

    tp_parser = subparsers.add_parser(
        "tp",
        help="运行 toy tensor parallel 正确性检查",
    )
    tp_subparsers = tp_parser.add_subparsers(dest="tp_command", required=True)
    tp_check_parser = tp_subparsers.add_parser(
        "check",
        help="比较 Column/Row Parallel Linear 与单卡 reference 是否一致",
    )
    _add_common_distributed_arguments(tp_check_parser)
    tp_check_parser.add_argument("--batch-size", type=int, default=2)
    tp_check_parser.add_argument("--seq-length", type=int, default=8)
    tp_check_parser.add_argument("--in-features", type=int, default=16)
    tp_check_parser.add_argument("--out-features", type=int, default=32)
    tp_check_parser.add_argument("--seed", type=int, default=2026)
    tp_check_parser.add_argument("--atol", type=float, default=1e-3)
    tp_check_parser.add_argument("--output", default=None)

    tp_mlp_parser = tp_subparsers.add_parser(
        "mlp",
        help="验证 toy TP MLP 与单卡 reference 是否一致",
    )
    _add_common_distributed_arguments(tp_mlp_parser)
    tp_mlp_parser.add_argument("--batch-size", type=int, default=2)
    tp_mlp_parser.add_argument("--seq-length", type=int, default=8)
    tp_mlp_parser.add_argument("--in-features", type=int, default=16)
    tp_mlp_parser.add_argument("--hidden-features", type=int, default=64)
    tp_mlp_parser.add_argument("--out-features", type=int, default=16)
    tp_mlp_parser.add_argument("--seed", type=int, default=2026)
    tp_mlp_parser.add_argument("--atol", type=float, default=1e-3)
    tp_mlp_parser.add_argument("--output", default=None)

    tp_sequence_parser = tp_subparsers.add_parser(
        "sequence",
        help="验证 toy sequence parallel LayerNorm/Dropout shard 语义",
    )
    _add_common_distributed_arguments(tp_sequence_parser)
    tp_sequence_parser.add_argument("--batch-size", type=int, default=2)
    tp_sequence_parser.add_argument("--seq-length", type=int, default=8)
    tp_sequence_parser.add_argument("--hidden-size", type=int, default=16)
    tp_sequence_parser.add_argument("--dropout", type=float, default=0.1)
    tp_sequence_parser.add_argument("--seed", type=int, default=2026)
    tp_sequence_parser.add_argument("--atol", type=float, default=1e-3)
    tp_sequence_parser.add_argument("--output", default=None)

    moe_parser = subparsers.add_parser("moe", help="运行 toy MoE routing/dispatch 检查")
    moe_subparsers = moe_parser.add_subparsers(dest="moe_command", required=True)
    moe_route_parser = moe_subparsers.add_parser(
        "route",
        help="运行 top-1 routing、capacity 和 all-to-all dispatch/combine demo",
    )
    _add_common_distributed_arguments(moe_route_parser)
    moe_route_parser.add_argument("--tokens-per-rank", type=int, default=64)
    moe_route_parser.add_argument("--hidden-size", type=int, default=32)
    moe_route_parser.add_argument("--num-experts", type=int, default=4)
    moe_route_parser.add_argument("--capacity-factor", type=float, default=1.25)
    moe_route_parser.add_argument("--seed", type=int, default=2026)
    moe_route_parser.add_argument("--output", default=None)

    fault_parser = subparsers.add_parser("fault", help="运行训练稳定性和故障恢复 smoke")
    fault_subparsers = fault_parser.add_subparsers(dest="fault_command", required=True)
    fault_smoke_parser = fault_subparsers.add_parser(
        "smoke",
        help="生成 checkpoint/resume 与故障处理证据",
    )
    _add_common_distributed_arguments(fault_smoke_parser)
    fault_smoke_parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    fault_smoke_parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    _add_common_model_arguments(fault_smoke_parser)
    _add_runtime_stability_arguments(fault_smoke_parser)
    fault_smoke_parser.add_argument(
        "--gradient-sync-mode",
        choices=["auto", "every", "last"],
        default="auto",
    )
    fault_smoke_parser.add_argument("--warmup-steps", type=int, default=0)
    fault_smoke_parser.add_argument("--continuous-steps", type=int, default=3)
    fault_smoke_parser.add_argument("--interrupted-steps", type=int, default=2)
    fault_smoke_parser.add_argument("--resume-steps", type=int, default=1)
    fault_smoke_parser.add_argument("--keep-last", type=int, default=0)
    fault_smoke_parser.add_argument("--checkpoint-dir", default=None)
    fault_smoke_parser.add_argument("--output", default=None)

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
    if args.command == "doctor":
        if args.timeout <= 0 or args.expected_world_size < 0 or args.expected_gpus < 0:
            raise SystemExit("timeout 必须大于 0；expected-* 不能为负数")
        run_doctor(args)
    elif args.command == "train":
        _validate_training_shape(args)
        _validate_runtime_stability(args)
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
            or args.max_grad_norm < 0
        ):
            raise SystemExit(
                "grad-accum-steps 和 profile-active 必须为正数；"
                "profile-wait 和 profile-warmup 不能为负数"
            )
        _validate_runtime_stability(args)
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
        try:
            communication_benchmark(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "tp" and args.tp_command == "check":
        if (
            args.batch_size < 1
            or args.seq_length < 1
            or args.in_features < 1
            or args.out_features < 1
            or args.atol <= 0
        ):
            raise SystemExit(
                "batch-size、seq-length、in-features 和 out-features 必须为正数；"
                "atol 必须大于 0"
            )
        try:
            result = tensor_parallel_check(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if result is not None and result["status"] != "ok":
            raise SystemExit("tensor parallel 正确性检查失败，详见输出 JSON")
    elif args.command == "tp" and args.tp_command == "mlp":
        if (
            args.batch_size < 1
            or args.seq_length < 1
            or args.in_features < 1
            or args.hidden_features < 1
            or args.out_features < 1
            or args.atol <= 0
        ):
            raise SystemExit(
                "batch-size、seq-length、in-features、hidden-features 和 "
                "out-features 必须为正数；atol 必须大于 0"
            )
        try:
            result = tensor_parallel_mlp_check(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if result is not None and result["status"] != "ok":
            raise SystemExit("TP MLP 正确性检查失败，详见输出 JSON")
    elif args.command == "tp" and args.tp_command == "sequence":
        if (
            args.batch_size < 1
            or args.seq_length < 1
            or args.hidden_size < 1
            or args.atol <= 0
            or args.dropout < 0
            or args.dropout >= 1
        ):
            raise SystemExit(
                "batch-size、seq-length 和 hidden-size 必须为正数；"
                "atol 必须大于 0；dropout 必须在 [0, 1) 范围内"
            )
        try:
            result = tensor_parallel_sequence_check(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if result is not None and result["status"] != "ok":
            raise SystemExit("Sequence Parallel 正确性检查失败，详见输出 JSON")
    elif args.command == "moe" and args.moe_command == "route":
        if (
            args.tokens_per_rank < 1
            or args.hidden_size < 1
            or args.num_experts < 1
            or args.capacity_factor <= 0
        ):
            raise SystemExit(
                "tokens-per-rank、hidden-size、num-experts 必须为正数；"
                "capacity-factor 必须大于 0"
            )
        try:
            moe_routing_benchmark(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "fault" and args.fault_command == "smoke":
        if (
            args.grad_accum_steps < 1
            or args.warmup_steps < 0
            or args.continuous_steps < 2
            or args.interrupted_steps < 1
            or args.resume_steps < 1
            or args.interrupted_steps + args.resume_steps != args.continuous_steps
            or args.keep_last < 0
            or args.max_grad_norm < 0
        ):
            raise SystemExit(
                "fault smoke 要求 continuous-steps>=2，interrupted/resume steps 为正，"
                "且 interrupted-steps + resume-steps 等于 continuous-steps；"
                "warmup-steps 和 keep-last 不能为负数"
            )
        _validate_runtime_stability(args)
        try:
            fault_tolerance_smoke(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "report":
        print(write_report(args.input, args.output), end="")
    elif args.command == "checkpoint" and args.checkpoint_command == "verify":
        try:
            result = verify_checkpoints(args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if result is not None and not result["exact_match"]:
            raise SystemExit("checkpoint 不一致，详见校验 JSON")
