# MiniTrainBench 显存压力矩阵

每个档位独立初始化；OOM 和解析失败也是实验结果，不会被静默丢弃。

| 档位 | 策略 | GPU 数 | 模型配置 | Batch/Seq | AC | 目标参数量 | 状态 | Tokens/sec | Step (ms) | 峰值显存 (MB) | 原因 |
| --- | --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| large | ddp | 8 | L24/H1536/A24 | 1/512 | 1 | 731.1M | success | 18842.88 | 217.38 | 16904.04 | - |
| large | fsdp | 8 | L24/H1536/A24 | 1/512 | 1 | 731.1M | success | 23569.80 | 173.78 | 1843.88 | - |
| large | deepspeed_zero2 | 8 | L24/H1536/A24 | 1/512 | 1 | 731.1M | success | 31609.20 | 129.58 | 5472.05 | - |
| large | deepspeed_zero3 | 8 | L24/H1536/A24 | 1/512 | 1 | 731.1M | success | 4517.40 | 906.72 | 3974.11 | - |
| medium | ddp | 8 | L12/H1024/A16 | 1/512 | 1 | 168.5M | success | 56742.31 | 72.19 | 3891.96 | - |
| medium | fsdp | 8 | L12/H1024/A16 | 1/512 | 1 | 168.5M | success | 48806.68 | 83.92 | 500.66 | - |
| medium | deepspeed_zero2 | 8 | L12/H1024/A16 | 1/512 | 1 | 168.5M | success | 62915.36 | 65.10 | 2828.36 | - |
| medium | deepspeed_zero3 | 8 | L12/H1024/A16 | 1/512 | 1 | 168.5M | success | 10311.31 | 397.23 | 1731.63 | - |
| small | ddp | 8 | L6/H512/A8 | 2/256 | 1 | 23.2M | success | 133570.42 | 30.67 | 568.00 | - |
| small | fsdp | 8 | L6/H512/A8 | 2/256 | 1 | 23.2M | success | 90986.08 | 45.02 | 122.82 | - |
| small | deepspeed_zero2 | 8 | L6/H512/A8 | 2/256 | 1 | 23.2M | success | 114606.40 | 35.74 | 2057.14 | - |
| small | deepspeed_zero3 | 8 | L6/H512/A8 | 2/256 | 1 | 23.2M | success | 25174.07 | 162.71 | 1118.02 | - |
| stress | ddp | 8 | L32/H2560/A32 | 1/512 | 1 | 2602.8M | oom | - | - | - | CUDA 内存不足 |
| stress | fsdp | 8 | L32/H2560/A32 | 1/512 | 1 | 2602.8M | success | 16597.60 | 246.78 | 6371.24 | - |
| stress | deepspeed_zero2 | 8 | L32/H2560/A32 | 1/512 | 1 | 2602.8M | success | 18619.17 | 219.99 | 12581.98 | - |
| stress | deepspeed_zero3 | 8 | L32/H2560/A32 | 1/512 | 1 | 2602.8M | success | 4261.41 | 961.18 | 10153.33 | - |

解释边界：small 模型主要观察框架开销；medium/large 才更接近参数、梯度和优化器
状态分片的实际收益。未采集的指标显示为 `-`，不根据模型规模估算显存。

## 自动分析

- `small`：FSDP 相对 DDP 节省 78.4% 峰值显存，吞吐变化 -31.9%。
- `medium`：FSDP 相对 DDP 节省 87.1% 峰值显存，吞吐变化 -14.0%。
- `large`：FSDP 相对 DDP 节省 89.1% 峰值显存，吞吐变化 +25.1%。
- `stress`：DDP OOM，FSDP 成功，峰值显存 6371.24 MB。

## 完整启动命令

### large_ddp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy ddp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 1536 --n-heads 24 --n-layers 24 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/large_ddp_8gpu.json --activation-checkpointing
```

### large_fsdp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy fsdp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 1536 --n-heads 24 --n-layers 24 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/large_fsdp_8gpu.json --activation-checkpointing
```

### large_zero2_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 2 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 1536 --n-heads 24 --n-layers 24 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/large_zero2_8gpu.json --activation-checkpointing
```

### large_zero3_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 3 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 1536 --n-heads 24 --n-layers 24 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/large_zero3_8gpu.json --activation-checkpointing
```

### medium_ddp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy ddp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 16384 --d-model 1024 --n-heads 16 --n-layers 12 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/medium_ddp_8gpu.json --activation-checkpointing
```

### medium_fsdp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy fsdp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 16384 --d-model 1024 --n-heads 16 --n-layers 12 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/medium_fsdp_8gpu.json --activation-checkpointing
```

### medium_zero2_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 2 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 16384 --d-model 1024 --n-heads 16 --n-layers 12 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/medium_zero2_8gpu.json --activation-checkpointing
```

### medium_zero3_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 3 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 16384 --d-model 1024 --n-heads 16 --n-layers 12 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/medium_zero3_8gpu.json --activation-checkpointing
```

### small_ddp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy ddp --precision bf16 --batch-size 2 --seq-length 256 --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/small_ddp_8gpu.json --activation-checkpointing
```

### small_fsdp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy fsdp --precision bf16 --batch-size 2 --seq-length 256 --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/small_fsdp_8gpu.json --activation-checkpointing
```

### small_zero2_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 2 --precision bf16 --batch-size 2 --seq-length 256 --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/small_zero2_8gpu.json --activation-checkpointing
```

### small_zero3_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 3 --precision bf16 --batch-size 2 --seq-length 256 --vocab-size 8192 --d-model 512 --n-heads 8 --n-layers 6 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/small_zero3_8gpu.json --activation-checkpointing
```

### stress_ddp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy ddp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 2560 --n-heads 32 --n-layers 32 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/stress_ddp_8gpu.json --activation-checkpointing
```

### stress_fsdp_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:gpu torchrun --standalone --nproc_per_node=8 -m minitrainbench train --strategy fsdp --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 2560 --n-heads 32 --n-layers 32 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/stress_fsdp_8gpu.json --activation-checkpointing
```

### stress_zero2_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 2 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 2560 --n-heads 32 --n-layers 32 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/stress_zero2_8gpu.json --activation-checkpointing
```

### stress_zero3_8gpu

```bash
docker run --rm --gpus all --ipc=host --network=host -v /data/demo:/workspace -w /workspace minitrainbench:deepspeed torchrun --standalone --nproc_per_node=8 -m minitrainbench deepspeed --zero-stage 3 --precision bf16 --batch-size 1 --seq-length 512 --vocab-size 32768 --d-model 2560 --n-heads 32 --n-layers 32 --steps 3 --warmup-steps 1 --repeat 1 --output /workspace/results/memory_pressure/raw/stress_zero3_8gpu.json --activation-checkpointing
```
