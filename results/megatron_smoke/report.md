# Megatron-LM 8 卡 Smoke / TP-PP-DP 矩阵

## 当前状态

`not_run`。目标版本固定为 `core_v0.18.2`，目标镜像为
`nvcr.io/nvidia/pytorch:26.01-py3`。当前节点无法从 GitHub 获取该固定 ref，工作区也没有
可校验的外部官方 Megatron-LM 源码，因此没有启动训练，也没有生成 tokens/sec、step time
或显存数字。

这不是 benchmark 失败记录，更不是性能结论。`manifest.json` 的 `records` 保持为空，
README 能力矩阵也不将 Megatron 标为 `benchmarked`。

## 复现

准备好外部官方源码并将 HEAD 切到固定 ref 后运行：

```bash
MEGATRON_DIR=/path/to/Megatron-LM \
MEGATRON_REF=core_v0.18.2 \
MEGATRON_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 \
  scripts/run_megatron_tp_pp_matrix.sh
```

脚本会先校验 ref/commit 和容器内 PyTorch、CUDA、NCCL、Megatron Core 版本，再依次运行
TP/PP=`1/1`、`2/1`、`4/1`、`2/2`、`1/4`。环境探针、训练或日志解析失败都会写入
结构化 record；只有成功且 metadata 完整的记录才能作为实测证据。
