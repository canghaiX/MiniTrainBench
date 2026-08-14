# 锁定环境 GPU 重跑故障复盘

## 背景

本轮目标是用公开、锁定、可追溯的容器重跑全部 GPU 证据，并补 Megatron-LM 8 卡矩阵。
复盘只记录实际发生的问题。NCCL timeout 和 rank hang 虽然有诊断文档，但本轮没有真实
发生，因此不把它们包装成事故。

## 影响概览

| 事件 | 直接影响 | 最终处理 |
| --- | --- | --- |
| 官方镜像拉取 EOF / 限速 | 正式重跑无法按计划开始 | digest 锁定、pull preflight、保留失败边界 |
| Python 3.12 PEP 668 | editable install 被系统 Python 拒绝 | 显式允许容器内 pip 安装 |
| DeepSpeed 探测 `CUDA_HOME/nvcc` | runtime-only 镜像中 import 失败 | 禁止 import 阶段 CUDA toolkit 探测 |
| ZeRO repeat 汇总 `KeyError` | 通用汇总无法读取 adapter 结果 | adapter-specific 可选指标契约 |
| Megatron mock data helper 依赖缺失 | `pretrain_gpt.py` 在初始化阶段失败 | 安装编译工具和固定 `pybind11` |
| Megatron fallback 缺 fused extensions | local Transformer 构建/forward 失败 | 显式选择纯 PyTorch kernel 路径 |
| GPU 被外部任务占用 | Megatron 数值不可作为性能结论 | 兼容性/性能证据分级并隐藏无效指标 |

## 1. 官方镜像拉取中断

**现象**：拉取锁定的公开 PyTorch/NGC 大镜像时出现 EOF；NGC CDN 在部分大层下载后降到
不可接受的速度。重复执行无法在合理时间内完成 NGC 环境准备。

**定位**：manifest 和 amd64 digest 均可解析，失败发生在 blob 传输阶段，不是 tag、架构或
登录状态错误。直接下载和容器工具表现一致，说明瓶颈位于网络传输链路。

**处理**：`1fc930c` 引入锁定 base digest、镜像 revision 和 provenance gate；`0ed4297`
增加显式的官方 PyTorch fallback，只用于兼容性验证。没有沿用私有 registry 生成公开结果，
也没有伪造 NGC 实测。

**预防**：正式矩阵前先做 digest、磁盘、8 卡可见性和小层 pull preflight；大镜像提前缓存；
结果 manifest 必须记录实际 base digest，而不是只记录 tag。

## 2. Python 3.12 与 PEP 668

**现象**：新基础镜像中的系统 Python 将环境标记为 externally managed，`pip install -e .`
直接失败。

**定位**：错误发生在包装策略而非项目依赖解析；容器是一次性隔离环境，不会污染宿主机
Python。

**处理**：`d1501ae` 在容器中设置 `PIP_BREAK_SYSTEM_PACKAGES=1`，宿主机仍不安装
PyTorch 或测试依赖。

**预防**：镜像升级检查 Python minor version、PEP 668、editable install 和 CLI import，
而不是只验证 `import torch`。

## 3. DeepSpeed 在 runtime-only CUDA 镜像中导入失败

**现象**：DeepSpeed 在 import 阶段探测 `CUDA_HOME` 和 `nvcc`，但公开基础镜像是 runtime
而不是 devel，导致 ZeRO runner 启动失败。

**定位**：项目使用 `DS_BUILD_OPS=0`，并不需要现场编译 fused optimizer；失败来自不必要的
toolkit 探测。

**处理**：`7151553` 设置 `DS_IGNORE_CUDA_DETECTION=1`，保留 PyTorch 自带 CUDA runtime，
不为了 benchmark adapter 引入完整编译工具链。

**预防**：可选 adapter 增加 import smoke，分别验证 runtime 和 devel 镜像假设。

## 4. ZeRO repeat 汇总契约不一致

**现象**：通用 repeat 汇总假设每个训练结果都有 `grad_norm_mean`，DeepSpeed adapter 未输出
该字段，矩阵结束时触发 `KeyError`。

**定位**：训练本身成功，失败发生在结果归一化层。根因是核心 Runtime 指标被错误当成所有
adapter 的必选 schema。

**处理**：`048ca3e` 将 adapter-specific 指标改为可选，并补缺失字段单测。

**预防**：结果 schema 区分 required core metrics 和 optional adapter metrics；每个 adapter
必须通过 contract test 后才能进入矩阵脚本。

## 5. Megatron mock data 仍需要本地编译

**现象**：虽然使用 `--mock-data`，Megatron 仍会编译 dataset index helper。第一次缺 `make`，
第二次缺 `pybind11` 头文件，且只读源码挂载无法写入生成的 `.so`。

**定位**：`pretrain_gpt.py` 已完成 process group 初始化，失败位于
`megatron/core/datasets` 的 helper build，不是 NCCL 或模型配置问题。

**处理**：`beadf4d` 安装 `build-essential` 并允许外部工作树写入被 Git 忽略的 build artifact；
`5297efe` 固定 `pybind11==2.13.6`。外部源码仍固定在 `core_v0.18.2` commit，不复制进项目。

**预防**：Megatron image preflight 增加 dataset helper compile/import，而不是只检查
`megatron.core` wheel 版本。`9047521` 还将动态 OCI revision/build date 移到依赖层之后，
避免每次源码提交都让 Docker 重新解析整套训练依赖。

## 6. NGC 与纯 PyTorch fallback 的能力差异

**现象**：fallback 中缺少 Transformer Engine/APEX 自定义扩展，依次暴露 RoPE fusion、
persistent LayerNorm、gradient accumulation fusion 和 masked softmax fusion 错误；TP 还要求
`CUDA_DEVICE_MAX_CONNECTIONS=1`，而 torch LayerNorm 不支持 sequence parallel。

**定位**：这些错误都发生在明确的可选 fused kernel 或环境约束处。它们不能通过安装普通
Python wheel 等价解决。

**处理**：`cf62559`、`46db13c`、`cd3b3cb`、`968dc1f` 显式选择 local unfused path；
`13f7030` 记录 TP kernel launch ordering；`fcf70a0` 让 sequence parallel 按环境能力解析，
NGC 自动启用，fallback 明确关闭。

**预防**：环境 profile 必须进入 JSON。不同 profile 的结果不直接横向比较，fused/unfused
配置也不能混进同一性能表。

## 7. GPU 并发污染

**现象**：正式 Megatron 矩阵准备期间，8 张卡出现外部计算进程，每卡已有稳定显存占用。
短 smoke 可以验证启动链路，但吞吐、显存和 rank straggler 都不再满足独占实验条件。

**处理**：`913d6bc` 增加 `formal` 与 `compatibility` 两类证据。formal 模式检测到已有计算
进程就拒绝启动；compatibility 模式记录失效原因，保留原始 JSON，但 Markdown 隐藏性能值。
本轮五组 TP/PP/DP 拓扑全部跑通，只标记为 compatibility smoke。

**预防**：正式实验统一执行 GPU occupancy preflight；`performance_valid` 成为报告和能力矩阵
的硬门禁，而不是靠人工备注。

## 结论

这轮最重要的产出不是多出一组数字，而是把“能运行”“可比较”“可作为正式证据”拆成不同
状态。正式结果需要锁定源码、锁定镜像、完整 provenance、独占 GPU 和完整 repeat 协议；
任何条件不满足时保留结构化诊断，但不升级能力声明。
