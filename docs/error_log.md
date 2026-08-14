# MiniTrainBench 错误记录与诊断手册

本文只记录实际遇到或在本仓库测试中验证过的问题。没有真实发生过的 NCCL hang、自动
弹性恢复、多机故障，不写成事故，也不把文档化处理方式说成已经实现。

## 记录格式

每条问题按下面六项记录：

1. **现象**：用户看到了什么错误，影响了什么实验。
2. **诊断**：收集了哪些日志、环境和进程信息。
3. **根因**：问题属于 launcher、容器、模型、通信、checkpoint 还是报告层。
4. **修复**：代码、镜像、脚本或实验流程做了什么调整。
5. **验证**：用什么命令或结果证明修复有效。
6. **面试表达**：如何用事实、影响和边界讲清楚。

## 已验证问题

### E001：官方 CUDA 镜像拉取 EOF 或长时间卡住

- **现象**：拉取大体积 CUDA/NGC 镜像时出现 `EOF`、blob 下载失败或长时间无进度，正式
  实验无法开始。
- **诊断**：记录镜像 tag/digest、Docker daemon 日志、磁盘空间和网络重试结果；没有把
  私有代理地址、凭据或宿主机标识写入仓库。
- **根因**：镜像层大，网络链路或 registry 连接不稳定；这不是 PyTorch 训练代码错误。
- **修复**：锁定官方镜像 digest，预先执行镜像拉取和 `docker image inspect`，正式实验
  只接受通过 preflight 的镜像。失败时保留 compatibility 结果，不用另一套环境伪造新结果。
- **验证**：镜像可启动，label 中的 base digest 与结果 provenance 一致。
- **面试表达**：我把“环境不可复现”和“训练失败”分开处理，先固定 digest 和 preflight，
  再生成正式性能证据。

### E002：Python 3.12 的 PEP 668 阻止 editable install

- **现象**：在容器系统 Python 中执行 `pip install -e .`，提示 externally-managed-environment。
- **诊断**：确认 Python 版本、pip 错误和容器是否使用系统环境；没有直接修改宿主机 Python。
- **根因**：发行版通过 PEP 668 保护系统 Python，禁止未经声明的全局 pip 写入。
- **修复**：GPU 容器使用临时虚拟环境，或显式设置容器内的 `PIP_BREAK_SYSTEM_PACKAGES=1`；
  CI 仍使用隔离的 Python 安装方式。
- **验证**：editable package 能被 import，CLI 和 pytest 正常运行。
- **面试表达**：依赖安装是实验协议的一部分，不能把系统 Python 的偶然状态当作 benchmark
  环境；我让安装策略显式化并写进容器/脚本。

### E003：DeepSpeed 在 runtime-only 镜像中探测 CUDA_HOME/nvcc 失败

- **现象**：DeepSpeed import 或 engine 初始化阶段尝试编译 CUDA op，提示 `CUDA_HOME`
  或 `nvcc` 缺失。
- **诊断**：检查 `torch.version.cuda`、`CUDA_HOME`、`nvcc --version` 和 DeepSpeed build
  日志，区分“PyTorch 能运行 CUDA”和“镜像能编译 CUDA 扩展”。
- **根因**：runtime 镜像通常有 CUDA runtime，但不一定包含 toolkit、nvcc 和编译头文件。
  DeepSpeed 某些路径默认探测或构建 fused op。
- **修复**：ZeRO adapter 使用 `DS_BUILD_OPS=0`，关闭不必要的扩展构建，并将 CUDA 检测
  配置显式化；不把 DeepSpeed adapter 混入核心 DDP/FSDP Runtime。
- **验证**：ZeRO 配置可以构建，缺少可选编译能力时返回清晰错误；已有可运行结果保留
  adapter 的版本和环境字段。
- **面试表达**：我区分 runtime image 和 devel image 的能力边界，并通过 adapter contract
  保证可选依赖失败不会污染核心训练路径。

### E004：ZeRO 结果缺少通用汇总字段导致报告 KeyError

- **现象**：通用 repeat/report 代码访问 `grad_norm_mean` 时，DeepSpeed 结果因字段缺失
  触发 `KeyError`。
- **诊断**：比较 DDP/FSDP 原生 Runtime schema 与 ZeRO adapter 输出，定位到报告层假设了
  核心 Runtime 才提供的字段。
- **根因**：adapter 结果没有遵守报告所需的最小 schema；可选字段没有被当作可选处理。
- **修复**：把核心必填字段和 adapter 可选字段分开，报告使用兼容读取并显示 `-`；同时
  为 ZeRO config builder 和旧 JSON 增加测试。
- **验证**：旧 JSON、DDP/FSDP、新 Runtime JSON 和 ZeRO JSON 都能生成报告，缺失字段不会
  阻止其他结果渲染。
- **面试表达**：性能框架的 schema 兼容性和训练代码同样重要；我修的是 adapter contract，
  不是给单个 JSON 打补丁。

### E005：Megatron mock data 仍触发 dataset helper 编译

- **现象**：即使用 mock data，Megatron 启动阶段仍尝试构建 dataset helper，缺少编译工具
  或 Python build 依赖时失败。
- **诊断**：保存完整启动命令和导入日志，确认失败发生在 dataset 初始化辅助模块，而不
  是模型 forward 或 NCCL collective。
- **根因**：mock data 避免数据下载，但不代表整个数据路径不需要构建 helper；外部框架的
  启动依赖仍然存在。
- **修复**：runner 的 preflight 明确检查 build dependency；兼容性 smoke 失败时保留
  `failed`/`failed_parse` 记录，不把它降级成成功性能结果。
- **验证**：安装构建依赖后，固定版本的 Megatron compatibility smoke 可启动；正式 NGC
  performance matrix 仍要求独占 GPU 和完整 provenance。
- **面试表达**：我没有把“没有真实数据集”误解成“没有数据管线依赖”，而是沿启动链路
  定位到 mock data 仍会触发的 helper。

### E006：local fallback 缺少 Transformer Engine/APEX fused capability

- **现象**：PyTorch fallback 环境可以做基础启动，但不具备 Transformer Engine 或 APEX
  的 fused kernel 能力，不能与 NGC profile 的性能数字直接比较。
- **诊断**：记录 requested/resolved transformer implementation、TE/Apex import 状态和
  fusion flags，区分 correctness smoke 与 performance benchmark。
- **根因**：local fallback 和 NGC + TE 是两种软件栈，kernel profile 不同。
- **修复**：local profile 使用显式 unfused 参数；结果写入环境 profile，正式 Megatron
  结果不允许把 fallback 标成 NGC/TE 性能证据。
- **验证**：fallback 只用于兼容性或功能验证，NGC 结果必须有固定 base digest 和 TE import。
- **面试表达**：我没有只比较命令行 topology，而是把 kernel profile 也视为实验变量。

### E007：NGC banner 污染环境探针 JSON

- **现象**：容器启动时的 NGC banner 或环境输出混入 stdout，调用方直接 `json.loads(stdout)`
  失败。
- **诊断**：保留原始 stdout，发现环境信息前存在非 JSON 文本；确认不是 JSON schema 本身
  的问题。
- **根因**：容器 entrypoint 的欢迎信息和探针协议共用了标准输出。
- **修复**：探针输出带 `MINITRAINBENCH_ENVIRONMENT_JSON=` 前缀，调用方只提取带前缀的
  行；普通 banner 不再被当作结构化结果。
- **验证**：同一探针在 NGC banner 开启时仍能生成合法环境 JSON，并且结果中不出现 banner
  或私密环境变量。
- **面试表达**：这是一个小但典型的工具链问题：结构化输出必须有明确 framing，不能假设
  stdout 永远只有 JSON。

### E008：外部 GPU 进程污染正式性能实验

- **现象**：实验启动时 GPU 空闲，但运行到后续 topology 时出现外部 compute process，
  导致 step time、显存和 NCCL 结果不可比较。
- **诊断**：用 `nvidia-smi` 查询 PID、显存和进程命令，在 trial 开始/结束记录计算进程；
  发现不是当前 runner 启动的进程。
- **根因**：节点没有覆盖整个矩阵的连续独占窗口，外部任务在矩阵中途进入。
- **修复**：正式 runner 增加 GPU occupancy 和 performance validity gate；检测污染时停止
  发布该批结果，不发送 kill 信号，也不使用污染数据更新 README。
- **验证**：被污染的 trial 被标为 invalid 或不发布，已有 compatibility evidence 保持不变。
- **面试表达**：性能数据首先要回答“是否有效”，所以我宁愿少一组结果，也不把共享节点
  上的数字包装成可比较的 benchmark。

### E009：Megatron 正式矩阵没有连续独占时间窗口

- **现象**：TP=1/PP=1 和部分 topology 可以启动，但矩阵中途被外部 8 卡任务占用，无法
  完成全部 15 个 repeat trial。
- **诊断**：记录每个 topology 的状态、启动前后的 compute process、失败时的 GPU 快照和
  `post_run_gpu_compute_processes`。
- **根因**：实验资源在矩阵执行期间发生变化，不是 Megatron topology 约束或模型配置本身
  已被证明错误。
- **修复**：使用 staging 目录；只有全部 trial 成功、provenance 完整且 performance
  valid 时才原子替换公开结果。当前仓库保留 compatibility smoke，并明确性能 matrix pending。
- **验证**：公开 `results/megatron_smoke/` 没有被部分结果覆盖，README 没有把 pending 改成
  benchmarked。
- **面试表达**：失败矩阵不等于失败功能；我把资源污染、代码错误和结果发布分别建模，
  不发布不完整的性能结论。

### E010：FSDP sharded state 不能按普通 tensor 直接 digest

- **现象**：对 FSDP/DCP 结果直接调用普通 tensor digest，遇到 sharded tensor、不同 rank
  local shape 或 state dict 类型不匹配。
- **诊断**：分别检查 full state、local shard、metadata 和 rank 顺序，确认模型参数本身
  没有被错误判断为不一致。
- **根因**：FSDP 的分片状态不是单个 rank 上可直接比较的完整 tensor；optimizer state
  也可能有不同分片布局。
- **修复**：对每个 rank 计算稳定的 local shard digest，再按 rank 有序 gather；verify 同时
  比较 metadata、TrainState、scheduler、RNG 和模型/optimizer 分片摘要。
- **验证**：连续训练与中断恢复的 FSDP checkpoint 可以得到 `exact_match=true`；旧 checkpoint
  缺少 RNG 时仍允许功能恢复，但明确标记为非精确恢复。
- **面试表达**：分布式 checkpoint 的正确性不是“rank 0 拿到一个 state dict”这么简单，
  必须先定义 shard identity 和比较协议。

### E011：GPU 可见容器中的 CPU DCP 测试触发 CUDA OOM

- **现象**：命令明确使用 `--device cpu --backend gloo`，但 checkpoint 用例仍在
  `torch.cuda.current_stream()` 处报 CUDA OOM。
- **诊断**：最小复现显示 model 和 process group 都在 CPU，异常来自 PyTorch 2.10 DCP 的
  filesystem overlapping loader；同时宿主 GPU 正被外部任务占用。
- **根因**：测试容器仍能看见 CUDA，DCP 文件写入器根据可用 accelerator 选择 CUDA stream。
  CPU model 并不等于进程完全不会访问 CUDA runtime。
- **修复**：CPU/Gloo 回归容器显式设置 `CUDA_VISIBLE_DEVICES=`；GPU benchmark 不使用这个
  设置，避免混淆两类测试环境。
- **验证**：同一完整 pytest suite 在隐藏 GPU 后全部通过，checkpoint、rank crash 和 exact
  resume 用例均恢复正常。
- **面试表达**：我沿 traceback 区分了 model device 和 checkpoint I/O accelerator，最终
  修的是测试环境隔离，不是掩盖成训练 OOM。

## 通用诊断顺序

遇到新问题时先收集最小证据，再改代码。下面的命令不会写入密钥、代理或 GPU UUID：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
docker ps -a
df -h
git status --short
git rev-parse HEAD
docker image inspect minitrainbench:megatron --format '{{json .Config.Labels}}'
ps -o pid,etime,stat,args -p <PID>
```

按故障层级分类：

| 层级 | 先看什么 | 常见误判 |
| --- | --- | --- |
| launcher | world size、rank、环境变量、退出码 | 把 rank mismatch 当成模型错误 |
| container | image digest、CUDA/PyTorch/NCCL、可写目录 | 把缺 nvcc 当成 GPU 不可用 |
| model/runtime | shape、dtype、loss、显存、grad norm | 把 OOM 当成通信 hang |
| collective | backend、tensor size、split、timeout、rank 是否全部到达 | 只看单 rank 日志就判断 NCCL |
| checkpoint | READY、latest、metadata、DCP/RNG 文件、digest | 看到目录存在就认为 checkpoint 完整 |
| report | schema、repeat、provenance、性能有效性 | 把缺字段修成 0 而不是显示未知 |

## 当前未宣称的能力

- 没有真实 NCCL timeout/hang 事故记录，因此不声称已经完成 hang 自动恢复。
- rank crash 已验证 `SIGKILL -> torchrun 非零退出 -> 手工重启 -> exact resume`，不声称
  支持 TorchElastic 自动重启。
- 没有正式多机性能结果，不把 single-node doctor 或多机模板当成多机 benchmark。
- Megatron 当前公开的是固定版本 compatibility smoke 和读码材料；NGC repeat=3 正式性能
  matrix 没有完整发布前，不写成 `external benchmarked`。

## 新增问题记录模板

```text
编号：E___
日期：
实验命令：
源码 revision：
镜像 digest：
现象：
影响范围：
诊断证据：
根因：
修复：
验证命令与结果：
是否进入正式结果：
面试一句话：
```

完整项目复盘见[训练 Infra 面试指南](interview_guide.md)，源码学习顺序见[学习路线与源码导读](learning_guide.md)。
