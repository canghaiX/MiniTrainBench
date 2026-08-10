## Generated Benchmark Results

### Training

| Strategy | GPUs | Precision | Tokens/sec | Step time (ms) | Max memory (MB) | Scaling efficiency | Memory saving vs DDP | Step delta vs DDP (ms) | Repeats |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ddp | 1 | bf16 | 30362.52 | 16.86 | 481.47 | 100.00% | - | - | 1 |
| ddp | 2 | bf16 | 38857.41 | 26.35 | 567.13 | 63.99% | - | - | 1 |
| ddp | 4 | bf16 | 85459.30 | 23.96 | 615.19 | 70.37% | - | - | 1 |
| fsdp | 1 | bf16 | 15478.27 | 33.08 | 479.77 | 100.00% | 0.35% | 16.22 | 1 |
| fsdp | 2 | bf16 | 29016.16 | 35.29 | 274.86 | 93.73% | 51.54% | 8.94 | 1 |
| fsdp | 4 | bf16 | 16230.08 | 126.19 | 209.60 | 26.21% | 65.93% | 102.22 | 1 |

Scaling efficiency is normalized to each strategy's 1-GPU throughput. FSDP memory saving and step delta are computed against DDP at the same GPU count.

### Communication

| Operation | GPUs | Elements | Latency (ms) | Bandwidth (GB/s) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| all_reduce | 4 | 1024 | 0.054 | 0.076 | ok |
| all_gather | 4 | 1024 | 0.135 | 0.122 | ok |
| reduce_scatter | 4 | 1024 | 0.061 | 0.269 | ok |
| all_reduce | 4 | 1048576 | 0.102 | 41.172 | ok |
| all_gather | 4 | 1048576 | 0.232 | 72.433 | ok |
| reduce_scatter | 4 | 1048576 | 0.129 | 129.768 | ok |
| all_reduce | 4 | 16777216 | 0.648 | 103.577 | ok |
| all_gather | 4 | 16777216 | 6.092 | 44.062 | ok |
| reduce_scatter | 4 | 16777216 | 1.818 | 147.652 | ok |

Small collective sizes are latency-bound; larger tensors expose bandwidth limits. Compare these rows with training step time to estimate communication pressure.
