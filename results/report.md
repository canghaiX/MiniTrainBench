## Generated Benchmark Results

### Training

| Strategy | GPUs | Precision | Tokens/sec | Step time (ms) | Max memory (MB) |
| --- | ---: | --- | ---: | ---: | ---: |
| ddp | 1 | bf16 | 30362.52 | 16.86 | 481.47 |
| ddp | 2 | bf16 | 38857.41 | 26.35 | 567.13 |
| ddp | 4 | bf16 | 85459.30 | 23.96 | 615.19 |
| fsdp | 1 | bf16 | 15478.27 | 33.08 | 479.77 |
| fsdp | 2 | bf16 | 29016.16 | 35.29 | 274.86 |
| fsdp | 4 | bf16 | 16230.08 | 126.19 | 209.60 |

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
