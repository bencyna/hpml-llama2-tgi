# HPML Project - LLama2 - TGI

To run the baseline tests, all cells should be run in order in the Jupyter Notebook. Note that W&B sometimes has weird behaviors where the runtime will have to be restarted. 

**Note on TGI KV cache measurement:** vLLM uses PagedAttention with dynamic memory management and exposes real-time KV cache utilization via its `/metrics` endpoint (e.g., `vllm:gpu_cache_usage_perc`). TGI, on the other hand, pre-allocates GPU memory for the KV cache at startup, so nvidia-smi shows constant memory usage throughout inference (~88% in our tests). The TGI metrics endpoint (port 9000) doesn't expose KV cache utilization metrics, and memory fluctuations are minimal and within nvidia-smi's MB-level granularity. As a result, our `kv_cache.py` timeline numbers appears constant.