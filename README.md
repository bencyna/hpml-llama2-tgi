# Decomposing LLM Inference Optimizations

This repository contains scripts and notebooks for benchmarking vLLM and Text Generation Inference (TGI) against a naive HuggingFace Transformers baseline using [LLaMA-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).  

We evaluate inference-time performance under latency-bound and throughput-bound workloads.

Metrics measured:
- Throughput (tokens/second)
- Time to First Token (TTFT, ms)
- KV-cache / GPU memory utilization
- Cost per million tokens (based on GPU hourly rate)

## Requirements

### System 
* GPU: NVIDIA L4 (24 GB VRAM) or similar
* OS: Linux (tested on Debian 12)
* Docker: ≥ 20.10
* NVIDIA Container Toolkit: required for GPU access in Docker
* nvidia-smi available on host

### Python
```bash
pip install torch transformers requests pandas numpy matplotlib huggingface-hub wandb vllm einops
```

Ensure `HF_TOKEN` is set for gated model access:
```bash
export HF_TOKEN=<your_huggingface_token>
```

### Baseline Model
To run the baseline model and tests, the Jupyter Notebook cells should be run in order. 

## Text Generation Inference (TGI)

TGI is deployed via Docker using the official HuggingFace image.

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf \
  --max-concurrent-requests 128 \
  --max-batch-total-tokens 32768
```
Verify the server:
```bash
curl http://localhost:8080/info
```
### Running Experiments

A) Throughput and TTFT (metrics.py)

Measures throughput, TTFT, and cost per token across batch sizes.
```bash
python3 TGI/metrics.py
```
Outputs:
* Raw results: results/metrics_study/metrics_data.json
* Aggregated results: results/metrics_study/metrics_summary.csv
* Plots: throughput, TTFT, and cost vs. batch size

B) GPU Memory / KV-Cache Profiling (kv_cache.py)

Profiles GPU memory usage during concurrent inference to estimate KV-cache overhead.
```bash
python3 TGI/kv_cache.py
```
Notes:
* vLLM exposes KV-cache utilization directly via its /metrics endpoint.
* TGI pre-allocates KV-cache memory at startup; utilization is inferred indirectly via nvidia-smi.

Outputs:
* Memory time series: results/kv_cache_study/gpu_memory_data.json
* Memory utilization plot: results/kv_cache_study/gpu_memory_timeline.png

---

## Naive HuggingFace Baseline

Baseline experiments are implemented in a Jupyter Notebook.

Instructions:
- Run all notebook cells sequentially from top to bottom
- The notebook measures TTFT, throughput, KV-cache size, and cost
- If Weights & Biases logging becomes unresponsive, restart the runtime and re-run

---

## vLLM

vLLM is run natively using the `vllm` Python package and CLI tools.

### Starting the vLLM Server

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --trust-remote-code \
  --disable-log-requests \
  --max-model-len 4096
```

Verify the server:
```bash
curl http://localhost:8000/health
```

### Running Experiments

A) Throughput Benchmarks (`vllm_benchmarks.py`)

Sweeps across concurrency levels to measure throughput, TTFT, and latency metrics using `vllm bench sweep`.

```bash
cd "Ethan VLLM"
python3 vllm_benchmarks.py
```

Outputs:
* Per-run results: `<OUTPUT_DIR>/<timestamp>/SERVE--*/run=*.json`
* Aggregated summary: `<OUTPUT_DIR>/<timestamp>/summary.csv`
* Metrics include: `request_throughput`, `output_throughput`, `mean_ttft_ms`, `mean_tpot_ms`, `mean_e2el_ms`

B) KV-Cache Profiling (`kv_cache.py`)

Monitors KV-cache utilization in real-time via vLLM's `/metrics` endpoint during high-concurrency inference.

```bash
cd "Ethan VLLM"
python3 kv_cache.py
```

Notes:
* vLLM exposes `vllm:*cache_usage_perc` metrics directly via Prometheus-style `/metrics` endpoint
* The script polls at 100ms intervals and records peak utilization

Outputs:
* Time series data: `kv_cache_study/<task>/kv_cache_data.json`
* Utilization plot: `kv_cache_study/<task>/kv_cache_timeline.png`

---

## Implementation Notes
* All experiments use FP16 precision with KV caching enabled
* Prompts used for evaluation are included in `prompts.txt`
* GPU cost calculations assume $2.93/hour (NVIDIA L4 on-demand)