# Decomposing LLM Inference Optimizations

## Team Information
- **Members**:
  - Benjamin Cyna (bc3096)
  - Ethan Simpson (ems2373)
  - Michael Carrion (mc5672)
  - Andre Mao (am5994)
  - Kevin Wang (kjw2169)

---

## 1. Problem Statement

Existing comparative studies benchmark vLLM and TGI against each other but rarely against an unoptimized baseline. As a result, it's often difficult to quantify the absolute improvement each framework provides. We address this gap by establishing a minimal HuggingFace Transformers baseline and systematically measuring how each framework's optimizations translate to performance gains.

We evaluate inference-time performance under latency-bound (single-request) and throughput-bound (batched) workloads, targeting four metrics: throughput (tokens/s), time to first token (TTFT), KV-cache utilization, and cost per million tokens.

---

## 2. Model Description

**Model**: Meta's [LLaMA-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

**Framework**: PyTorch with FP16 precision and KV caching enabled

**Inference Configurations**:
- **HuggingFace Transformers Baseline**: Standard `transformers` library with per-request processing. No inter-request batching or memory management optimizations.
- **vLLM**: Inference engine using PagedAttention for paged KV-cache storage with continuous batching, reducing memory fragmentation and enabling efficient reuse.
- **TGI (Text Generation Inference)**: Production server using dynamic batching with FlashAttention and pre-allocated KV-cache budget.

---

## 3. Final Results Summary
We summarize representative results below; full tables are available in the paper.

### Latency-Bound (Single Request)

| Framework | Task       | Throughput (tok/s) | TTFT (ms) | Cost/M ($) |
|-----------|------------|-------------------|-----------|------------|
| Baseline  | QA         | 17.50             | 64.22     | 46.50      |
| TGI       | QA         | 18.05             | 58.71     | 45.09      |
| vLLM      | QA         | 18.07             | 63.16     | 45.04      |

### Throughput-Bound (Batch Size 128)

| Framework | Task       | Throughput (tok/s) | TTFT (ms) | Cost/M ($) |
|-----------|------------|-------------------|-----------|------------|
| Baseline  | QA         | 360.51            | 922.34    | 2.26       |
| TGI       | QA         | 575.78            | 153.31    | 1.41       |
| vLLM      | QA         | 537.60            | 593.00    | 1.51       |
| Baseline  | Reasoning  | 487.72 (batch 64) | 1115.99   | 1.67       |
| TGI       | Reasoning  | 194.48            | 167.22    | 4.18       |
| vLLM      | Reasoning  | 1142.80           | 601.00    | 0.71       |

### Key Findings

| Metric                          | Value                                      |
|---------------------------------|--------------------------------------------|
| Best Throughput                 | 1142.8 tok/s (vLLM, Reasoning, batch 128)  |
| Best TTFT                       | 153.31 ms (TGI, QA, batch 128)             |
| Lowest Cost/M Tokens            | $0.71 (vLLM, Reasoning, batch 128)         |
| Max TTFT Reduction vs Baseline  | 27× (TGI on summarization)                 |
| Max Throughput Gain vs Baseline | 2.3× (vLLM on reasoning)                   |
| Device                          | NVIDIA L4 (24 GB VRAM)                     |

---

## 4. Reproducibility Instructions

### A. Requirements

**System**
- GPU: NVIDIA L4 (24 GB VRAM) or similar
- OS: Linux (tested on Debian 12)
- Docker: ≥ 20.10
- NVIDIA Container Toolkit: required for GPU access in Docker
- `nvidia-smi` available on host

**Python Dependencies**
```bash
pip install torch transformers requests pandas numpy matplotlib huggingface-hub wandb vllm einops
```

**Environment Setup**
```bash
export HF_TOKEN=<your_huggingface_token>
```

---

### B. W&B Dashboard

* https://wandb.ai/mc5672-columbia-university/hpml-tgi-benchmark?nw=nwusermc5672
* https://wandb.ai/am5994-columbia-university/hpml-final-project1?nw=nwuseram5994
* https://wandb.ai/ems2373-none/hpml-vllm-study


---

### C. Running Experiments

#### Naive HuggingFace Baseline

The baseline experiments are implemented in a Jupyter Notebook.

Run all notebook cells sequentially from top to bottom.


Note: If Weights & Biases logging becomes unresponsive, restart the runtime and re-run.

---

#### Text Generation Inference (TGI)

**Start the TGI Server**
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf \
  --max-concurrent-requests 128 \
  --max-batch-total-tokens 32768
```

**Verify Server**
```bash
curl http://localhost:8080/info
```

**Run Throughput and TTFT Experiments**
```bash
python3 TGI/metrics.py
```

Outputs:
- Raw results: `results/metrics_study/metrics_data.json`
- Aggregated results: `results/metrics_study/metrics_summary.csv`
- Plots: throughput, TTFT, and cost vs. batch size

**Run GPU Memory / KV-Cache Profiling**
```bash
python3 TGI/kv_cache.py
```

Outputs:
- Memory time series: `results/kv_cache_study/gpu_memory_data.json`
- Memory utilization plot: `results/kv_cache_study/gpu_memory_timeline.png`

---

#### vLLM

**Start the vLLM Server**
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --trust-remote-code \
  --disable-log-requests \
  --max-model-len 4096
```

**Verify Server**
```bash
curl http://localhost:8000/health
```

**Run Throughput Benchmarks**
```bash
cd "Ethan VLLM"
python3 vllm_benchmarks.py
```

Outputs:
- Per-run results: `<OUTPUT_DIR>/<timestamp>/SERVE--*/run=*.json`
- Aggregated summary: `<OUTPUT_DIR>/<timestamp>/summary.csv`
- Metrics: `request_throughput`, `output_throughput`, `mean_ttft_ms`, `mean_tpot_ms`, `mean_e2el_ms`

**Run KV-Cache Profiling**
```bash
cd "Ethan VLLM"
python3 kv_cache.py
```

Outputs:
- Time series data: `kv_cache_study/<task>/kv_cache_data.json`
- Utilization plot: `kv_cache_study/<task>/kv_cache_timeline.png`

---

### D. Evaluation

**Metrics Collected**:
- **Throughput**: Total tokens generated divided by wall-clock time (tokens/s)
- **Time to First Token (TTFT)**: Time from request submission to first token emission (ms)
- **KV-Cache Utilization**: Memory usage for attention context storage
- **Cost per Token**: Derived from throughput and GPU hourly rate ($2.93/hour for L4)

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our key result (vLLM achieving 1142.8 tok/s on reasoning at batch 128):

```bash
# Step 1: Set up environment
pip install torch transformers requests pandas numpy matplotlib huggingface-hub wandb vllm einops
export HF_TOKEN=<your_huggingface_token>

# Step 2: Start vLLM server
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --trust-remote-code \
  --disable-log-requests \
  --max-model-len 4096

# Step 3: Run benchmarks (in a separate terminal)
cd "Ethan VLLM"
python3 vllm_benchmarks.py

# Step 4: Check results
cat <OUTPUT_DIR>/<timestamp>/summary.csv
```

---

## 5. Notes

- All experiments use FP16 precision with KV caching enabled
- Prompts used for evaluation are included in `prompts.txt`
- GPU cost calculations assume $2.93/hour (NVIDIA L4 on-demand)
- vLLM exposes KV-cache utilization directly via its `/metrics` endpoint
- TGI pre-allocates KV-cache memory at startup; utilization is inferred indirectly via `nvidia-smi`
- Baseline exhausts GPU memory at lower batch sizes for complex tasks (Sentiment maxes at 32, Summary at 16)

Blog Post: https://medium.com/@am5994/decomposing-llm-inference-optimizations-a-comparative-performance-study-of-vllm-tgi-and-naive-2fb20a0f5568?postPublishedType=initial

---

## 6. Contributions

- **Ethan Simpson**: Implemented vLLM experiments
- **Michael Carrion & Benjamin Cyna**: Implemented TGI experiments
- **Andre Mao & Kevin Wang**: Implemented the baseline
- **All authors**: Contributed to writing and analysis

---

## 7. Acknowledgments

We thank Professor Kaoutar El Maghraoui for her guidance and feedback throughout this project.
