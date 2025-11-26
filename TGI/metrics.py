#!/usr/bin/env python3
"""
TGI Benchmark Script
-----------------------------
Measures TTFT, throughput, and cost/token for a locally running Text Generation Inference (TGI) instance.

REQUIREMENTS:
- The TGI Docker container must be running and accessible at http://localhost:8080.
  Example launch command:
      sudo docker run --gpus all -p 8080:80 \
        -v /mnt/docker:/data \
        -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
        ghcr.io/huggingface/text-generation-inference:latest \
        --model-id meta-llama/Llama-2-7b-chat-hf \
        --dtype float16 --num-shard 1

- Install dependencies:
      pip install huggingface-hub
"""

import time
import csv
import statistics
from datetime import datetime
from huggingface_hub import InferenceClient


# Configuration
URL = "http://localhost:8080"        # Must point to your active TGI instance
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
GPU_HOURLY_COST = 0.35       # USD/hour for GCP T4 (https://cloud.google.com/compute/gpus-pricing?hl=en)

# TODO: change these to HELM-style prompts
PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "Describe the process of photosynthesis."
]


# Metric Functions
def measure_ttft(client, prompt):
    """Return Time-To-First-Token (ms)."""
    start = time.time()
    for chunk in client.text_generation(..., stream=True):
      if hasattr(chunk, "token") and chunk.token is not None:
          return (time.time() - start) * 1000


def measure_throughput(client, prompt):
    """Return (tokens_generated, elapsed_time_sec)."""
    start = time.time()
    output = client.text_generation(prompt, max_new_tokens=MAX_TOKENS, details=True)
    elapsed = time.time() - start
    tokens = getattr(output, "generated_tokens", len(output.generated_text.split()))
    return tokens, elapsed


def compute_cost(elapsed_sec, tokens):
    """Compute cost/token given GPU runtime."""
    cost = (GPU_HOURLY_COST / 3600) * elapsed_sec
    return cost / tokens


def main():
    client = InferenceClient(model=URL)
    print(f"Connected to {URL}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"tgi_metrics_{timestamp}.csv"

    # TODO: add warm-up run?

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "run", "ttft_ms", "throughput_tps", "cost_per_token_usd"])

        for prompt in PROMPTS:
            print(f"Prompt: {prompt}")
            ttfts, throughputs, costs = [], [], []

            for i in range(RUNS_PER_PROMPT):
                ttft = measure_ttft(client, prompt)
                tokens, elapsed = measure_throughput(client, prompt)
                throughput = tokens / elapsed
                cost = compute_cost(elapsed, tokens)

                ttfts.append(ttft)
                throughputs.append(throughput)
                costs.append(cost)

                print(f"  Run {i+1}: TTFT={ttft:.2f} ms | Throughput={throughput:.2f} tok/s | Cost/token=${cost:.8f}")
                writer.writerow([prompt, i + 1, f"{ttft:.2f}", f"{throughput:.2f}", f"{cost:.8f}"])

            print(f"  Mean TTFT: {statistics.mean(ttfts):.2f} ms")
            print(f"  Mean Throughput: {statistics.mean(throughputs):.2f} tok/s")
            print(f"  Mean Cost/token: ${statistics.mean(costs):.8f}\n")

    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
