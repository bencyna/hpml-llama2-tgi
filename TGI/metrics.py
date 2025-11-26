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

- Install dependencies inside a venv:
      python3 -m venv venv
      source venv/bin/activate
      pip install --upgrade pip
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
GPU_HOURLY_COST = 0.71       # USD/hour for GCP L4 (https://getdeploying.com/reference/cloud-gpu/nvidia-l4)
WARMUP_RUNS_PER_PROMPT = 3 

# HELM-style prompts
PROMPTS = [
    # Basic factual recall
    "Basic Factual Recall: What is the capital of Australia? Answer with only the city name.",

    # Simple reasoning
    (
        "Simple Reasoning:\n"
        "Let's think step-by-step. If John is taller than Mark, and Mark is shorter than Sue, "
        "is John definitely taller than Sue? Answer 'Yes', 'No', or 'Cannot determine'."
    ),

    # Sentiment classification
    (
        "Sentiment Classification:\n"
        "Classify the sentiment of the text as 'Positive', 'Negative', or 'Neutral'. "
        "Text: The service was quick and the food was delicious. Sentiment: Positive. "
        "Text: The package arrived late and the box was damaged. Sentiment: Negative. "
        "Text: The meeting ended on time. Sentiment: Neutral. "
        "Text: I finished the book but found the ending disappointing. Sentiment: [FILL IN HERE]"
    ),

    # Summarization (GPU article)
    (
        "Summarization:\n"
        "You are an expert summarizer. Your goal is to write a single-paragraph, abstractive "
        "summary of the provided text, focusing on the main argument and conclusion. The summary "
        "must be brief, no more than 75 words. Use this article: https://en.wikipedia.org/wiki/Graphics_processing_unit"
    ),

    # ~100-token technical brief
    (
        "You are an analyst summarizing the reliability challenges of machine learning systems "
        "deployed in production. Write a concise technical brief that covers the following points:\n"
        "1. Why data drift and concept drift can silently degrade model accuracy over time.\n"
        "2. The difference between offline evaluation metrics and online performance monitoring.\n"
        "3. How organizations typically detect and respond to such degradations, including examples "
        "of monitoring signals or retraining strategies.\n"
        "4. End with a two-sentence recommendation for maintaining model robustness under changing "
        "data distributions.\n"
        "Keep the tone professional and information-dense, as if writing for a senior engineering audience."
    ),
]



# Metric Functions
def measure_ttft(client, prompt):
    """Return Time-To-First-Token (ms)."""
    start = time.time()
    for _ in client.text_generation(prompt, max_new_tokens=MAX_TOKENS, stream=True):
        return (time.time() - start) * 1000.0
    # shouldn't reach here
    return float("nan")

def measure_throughput_and_latency(client, prompt):
    """
    Return (tokens_generated, elapsed_time_sec) using non-streaming generation.
    """
    start = time.time()
    output = client.text_generation(prompt, max_new_tokens=MAX_TOKENS, details=True)
    elapsed = time.time() - start

    # Prefer the model's own token count if available
    tokens = getattr(output, "generated_tokens", None)
    if tokens is None:
        # otherwise just use whitespace tokenization from our prompts
        tokens = len(output.generated_text.split())

    return tokens, elapsed


def compute_cost(elapsed_sec: float, tokens: int) -> float:
    """
    Compute cost/token given GPU runtime and hourly rate.
    """
    if tokens <= 0:
        return float("inf")
    cost = (GPU_HOURLY_COST / 3600.0) * elapsed_sec
    return cost / tokens


def main():
    client = InferenceClient(model=URL)
    print(f"Connected to {URL}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"tgi_metrics_{timestamp}.csv"

    # add warm-up 
    print(f"Warmup: {WARMUP_RUNS_PER_PROMPT} runs per prompt (discarded)")
    for prompt in PROMPTS:
        for _ in range(WARMUP_RUNS_PER_PROMPT):
            try:
                _ = measure_ttft(client, prompt)
                _tokens, _elapsed = measure_throughput_and_latency(client, prompt)
            except Exception as e:
                # don't crash on a single failure
                print(f"  Warmup error for prompt: {e}")
    print("Warmup complete.\n")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt",
            "run",
            "ttft_ms",
            "latency_ms",
            "tokens",
            "throughput_tok_per_sec",
            "cost_per_token_usd",
        ])

        for prompt in PROMPTS:
            print(f"Prompt:\n{prompt}\n")

            ttfts_ms = []
            latencies_ms = []
            throughputs = []
            costs = []

            for i in range(RUNS_PER_PROMPT):
                # TTFT
                ttft_ms = measure_ttft(client, prompt)

                # Throughput & latency
                tokens, elapsed_sec = measure_throughput_and_latency(client, prompt)
                latency_ms = elapsed_sec * 1000.0
                throughput = tokens / elapsed_sec if elapsed_sec > 0 else 0.0
                cost = compute_cost(elapsed_sec, tokens)

                ttfts_ms.append(ttft_ms)
                latencies_ms.append(latency_ms)
                throughputs.append(throughput)
                costs.append(cost)

                print(
                    f"  Run {i+1}: "
                    f"TTFT={ttft_ms:.2f} ms | "
                    f"Latency={latency_ms:.2f} ms | "
                    f"Tokens={tokens} | "
                    f"Throughput={throughput:.2f} tok/s | "
                    f"Cost/token=${cost:.8f}"
                )

                writer.writerow([
                    prompt.replace("\n", "\\n"),
                    i + 1,
                    f"{ttft_ms:.2f}",
                    f"{latency_ms:.2f}",
                    tokens,
                    f"{throughput:.2f}",
                    f"{cost:.8f}",
                ])

            # Summary stats per prompt
            mean_ttft = statistics.mean(ttfts_ms)
            mean_lat = statistics.mean(latencies_ms)
            mean_throughput = statistics.mean(throughputs)
            mean_cost = statistics.mean(costs)

            # p50/p95 for TTFT and latency (only if enough runs)
            ttft_sorted = sorted(ttfts_ms)
            lat_sorted = sorted(latencies_ms)
            p50_ttft = ttft_sorted[len(ttft_sorted) // 2]
            p50_lat = lat_sorted[len(lat_sorted) // 2]

            def p95(vals):
                if len(vals) < 2:
                    return vals[-1]
                idx = int(0.95 * len(vals)) - 1
                idx = max(0, min(idx, len(vals) - 1))
                return sorted(vals)[idx]

            p95_ttft = p95(ttfts_ms)
            p95_lat = p95(latencies_ms)

            print(
                f"  Mean TTFT: {mean_ttft:.2f} ms (p50={p50_ttft:.2f}, p95={p95_ttft:.2f})"
            )
            print(
                f"  Mean Latency: {mean_lat:.2f} ms (p50={p50_lat:.2f}, p95={p95_lat:.2f})"
            )
            print(
                f"  Mean Throughput: {mean_throughput:.2f} tok/s | "
                f"Mean Cost/token: ${mean_cost:.8f}\n"
            )

    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
