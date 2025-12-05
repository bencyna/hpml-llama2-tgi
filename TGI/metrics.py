#!/usr/bin/env python3
"""
TGI Benchmark Script
-----------------------------
Measures TTFT, throughput, and cost/token for a locally running Text Generation Inference (TGI) instance.

USAGE
-----------------------------
# Single-request mode (for baseline latency and cost)
python3 tgi_bench.py --mode single

# Concurrency, latency-bound scenario
python3 tgi_bench.py --mode concurrency --scenario latency --runs-per-prompt 5

# Concurrency, throughput-bound scenario
python3 tgi_bench.py --mode concurrency --scenario throughput --runs-per-prompt 5


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
      pip install huggingface-hub wandb

- Set environment variable W_AND_B_API_KEY with your Weights & Biases API key.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from datetime import datetime

import httpx
import wandb
from huggingface_hub import InferenceClient

# Configuration
URL = "http://localhost:8080"        # Must point to your active TGI instance
RUNS_PER_PROMPT = 5
MAX_TOKENS = 200
GPU_HOURLY_COST = 0.71       # USD/hour for GCP L4 (https://getdeploying.com/reference/cloud-gpu/nvidia-l4)
WARMUP_RUNS_PER_PROMPT = 3
RUNS_PER_PROMPT_DEFAULT = 5
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# Prefix caching: caches KV computations across requests for shared prompt prefixes
PREFIX_CACHING = True        # Set to False when running with --disable-prefix-caching

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

# Concurrency scenarios
SCENARIOS = {
    "latency": {
        # Focus on per-request latency; moderate lengths, modest concurrency
        "description": "Latency-bound: longer generations, lower concurrency",
        "concurrency_levels": [1, 2, 4],
        "max_new_tokens": 256,
    },
    "throughput": {
        # Focus on aggregate throughput; shorter generations, higher concurrency
        "description": "Throughput-bound: shorter generations, higher concurrency",
        "concurrency_levels": [1, 4, 8, 16],
        "max_new_tokens": 64,
    },
}

  
def compute_cost(elapsed_sec: float, tokens: int) -> float:
    """Compute cost/token given GPU runtime and hourly rate."""
    if tokens <= 0:
        return float("inf")
    cost = (GPU_HOURLY_COST / 3600.0) * elapsed_sec
    return cost / tokens  

def p95(vals):
    if not vals:
        return float("nan")
    if len(vals) < 2:
        return vals[-1]
    idx = int(0.95 * len(vals)) - 1
    idx = max(0, min(idx, len(vals) - 1))
    return sorted(vals)[idx]
  
  
# Metric Functions Single request mode
def measure_ttft_single(client: InferenceClient, prompt: str, max_tokens: int) -> float:
    """
    Return Time-To-First-Token (ms) using the streaming API via InferenceClient.
    """
    start = time.time()
    for _ in client.text_generation(prompt, max_new_tokens=max_tokens, stream=True):
        return (time.time() - start) * 1000.0
    return float("nan")
  
  
def measure_throughput_and_latency_single(client: InferenceClient, prompt: str, max_tokens: int):
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
  
def init_wandb(mode: str, scenario: str = None):
    """Initialize Weights & Biases logging."""
    wandb.login(key=os.environ.get("W_AND_B_API_KEY"))
    run_name = f"tgi-{mode}" if scenario is None else f"tgi-{mode}-{scenario}"
    wandb.init(project="hpml-tgi-benchmark", name=run_name)
    wandb.config.update({
        "model_name": MODEL_NAME,
        "dtype": "float16",
        "num_shard": 1,
        "max_tokens": MAX_TOKENS,
        "runs_per_prompt": RUNS_PER_PROMPT_DEFAULT,
        "gpu_type": "L4",
        "gpu_hourly_cost": GPU_HOURLY_COST,
        "prefix_caching": PREFIX_CACHING,
        "mode": mode,
        "scenario": scenario
    })


def run_single_mode(runs_per_prompt: int):
    init_wandb("single")
    client = InferenceClient(model=URL)
    print(f"Connected to {URL} (single-request mode)\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"tgi_single_metrics_{timestamp}.csv"

    max_tokens = MAX_TOKENS

    # warmup
    print(f"Warmup: {WARMUP_RUNS_PER_PROMPT} runs per prompt (discarded)")
    for prompt in PROMPTS:
        for _ in range(WARMUP_RUNS_PER_PROMPT):
            try:
                _ = measure_ttft_single(client, prompt, max_tokens)
                _tokens, _elapsed = measure_throughput_and_latency_single(
                    client, prompt, max_tokens
                )
            except Exception as e:
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

            ttfts_ms, latencies_ms, throughputs, costs = [], [], [], []

            for i in range(runs_per_prompt):
                ttft_ms = measure_ttft_single(client, prompt, max_tokens)
                tokens, elapsed_sec = measure_throughput_and_latency_single(
                    client, prompt, max_tokens
                )
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

                # Log to W&B
                wandb.log({
                    "ttft_ms": ttft_ms,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "throughput_tps": throughput,
                    "cost_per_token_usd": cost
                })

                writer.writerow([
                    prompt.replace("\n", "\\n"),
                    i + 1,
                    f"{ttft_ms:.2f}",
                    f"{latency_ms:.2f}",
                    tokens,
                    f"{throughput:.2f}",
                    f"{cost:.8f}",
                ])

            mean_ttft = statistics.mean(ttfts_ms)
            mean_lat = statistics.mean(latencies_ms)
            mean_throughput = statistics.mean(throughputs)
            mean_cost = statistics.mean(costs)

            ttft_sorted = sorted(ttfts_ms)
            lat_sorted = sorted(latencies_ms)
            p50_ttft = ttft_sorted[len(ttft_sorted) // 2]
            p50_lat = lat_sorted[len(lat_sorted) // 2]
            p95_ttft = p95(ttfts_ms)
            p95_lat = p95(latencies_ms)

            print(
                f"  Mean TTFT: {mean_ttft:.2f} ms "
                f"(p50={p50_ttft:.2f}, p95={p95_ttft:.2f})"
            )
            print(
                f"  Mean Latency: {mean_lat:.2f} ms "
                f"(p50={p50_lat:.2f}, p95={p95_lat:.2f})"
            )
            print(
                f"  Mean Throughput: {mean_throughput:.2f} tok/s | "
                f"Mean Cost/token: ${mean_cost:.8f}\n"
            )

    print(f"Single-request results saved to {outfile}")
    wandb.finish()


async def measure_request_stream(
    client: httpx.AsyncClient,
    prompt: str,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    """
    Measure a single request using /generate_stream:

    Returns (ttft_ms, latency_ms, tokens_generated).

    - TTFT: time from send() to first token event.
    - Latency: time from send() to stream end ([DONE]).
    - Tokens: count of token events.
    """
    url = f"{URL}/generate_stream"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False, 
        },
    }

    t0 = time.perf_counter()
    ttft_ms = None
    tokens = 0

    async with client.stream("POST", url, json=payload, timeout=None) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            token_obj = obj.get("token")
            if token_obj is not None:
                tokens += 1
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0

    if ttft_ms is None:
        # No tokens produced; treat entire latency as TTFT (degenerate case)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ttft_ms, latency_ms, tokens


async def run_concurrency_mode(scenario: str, runs_per_prompt: int):
    init_wandb("concurrency", scenario)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'")

    cfg = SCENARIOS[scenario]
    desc = cfg["description"]
    concurrency_levels = cfg["concurrency_levels"]
    max_new_tokens = cfg["max_new_tokens"]

    print(f"Connected to {URL} (concurrency mode)")
    print(f"Scenario: {scenario} — {desc}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Concurrency levels: {concurrency_levels}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"tgi_concurrency_{scenario}_{timestamp}.csv"

    async with httpx.AsyncClient(headers={"Connection": "keep-alive"}) as client:
        # Optional warmup: one small batch per prompt
        print("Concurrency warmup (1 batch per prompt per concurrency level, not logged)")
        for prompt in PROMPTS:
            for c in concurrency_levels:
                tasks = [measure_request_stream(client, prompt, max_new_tokens)
                         for _ in range(min(c, 2))]
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    print(f"  Warmup error (c={c}) for prompt: {e}")
        print("Concurrency warmup complete.\n")

        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario",
                "concurrency",
                "prompt_idx",
                "run",
                "request_idx_in_batch",
                "ttft_ms",
                "latency_ms",
                "tokens",
                "throughput_tok_per_sec",
                "cost_per_token_usd",
            ])

            # For each prompt and concurrency level, collect stats
            for prompt_idx, prompt in enumerate(PROMPTS):
                print(f"Prompt #{prompt_idx} (scenario={scenario}):\n{prompt}\n")

                for c in concurrency_levels:
                    print(f"  Concurrency level: {c}")
                    all_ttft, all_lat, all_thr, all_cost = [], [], [], []

                    for run_id in range(runs_per_prompt):
                        # Launch c concurrent requests
                        tasks = [
                            measure_request_stream(client, prompt, max_new_tokens)
                            for _ in range(c)
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        for req_idx, res in enumerate(results):
                            if isinstance(res, Exception):
                                print(f"    Run {run_id+1}, req {req_idx}: ERROR {res}")
                                continue

                            ttft_ms, latency_ms, tokens = res
                            throughput = (
                                tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
                            )
                            cost = compute_cost(latency_ms / 1000.0, tokens)

                            all_ttft.append(ttft_ms)
                            all_lat.append(latency_ms)
                            all_thr.append(throughput)
                            all_cost.append(cost)

                            # Log to W&B
                            wandb.log({
                                "concurrency": c,
                                "ttft_ms": ttft_ms,
                                "latency_ms": latency_ms,
                                "tokens": tokens,
                                "throughput_tps": throughput,
                                "cost_per_token_usd": cost
                            })

                            writer.writerow([
                                scenario,
                                c,
                                prompt_idx,
                                run_id + 1,
                                req_idx,
                                f"{ttft_ms:.2f}",
                                f"{latency_ms:.2f}",
                                tokens,
                                f"{throughput:.2f}",
                                f"{cost:.8f}",
                            ])

                    if all_ttft:
                        mean_ttft = statistics.mean(all_ttft)
                        mean_lat = statistics.mean(all_lat)
                        mean_thr = statistics.mean(all_thr)
                        mean_cost = statistics.mean(all_cost)

                        p50_ttft = sorted(all_ttft)[len(all_ttft) // 2]
                        p50_lat = sorted(all_lat)[len(all_lat) // 2]
                        p95_ttft = p95(all_ttft)
                        p95_lat = p95(all_lat)

                        print(
                            f"    TTFT ms: mean={mean_ttft:.2f}, "
                            f"p50={p50_ttft:.2f}, p95={p95_ttft:.2f}"
                        )
                        print(
                            f"    Latency ms: mean={mean_lat:.2f}, "
                            f"p50={p50_lat:.2f}, p95={p95_lat:.2f}"
                        )
                        print(
                            f"    Throughput: mean={mean_thr:.2f} tok/s | "
                            f"Mean cost/token=${mean_cost:.8f}\n"
                        )
                    else:
                        print("    No successful requests for this setting.\n")

    print(f"Concurrency results saved to {outfile}")
    wandb.finish()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["single", "concurrency"],
        default="single",
        help="Benchmark mode: 'single' (per-request) or 'concurrency' (multiple parallel requests).",
    )
    ap.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="latency",
        help="Concurrency scenario: 'latency' or 'throughput' (only used in concurrency mode).",
    )
    ap.add_argument(
        "--runs-per-prompt",
        type=int,
        default=RUNS_PER_PROMPT_DEFAULT,
        help="Number of measured runs per prompt (per concurrency level in concurrency mode).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.mode == "single":
        run_single_mode(args.runs_per_prompt)
    else:
        asyncio.run(run_concurrency_mode(args.scenario, args.runs_per_prompt))


if __name__ == "__main__":
    main()
