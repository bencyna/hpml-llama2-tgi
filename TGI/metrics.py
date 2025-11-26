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

def measure_throughput(client, prompt):
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
