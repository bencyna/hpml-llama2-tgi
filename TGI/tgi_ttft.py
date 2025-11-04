#!/usr/bin/env python3
"""
TGI TTFT Benchmark Script
--------------------------
Measures Time To First Token (TTFT) for a locally running Text Generation Inference (TGI) instance.

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

This script measures how long it takes for the first token to be generated (TTFT) across several prompts, averaging results across multiple runs.
"""

import time
import statistics
from huggingface_hub import InferenceClient


# Configuration
URL = "http://localhost:8080"  # Must point to your active TGI instance
RUNS_PER_PROMPT = 5
MAX_TOKENS = 100

# TODO: Change these to HELM-style prompts
PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "Describe the process of photosynthesis."
]

def measure_ttft(client, prompt):
    """Return TTFT in milliseconds."""
    start = time.time()
    for _ in client.text_generation(prompt, max_new_tokens=MAX_TOKENS, stream=True):
        return (time.time() - start) * 1000  # first token only


def main():
    client = InferenceClient(model=URL)
    print(f"Connected to {URL}\n")

    for prompt in PROMPTS:
        ttfts = []
        print(f"Prompt: {prompt}")
        for i in range(RUNS_PER_PROMPT):
            ttft = measure_ttft(client, prompt)
            ttfts.append(ttft)
            print(f"  Run {i+1}: {ttft:.2f} ms")
        print(f"Mean TTFT:   {statistics.mean(ttfts):.2f} ms")
        print(f"Median TTFT: {statistics.median(ttfts):.2f} ms\n")


if __name__ == "__main__":
    main()
