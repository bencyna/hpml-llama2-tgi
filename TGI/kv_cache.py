#!/usr/bin/env python3
"""
TGI KV Cache / GPU Memory Utilization Script
---------------------------------------------
Measures peak GPU memory usage during TGI inference, which includes KV cache footprint.

Unlike vLLM which exposes explicit KV cache metrics, TGI's KV cache is measured indirectly
through GPU memory utilization via nvidia-smi.

REQUIREMENTS:
- TGI Docker container running at http://localhost:8080
- nvidia-smi available on the host

USAGE:
    python3 kv_cache.py
"""

import subprocess
import time
import requests
import threading
import os
import json
import re
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
URL = "http://localhost:8080"
OUTPUT_DIR = "results/kv_cache_study"
NUM_PROMPTS = 100
MAX_CONCURRENCY = 128
PROMPTS_FILE = "../prompts.txt"


def load_prompts(filepath):
    """Load prompts from a text file, stripping leading numbers like '1. '"""
    prompts = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove leading number and period (e.g., "1. " or "2. ")
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    prompts.append(cleaned)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default prompts")
        return None
    return prompts if prompts else None


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            memory_values = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(memory_values)
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
    return 0


def get_gpu_total_memory():
    """Get total GPU memory in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            memory_values = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(memory_values)
    except Exception:
        pass
    return 0


def monitor_gpu_memory(stop_event, results):
    """Monitor GPU memory usage in a background thread."""
    print("GPU memory monitoring started")

    start_time = time.time()
    results['history_times'] = []
    results['history_memory_mb'] = []
    results['history_memory_percent'] = []
    results['peak_memory_mb'] = 0
    results['peak_memory_percent'] = 0.0

    total_memory = get_gpu_total_memory()
    results['total_memory_mb'] = total_memory

    while not stop_event.is_set():
        memory_mb = get_gpu_memory_usage()
        memory_percent = (memory_mb / total_memory * 100) if total_memory > 0 else 0

        if memory_mb > results['peak_memory_mb']:
            results['peak_memory_mb'] = memory_mb
            results['peak_memory_percent'] = memory_percent

        elapsed = time.time() - start_time
        results['history_times'].append(elapsed)
        results['history_memory_mb'].append(memory_mb)
        results['history_memory_percent'].append(memory_percent)

        time.sleep(0.1)


def send_request(prompt, max_tokens=200):
    """Send a single generation request to TGI."""
    try:
        response = requests.post(
            f"{URL}/generate",
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens}
            },
            timeout=120
        )
        return response.status_code == 200
    except Exception as e:
        return False


def run_concurrent_requests(prompts, num_requests, max_concurrency):
    """Run concurrent requests to stress the KV cache."""
    import concurrent.futures

    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    completed = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(send_request, p) for p in request_prompts]

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                completed += 1
            else:
                failed += 1

            if (completed + failed) % 10 == 0:
                print(f"Progress: {completed + failed}/{num_requests} requests")

    return completed, failed


def plot_timeline(results):
    """Plot GPU memory usage over time."""
    if not results['history_memory_mb']:
        print("No data recorded to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(results['history_times'], results['history_memory_mb'],
             label="GPU Memory", color="#1f77b4", linewidth=1.5)
    ax1.axhline(y=results['peak_memory_mb'], color='r', linestyle='--',
                label=f"Peak: {results['peak_memory_mb']} MB")
    ax1.fill_between(results['history_times'], results['history_memory_mb'], alpha=0.3)
    ax1.set_ylabel("GPU Memory Usage (MB)", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(f"TGI GPU Memory Utilization Over Time\n(Model: {MODEL_NAME})", fontsize=14)

    ax2.plot(results['history_times'], results['history_memory_percent'],
             label="GPU Memory %", color="#2ca02c", linewidth=1.5)
    ax2.axhline(y=results['peak_memory_percent'], color='r', linestyle='--',
                label=f"Peak: {results['peak_memory_percent']:.1f}%")
    ax2.fill_between(results['history_times'], results['history_memory_percent'], alpha=0.3, color='green')
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("GPU Memory Usage (%)", fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "gpu_memory_timeline.png")
    plt.savefig(plot_file, dpi=150)
    print(f"Graph saved to: {plot_file}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load prompts from file
    prompts = load_prompts(PROMPTS_FILE)
    if prompts is None:
        print("No prompts loaded, exiting")
        return
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    # Check TGI is running
    try:
        response = requests.get(f"{URL}/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"Model: {info.get('model_id', 'unknown')}")
            print(f"Max concurrent requests: {info.get('max_concurrent_requests', 'unknown')}")
        else:
            print("TGI server not responding correctly")
            return
    except Exception as e:
        print(f"Error connecting to TGI: {e}")
        return

    # Get baseline memory (model loaded, no inference)
    baseline_memory = get_gpu_memory_usage()
    total_memory = get_gpu_total_memory()
    print(f"Baseline: {baseline_memory} MB / {total_memory} MB ({baseline_memory/total_memory*100:.1f}%)")

    # Start monitoring
    stop_event = threading.Event()
    results = {
        'baseline_memory_mb': baseline_memory,
        'model_name': MODEL_NAME,
        'num_prompts': NUM_PROMPTS,
        'max_concurrency': MAX_CONCURRENCY
    }
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(stop_event, results))
    monitor_thread.start()

    # Run concurrent requests to stress KV cache
    print(f"Running {NUM_PROMPTS} concurrent requests (max concurrency: {MAX_CONCURRENCY})...")
    start_time = time.time()
    completed, failed = run_concurrent_requests(prompts, NUM_PROMPTS, MAX_CONCURRENCY)
    elapsed = time.time() - start_time

    time.sleep(2)

    stop_event.set()
    monitor_thread.join()

    # Calculate KV cache overhead
    kv_cache_overhead = results['peak_memory_mb'] - baseline_memory

    # Results
    print(f"Total GPU Memory:     {total_memory} MB")
    print(f"Baseline (model):     {baseline_memory} MB ({baseline_memory/total_memory*100:.1f}%)")
    print(f"Peak Memory:          {results['peak_memory_mb']} MB ({results['peak_memory_percent']:.1f}%)")
    print(f"KV Cache Overhead:    {kv_cache_overhead} MB")
    print(f"Requests completed:   {completed}/{NUM_PROMPTS}")
    print(f"Requests failed:      {failed}")
    print(f"Total time:           {elapsed:.2f}s")
    print(f"Throughput:           {completed/elapsed:.2f} req/s")

    # Save results
    results['kv_cache_overhead_mb'] = kv_cache_overhead
    results['requests_completed'] = completed
    results['requests_failed'] = failed
    results['total_time_sec'] = elapsed
    results['throughput_req_per_sec'] = completed / elapsed if elapsed > 0 else 0

    json_path = os.path.join(OUTPUT_DIR, "gpu_memory_data.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Raw data saved to: {json_path}")

    # Generate plot
    try:
        plot_timeline(results)
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
