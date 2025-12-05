import subprocess
import time
import requests
import threading
import re
import os
import signal
import json
import matplotlib.pyplot as plt

#Configurations
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_PATH = "custom_dataset.jsonl"
CUSTOM_PROMPT = "You are an analyst summarizing the reliability challenges of machine learning systems deployed in production. Write a concise technical brief that covers the following points: 1. Why data drift and concept drift can silently degrade model accuracy over time. 2. The difference between offline evaluation metrics and online performance monitoring. 3. How organizations typically detect and respond to such degradations, including examples of monitoring signals or retraining strategies. 4. End with a two-sentence recommendation for maintaining model robustness under changing data distributions. Keep the tone professional and information-dense, as if writing for a senior engineering audience."
OUTPUT_DIR = "results/kv_cache_study/summarization_2"
MAX_SEQ_LEN = 4096
NUM_PROMPTS = 1000


def monitor_metrics(stop_event, results):
    url = "http://localhost:8000/metrics"
    print("Polling metrics started")
    
    start_time = time.time()
    results['history_times'] = []
    results['history_values'] = []

    while not stop_event.is_set():
        try:
            response = requests.get(url, timeout=0.2)
            if response.status_code == 200:
                content = response.text
                
                match = re.search(r'vllm:[a-z_]*cache_usage_perc(?:\{[^}]*\})?\s+([0-9\.]+)', content)
                
                if match:
                    usage_fraction = float(match.group(1))
                    usage_percent = usage_fraction * 100
                    
                    # Update Peak
                    if usage_percent > results['peak_usage_percent']:
                        results['peak_usage_percent'] = usage_percent
                    
                    # Record History
                    elapsed = time.time() - start_time
                    results['history_times'].append(elapsed)
                    results['history_values'].append(usage_percent)

        except Exception:
            pass
        # Sample every 100ms
        time.sleep(0.1)

def plot_timeline(results):
    if not results['history_values']:
        print("No data recorded to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(results['history_times'], results['history_values'], label="KV Cache Usage", color="#1f77b4", linewidth=2)
    
    plt.axhline(y=results['peak_usage_percent'], color='r', linestyle='--', label=f"Peak: {results['peak_usage_percent']:.1f}%")
    
    plt.fill_between(results['history_times'], results['history_values'], alpha=0.3)
    
    plt.ylim(0, 105)
    plt.title(f"KV Cache Utilization Over Time\n(Model: {MODEL_NAME})", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("GPU KV Cache Usage (%)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_file = os.path.join(OUTPUT_DIR, "kv_cache_timeline.png")
    plt.savefig(plot_file)
    print(f"Graph saved to: {plot_file}")
    plt.close()

def main():
    print("Peak KV-Cache Utilization (Time Series)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    abs_dataset_path = os.path.abspath(DATASET_PATH)
    print(f"Generating custom dataset")
    
    with open(abs_dataset_path, "w") as f:
        for _ in range(NUM_PROMPTS):
            f.write(json.dumps({"prompt": CUSTOM_PROMPT}) + "\n")

    print(f"Custom dataset saved to {abs_dataset_path}")
    print("1. Starting Server")
    server_process = subprocess.Popen(
        [
            "vllm", "serve", MODEL_NAME,
            "--port", "8000",
            "--disable-log-requests",
            "--trust-remote-code",
            "--max-model-len", str(MAX_SEQ_LEN)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    #Wait for Server
    print("   Waiting for server readiness...")
    ready = False
    for _ in range(120):
        try:
            if requests.get("http://localhost:8000/health").status_code == 200:
                ready = True
                break
        except:
            time.sleep(1)
            
    if not ready:
        print("Server failed to start. Exiting.")
        server_process.kill()
        return

    # Start Monitor
    stop_event = threading.Event()
    results = {'peak_usage_percent': 0.0, 'history_times': [], 'history_values': []}
    monitor_thread = threading.Thread(target=monitor_metrics, args=(stop_event, results))
    monitor_thread.start()

    # Run Benchmark
    print("Running High-Concurrency Benchmark")
    subprocess.run([
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL_NAME,
        "--dataset-name", "custom",
        "--dataset-path", abs_dataset_path,
        "--num-prompts", str(NUM_PROMPTS), 
        "--max-concurrency", "128", 
        "--request-rate", "inf",
        "--skip-chat-template"
    ], stdout=subprocess.DEVNULL)

    # Cleanup
    stop_event.set()
    monitor_thread.join()
    os.kill(server_process.pid, signal.SIGTERM)
    
    print("\n" + "="*40)
    print(f"RESULT: Peak KV Cache Utilization: {results['peak_usage_percent']:.2f}%")
    print("="*40)
    
    # Save raw data
    json_path = os.path.join(OUTPUT_DIR, "kv_cache_data.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Raw data saved to {json_path}")
    
    # Generate Plot
    try:
        plot_timeline(results)
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()