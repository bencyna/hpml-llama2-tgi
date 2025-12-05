import json
import subprocess
import os
import sys

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" 

DATASET_PATH = "custom_dataset.jsonl"
CUSTOM_PROMPT = "You are an analyst summarizing the reliability challenges of machine learning systems deployed in production. Write a concise technical brief that covers the following points:1. Why data drift and concept drift can silently degrade model accuracy over time. 2. The difference between offline evaluation metrics and online performance monitoring. 3. How organizations typically detect and respond to such degradations, including examples of monitoring signals or retraining strategies. 4. End with a two-sentence recommendation for maintaining model robustness under changing data distributions. Keep the tone professional and information-dense, as if writing for a senior engineering audience."
OUTPUT_DIR = "Throughput_bound_results/llama2_concurrency_sweep/summarization_2"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
MAX_SEQ_LEN = 4096
MAX_NUM_SEQS = 256
NUM_PROMPTS = 100

# 3 Runs to Ensure Stability
NUM_TRIALS = 3 

def main():
    # Check for Hugging Face Token 
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: 'HF_TOKEN' environment variable is not set.")

    # Generate Custom Dataset
    abs_dataset_path = os.path.abspath(DATASET_PATH)
    print(f"Generating custom dataset")
    
    with open(abs_dataset_path, "w") as f:
        for _ in range(NUM_PROMPTS):
            f.write(json.dumps({"prompt": CUSTOM_PROMPT}) + "\n")
        
    print(f"Custom dataset saved to {abs_dataset_path}")

    #Generate 'serve_params.json' 
    print(f"Generating configuration for Model: {MODEL_NAME}")
    serve_params = [
        {
            "max_num_seqs": MAX_NUM_SEQS,
            "max_model_len": MAX_SEQ_LEN,
            "dtype": "auto" 
        }
    ]
    with open("serve_params.json", "w") as f:
        json.dump(serve_params, f, indent=4)

    bench_params = [{"max_concurrency": c} for c in CONCURRENCY_LEVELS]
    
    with open("bench_params.json", "w") as f:
        json.dump(bench_params, f, indent=4)

    # Construct the vLLM Bench Sweep Command
    cmd = [
        "vllm", "bench", "sweep", "serve",
        
        # Server command
        "--serve-cmd", 
        f"vllm serve {MODEL_NAME} --trust-remote-code --disable-log-requests --max-model-len {MAX_SEQ_LEN}",
        
        "--bench-cmd", 
        f"vllm bench serve --backend vllm --model {MODEL_NAME} --dataset-name custom --dataset-path {abs_dataset_path} --endpoint /v1/completions --request-rate inf --num-prompts {NUM_PROMPTS} --skip-chat-template --extra-body '{{\"max_tokens\": 64}}'",
        
        # Config Files
        "--serve-params", "serve_params.json",
        "--bench-params", "bench_params.json",
        
        # Output
        "--output-dir", OUTPUT_DIR,
        "--num-runs", str(NUM_TRIALS)
    ]

    print("\nStarting Benchmark Sweep for Llama 2")
    print(f"Testing Concurrencies: {CONCURRENCY_LEVELS}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nBenchmarking Complete")
        
        #Plot results
        print("Generating plots...")
        plot_cmd = [
            "vllm", "bench", "sweep", "plot",
            OUTPUT_DIR,
            "--var-x", "max_concurrency",
            "--metric", "request_throughput"
        ]
        subprocess.run(plot_cmd, check=False)
        print(f"JSON results saved in {OUTPUT_DIR}")

    except subprocess.CalledProcessError as e:
        print(f"\nError occurred: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Cleanup temp files
    if os.path.exists("serve_params.json"): os.remove("serve_params.json")
    if os.path.exists("bench_params.json"): os.remove("bench_params.json")

if __name__ == "__main__":
    main()