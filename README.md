# HPML Project - LLama2 - TGI

Note for TGI kv cache measurement:
Unlike vLLM which dynamically manages memory blocks (and exposes cache utilization metrics), TGI allocates memory upfront for its KV cache and doesn't show dynamic fluctuations via nvidia-smi. So, at the moment, because TGI pre-allocates memory, our kv cache test shows constant GPU usage throughout the test, which isn't too informative.