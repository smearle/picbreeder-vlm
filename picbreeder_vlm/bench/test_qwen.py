"""
Test Qwen3-VL using both HuggingFace Transformers and vLLM backends, with profiling.

Usage:
    python test_qwen.py                           # Test with default image, both backends
    python test_qwen.py path/to/image.png         # Test with custom image
    python test_qwen.py --backend hf              # Test only HuggingFace backend
    python test_qwen.py --backend vllm            # Test only vLLM backend
    python test_qwen.py --warmup 2 --runs 5       # Custom warmup and runs for profiling
"""

import argparse
import base64
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import PIL.Image

from picbreeder_vlm.vlm.vlm_backends import create_vlm_backend


def load_image_bytes(source: str) -> bytes:
    """Load image bytes from URL or file path."""
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


class VLLMQwenBackend:
    """vLLM-based backend for Qwen3-VL."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ):
        self._model_name = model_name
        self._tensor_parallel_size = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._processor = None
    
    def _ensure_model(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
            
            print(f"Loading vLLM model: {self._model_name}")
            self._llm = LLM(
                model=self._model_name,
                tensor_parallel_size=self._tensor_parallel_size,
                max_model_len=self._max_model_len,
                gpu_memory_utilization=self._gpu_memory_utilization,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._SamplingParams = SamplingParams
    
    @property
    def name(self) -> str:
        return f"vLLM:{self._model_name}"
    
    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> str:
        self._ensure_model()
        
        # Convert bytes to PIL Image
        image = PIL.Image.open(BytesIO(image_bytes))
        
        # Build messages in the format expected by Qwen3-VL
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })
        
        # Apply chat template to get the prompt text
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Create sampling params
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Generate with vLLM
        outputs = self._llm.generate(
            [{
                "prompt": prompt_text,
                "multi_modal_data": {"image": image},
            }],
            sampling_params=sampling_params,
        )
        
        return outputs[0].outputs[0].text


def profile_backend(backend, image_bytes: bytes, mime_type: str, prompt: str,
                   warmup_runs: int = 1, timed_runs: int = 3, backend_name: str = ""):
    """Profile a backend's inference speed."""
    times = []
    
    # Warmup runs (first run includes model loading)
    print(f"\n{'='*60}")
    print(f"Profiling: {backend_name}")
    print(f"{'='*60}")
    
    for i in range(warmup_runs):
        print(f"  Warmup run {i+1}/{warmup_runs}...", end=" ", flush=True)
        start = time.perf_counter()
        if hasattr(backend, 'query'):
            response = backend.query(image_bytes, prompt=prompt, mime_type=mime_type)
            text = response.text if hasattr(response, 'text') else response
        else:
            text = backend.query(image_bytes, prompt=prompt, mime_type=mime_type)
        elapsed = time.perf_counter() - start
        print(f"{elapsed:.2f}s")
        if i == 0:
            first_response = text
    
    # Timed runs
    print(f"\n  Timed runs:")
    for i in range(timed_runs):
        print(f"    Run {i+1}/{timed_runs}...", end=" ", flush=True)
        start = time.perf_counter()
        if hasattr(backend, 'query'):
            response = backend.query(image_bytes, prompt=prompt, mime_type=mime_type)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Statistics ({timed_runs} runs):")
    print(f"    Average: {avg_time:.3f}s")
    print(f"    Min:     {min_time:.3f}s")
    print(f"    Max:     {max_time:.3f}s")
    
    return {
        "backend": backend_name,
        "times": times,
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
        "response": first_response,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test and profile Qwen3-VL with HuggingFace and vLLM backends"
    )
    parser.add_argument(
        "image", nargs="?",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image path or URL (default: Qwen demo image)"
    )
    parser.add_argument(
        "--backend", "-b", choices=["hf", "vllm", "both"], default="both",
        help="Which backend to test (default: both)"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=1,
        help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3,
        help="Number of timed runs (default: 3)"
    )
    parser.add_argument(
        "--prompt", "-p", default="Describe this image.",
        help="Prompt to use (default: 'Describe this image.')"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Max model length for vLLM (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image from: {args.image}")
    image_bytes = load_image_bytes(args.image)
    mime_type = "image/png" if args.image.endswith(".png") else "image/jpeg"
    
    results = []
    
    # Test HuggingFace backend
    if args.backend in ("hf", "both"):
        print("\nCreating HuggingFace Transformers backend...")
        hf_backend = create_vlm_backend("qwen3-vl-8b", backend="hf")
        hf_result = profile_backend(
            hf_backend, image_bytes, mime_type, args.prompt,
            warmup_runs=args.warmup, timed_runs=args.runs,
            backend_name="HuggingFace Transformers"
        )
        results.append(hf_result)
        
        # Free memory if testing both
        if args.backend == "both":
            print("\nFreeing HuggingFace model from GPU memory...")
            import torch
            import gc
            del hf_backend
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)  # Give GPU time to free memory
    
    # Test vLLM backend
    if args.backend in ("vllm", "both"):
        print("\nCreating vLLM backend...")
        vllm_backend = VLLMQwenBackend(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            max_model_len=args.max_model_len,
        )
        vllm_result = profile_backend(
            vllm_backend, image_bytes, mime_type, args.prompt,
            warmup_runs=args.warmup, timed_runs=args.runs,
            backend_name="vLLM"
        )
        results.append(vllm_result)
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Backend':<30} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        print("-" * 66)
        for r in results:
            print(f"{r['backend']:<30} {r['avg']:<12.3f} {r['min']:<12.3f} {r['max']:<12.3f}")
        
        # Speedup calculation
        hf_avg = results[0]["avg"]
        vllm_avg = results[1]["avg"]
        if vllm_avg < hf_avg:
            speedup = hf_avg / vllm_avg
            print(f"\nvLLM is {speedup:.2f}x faster than HuggingFace Transformers")
        else:
            speedup = vllm_avg / hf_avg
            print(f"\nHuggingFace Transformers is {speedup:.2f}x faster than vLLM")
    
    # Show sample responses
    print(f"\n{'='*60}")
    print("SAMPLE RESPONSES")
    print(f"{'='*60}")
    for r in results:
        print(f"\n[{r['backend']}]:")
        print(r["response"][:500] + "..." if len(r["response"]) > 500 else r["response"])


if __name__ == "__main__":
    main()
