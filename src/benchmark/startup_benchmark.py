"""
CLI Startup Benchmark
=====================
Benchmarks the end-to-end execution time of the CLI for a simple search query,
comparing cold-start (no compiled OpenVINO cache) vs warm-start (cached blob).

This captures the full overhead of Python imports, OpenVINO initialization,
model compilation/loading, and FAISS indexing inside a fresh process.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict

from src.embeddings.openvino_encoder import _CACHE_DIR

def run_cli_startup_benchmark(args: list = ["python", "cli.py", "search", "benchmark_test"]) -> Dict:
    """
    Run the end-to-end CLI command and measure execution time.
    """
    start = time.perf_counter()
    subprocess.run(
        args, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        check=False
    )
    return time.perf_counter() - start

def benchmark_startup(n_warm_runs: int = 3) -> Dict:
    results = {}
    
    # ---------------------------------------------------------
    # 1. Cold Start (Delete Cache)
    # ---------------------------------------------------------
    print("Measuring Cold Start (Clearing cache)...")
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)
        
    cold_time = run_cli_startup_benchmark()
    results["cold_s"] = round(cold_time, 2)
    
    # ---------------------------------------------------------
    # 2. Warm Start (Cache exists from the cold run)
    # ---------------------------------------------------------
    warm_times = []
    print(f"Measuring Warm Start ({n_warm_runs} runs)...")
    for i in range(n_warm_runs):
        print(f"  Warm run {i+1}/{n_warm_runs}...")
        warm_time = run_cli_startup_benchmark()
        warm_times.append(warm_time)
        
    results["warm_s_avg"] = round(sum(warm_times) / len(warm_times), 2)
    results["warm_s_min"] = round(min(warm_times), 2)
    results["warm_s_max"] = round(max(warm_times), 2)
    
    if results["warm_s_avg"] > 0:
        results["speedup"] = round(results["cold_s"] / results["warm_s_avg"], 2)

    # ---------------------------------------------------------
    # 3. Daemon Start (Fully resident memory via HTTP API)
    # ---------------------------------------------------------
    print("Measuring Daemon Start (Spawning resident daemon proc)...")
    import urllib.request
    import yaml
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    try:
        with open(PROJECT_ROOT / "configs" / "settings.yaml", "r", encoding="utf-8") as f:
            SETTINGS = yaml.safe_load(f) or {}
    except Exception:
        SETTINGS = {}
    port = SETTINGS.get("daemon", {}).get("port", 8500)
    data_dir = str(PROJECT_ROOT / "data" / "processed")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    daemon_proc = subprocess.Popen(["python", "-m", "src.daemon", "--path", data_dir], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for daemon to become healthy
    daemon_ready = False
    for _ in range(60): # wait up to 30s
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    daemon_ready = True
                    break
        except Exception:
            time.sleep(0.5)

    if daemon_ready:
        daemon_times = []
        for i in range(n_warm_runs):
            print(f"  Daemon run {i+1}/{n_warm_runs}...")
            daemon_time = run_cli_startup_benchmark()
            daemon_times.append(daemon_time)
        results["daemon_s_avg"] = round(sum(daemon_times) / len(daemon_times), 4)
        results["daemon_speedup"] = round(results["cold_s"] / results["daemon_s_avg"], 2)
    else:
        print("  Daemon failed to boot. Skipping daemon benchmark.")

    # Cleanup daemon
    daemon_proc.terminate()
    daemon_proc.wait()
        
    return results

def print_startup_results(results: Dict):
    sep = "-" * 50
    print()
    print("CLI End-to-End Startup Benchmark")
    print(sep)
    
    print(f"Cold Start (No Cache) : {results.get('cold_s')} s")
    print(f"Warm Start (Avg)      : {results.get('warm_s_avg')} s")
    print(f"Warm Start (Min)      : {results.get('warm_s_min')} s")
    print(f"Warm Start (Max)      : {results.get('warm_s_max')} s")
    
    if "daemon_s_avg" in results:
        print(f"\nDaemon Start (Avg)    : {results.get('daemon_s_avg')} s")
    
    speedup = results.get("speedup")
    if speedup:
        print(f"\nCache Speedup Factor  : {speedup}x")
        
    dspeedup = results.get("daemon_speedup")
    if dspeedup:
        print(f"Daemon Speedup Factor : {dspeedup}x")
        
    print(f"\n{sep}")

if __name__ == "__main__":
    results = benchmark_startup()
    print_startup_results(results)
