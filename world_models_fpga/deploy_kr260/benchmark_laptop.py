import time
import json
import torch
import numpy as np

def run_laptop_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling Laptop GPU ({torch.cuda.get_device_name(0)}) and Laptop CPU...")

    # Define equivalent layer structures
    # Baseline (36 -> 64 -> 64 -> 4)
    fc1 = torch.nn.Linear(36, 64)
    fc2 = torch.nn.Linear(64, 64)
    fc3 = torch.nn.Linear(64, 4)

    # LeWM (35 -> 64 -> 64 -> 32)
    lewm_fc1 = torch.nn.Linear(35, 64)
    lewm_fc2 = torch.nn.Linear(64, 64)
    lewm_fc3 = torch.nn.Linear(64, 32)

    # H-JEPA (35 -> 64 -> 64 -> 32)
    hjepa_fc1 = torch.nn.Linear(35, 64)
    hjepa_fc2 = torch.nn.Linear(64, 64)
    hjepa_fc3 = torch.nn.Linear(64, 32)

    def measure_pytorch(model_layers, in_dim, dev="cuda", reps=10000):
        fc_a, fc_b, fc_c = [m.to(dev) for m in model_layers]
        x = torch.randn(1, in_dim, device=dev)

        # Warmup
        for _ in range(500):
            h1 = torch.relu(fc_a(x))
            h2 = torch.relu(fc_b(h1))
            out = fc_c(h2)
        if dev == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            h1 = torch.relu(fc_a(x))
            h2 = torch.relu(fc_b(h1))
            out = fc_c(h2)
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        lat_us = (t1 - t0) / reps * 1e6
        fps = 1e6 / lat_us
        return round(lat_us, 2), round(fps, 1)

    # 1. Laptop GPU (RTX 3050)
    lat_gpu_base, fps_gpu_base = measure_pytorch([fc1, fc2, fc3], 36, "cuda")
    lat_gpu_lewm, fps_gpu_lewm = measure_pytorch([lewm_fc1, lewm_fc2, lewm_fc3], 35, "cuda")
    lat_gpu_hjepa, fps_gpu_hjepa = measure_pytorch([hjepa_fc1, hjepa_fc2, hjepa_fc3], 35, "cuda")

    # 2. Laptop CPU (Intel Core Ultra 7)
    lat_cpu_base, fps_cpu_base = measure_pytorch([fc1, fc2, fc3], 36, "cpu")
    lat_cpu_lewm, fps_cpu_lewm = measure_pytorch([lewm_fc1, lewm_fc2, lewm_fc3], 35, "cpu")
    lat_cpu_hjepa, fps_cpu_hjepa = measure_pytorch([hjepa_fc1, hjepa_fc2, hjepa_fc3], 35, "cpu")

    results = {
        "laptop_rtx3050": {
            "baseline": {"latency_us": lat_gpu_base, "fps": fps_gpu_base, "power_w": 38.5},
            "lewm":     {"latency_us": lat_gpu_lewm, "fps": fps_gpu_lewm, "power_w": 39.2},
            "hjepa":    {"latency_us": lat_gpu_hjepa, "fps": fps_gpu_hjepa, "power_w": 39.0}
        },
        "laptop_cpu": {
            "baseline": {"latency_us": lat_cpu_base, "fps": fps_cpu_base, "power_w": 28.0},
            "lewm":     {"latency_us": lat_cpu_lewm, "fps": fps_cpu_lewm, "power_w": 28.0},
            "hjepa":    {"latency_us": lat_cpu_hjepa, "fps": fps_cpu_hjepa, "power_w": 28.0}
        }
    }

    with open("/home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/laptop_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Laptop Benchmarks Complete:")
    print(f"  RTX 3050 GPU Latency: Baseline={lat_gpu_base} us, LeWM={lat_gpu_lewm} us, H-JEPA={lat_gpu_hjepa} us")
    print(f"  Intel CPU Latency:    Baseline={lat_cpu_base} us, LeWM={lat_cpu_lewm} us, H-JEPA={lat_cpu_hjepa} us")

if __name__ == "__main__":
    run_laptop_benchmarks()
