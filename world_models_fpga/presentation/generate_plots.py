import os
import matplotlib.pyplot as plt
import numpy as np

def generate_presentation_plots(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Latency Comparison (Log scale)
    platforms = ["KR260 ARM CPU", "Laptop GPU\n(RTX 3050)", "Laptop CPU\n(Core Ultra 7)", "KR260 FPGA PL\n(Q8.8 DSP)"]
    latencies_lewm = [593.86, 83.71, 44.64, 0.48] # us
    colors = ["#e11d48", "#f59e0b", "#3b82f6", "#10b981"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    bars = ax.bar(platforms, latencies_lewm, color=colors, width=0.55, edgecolor="#1e293b", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Inference Latency (microseconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("World Model (LeWM) Inference Latency Across Platforms", fontsize=13, fontweight="bold", pad=12)

    for bar, lat in zip(bars, latencies_lewm):
        height = bar.get_height()
        ax.annotate(f"{lat:.2f} µs\n({lat/0.48:.0f}x)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "latency_comparison.png")
    plt.savefig(p1)
    plt.close()
    print(f"Generated {p1}")

    # 2. Power and Energy Efficiency
    power_w = [3.10, 38.5, 28.0, 3.25]
    energy_uj = [1841.0, 3281.0, 1250.0, 1.56] # uJ per inference

    fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax2 = ax1.twinx()

    x = np.arange(len(platforms))
    w = 0.32

    b1 = ax1.bar(x - w/2, power_w, width=w, color="#64748b", label="Total Power (Watts)")
    b2 = ax2.bar(x + w/2, energy_uj, width=w, color="#059669", label="Energy per Inf (µJ)")

    ax1.set_ylabel("Power Consumption (Watts)", color="#334155", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Energy per Inference (µJ, log scale)", color="#059669", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms, fontsize=10, fontweight="bold")
    ax1.set_title("Power & Energy per Inference: 2,100x Efficiency on FPGA", fontsize=13, fontweight="bold", pad=12)

    for bar, eng in zip(b2, energy_uj):
        height = bar.get_height()
        ax2.annotate(f"{eng:.1f} µJ",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46")

    plt.tight_layout()
    p2 = os.path.join(out_dir, "power_energy_comparison.png")
    plt.savefig(p2)
    plt.close()
    print(f"Generated {p2}")

    # 3. Model Navigation Success & Collisions
    models = ["Baseline Policy\n(Reactive)", "LeWorldModel\n(SIGReg JEPA)", "H-JEPA\n(Hierarchical)"]
    success_rate = [0.0, 100.0, 100.0]
    collisions = [13, 0, 0]

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=200)
    b_succ = ax.bar(models, success_rate, color=["#ef4444", "#3b82f6", "#10b981"], width=0.5, edgecolor="#0f172a", lw=1.2)
    ax.set_ylabel("Goal Verification Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.set_title("Language Navigation Evaluation: Behind Yellow Cylinder", fontsize=13, fontweight="bold", pad=12)

    for bar, s, c in zip(b_succ, success_rate, collisions):
        height = bar.get_height()
        status_txt = f"{s:.0f}% Success\n({c} Collisions)" if c > 0 else f"{s:.0f}% Success\n(0 Collisions!)"
        ax.annotate(status_txt,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "model_success_comparison.png")
    plt.savefig(p3)
    plt.close()
    print(f"Generated {p3}")

if __name__ == "__main__":
    out_dir = "/home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/presentation/figures"
    generate_presentation_plots(out_dir)
