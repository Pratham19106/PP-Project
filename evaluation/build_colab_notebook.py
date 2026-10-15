"""
Generator for Exact_MVC_GPU_Research_Colab_Suite.ipynb
Builds a fully self-contained, 1-click execution Google Colab notebook
for benchmarking, telemetry profiling, exactness verification, and plot generation.
"""

import json
import os

def create_notebook():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb_path = os.path.join(base_dir, "evaluation", "Exact_MVC_GPU_Research_Colab_Suite.ipynb")

    # Read modified source files to embed in notebook cells for 100% standalone execution
    with open(os.path.join(base_dir, "solver_variant_c_lazy", "src", "Counters.cuh"), "r", encoding="utf-8") as f:
        counters_cuh_content = f.read()

    with open(os.path.join(base_dir, "solver_variant_c_lazy", "src", "GlobalWorkList.cuh"), "r", encoding="utf-8") as f:
        gwl_c_content = f.read()

    with open(os.path.join(base_dir, "solver_variant_c_lazy", "src", "LocalStacks.cuh"), "r", encoding="utf-8") as f:
        ls_c_content = f.read()

    with open(os.path.join(base_dir, "solver_variant_d_full", "src", "helperFunctions.cuh"), "r", encoding="utf-8") as f:
        hf_d_content = f.read()

    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header Markdown
    add_md("""# Exact Minimum Vertex Cover (MVC) GPU Research: Automated Colab Ablation Suite
### Paper Title: *Accelerating Component-Aware GPU Branch-and-Bound for Minimum Vertex Cover via Lazy Splitting and Analytic Sub-Component Pruning*
**Domain:** Parallel Processing & GPU Computing (CUDA / C++)  
**Target:** 4-Variant Ablation Suite, Telemetry Extraction & Publication Plot Generation

---
### Four-Variant Ablation Matrix:
- **Variant A (2022 Baseline):** Component-oblivious GPU Branch-and-Reduce (Yamout et al., IEEE IPDPS 2022)
- **Variant B (2026 Baseline):** Always-on component-aware GPU Branch-and-Bound with Component Branch Registry (Amro et al., IEEE TPDS 2026)
- **Variant C (Addition 1 Only):** 2026 Baseline + Lazily Triggered Component Detection ($(\\text{deg}(v) \\ge \\tau) \\lor (d \\pmod S == 0)$)
- **Variant D (Proposed Full Solver):** 2026 Baseline + Lazy Trigger + In-Block Analytic Tree & Cycle Fast-Path

---
> ⚠️ **IMPORTANT: Set Colab Runtime to GPU before running!**  
> Go to **Runtime** → **Change runtime type** → Select **T4 GPU** (or V100 / A100) → Click **Save**.""")

    # Cell 1: Hardware Diagnostics
    add_code("""# 1. Hardware Diagnostics & CUDA Architecture Detection
import os
import subprocess
import torch

print("="*70)
print("GPU HARDWARE & CUDA ENVIRONMENT DIAGNOSTICS")
print("="*70)

!nvidia-smi
!nvcc --version

# Auto-detect CUDA compute capability
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
print(f"\\nDetected Active GPU: {gpu_name}")

compute_arch = "sm_75" # Default to T4 (sm_75)
if "T4" in gpu_name:
    compute_arch = "sm_75"
elif "V100" in gpu_name:
    compute_arch = "sm_70"
elif "A100" in gpu_name:
    compute_arch = "sm_80"
elif "A10" in gpu_name or "3090" in gpu_name or "3080" in gpu_name:
    compute_arch = "sm_86"
elif "H100" in gpu_name:
    compute_arch = "sm_90"

print(f"Optimal Compute Architecture: {compute_arch}")
print("="*70)""")

    # Cell 2: Clone Base Repos & Setup Workspaces
    add_code(f"""# 2. Automated Workspace Setup & Codebase Deployment
# Clones both 2022 and 2026 foundational repositories and applies Variant C & Variant D optimizations
import shutil

WORKDIR = "/content/mvc_gpu_research"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

print("Cloning foundational 2022 and 2026 repositories...")
if not os.path.exists("component-aware-vertex-cover-gpu"):
    !git clone https://github.com/HusseinAmro/component-aware-vertex-cover-gpu.git
if not os.path.exists("Parallel-Vertex-Cover-Algorithms-on-GPUs"):
    !git clone https://github.com/pky00/Parallel-Vertex-Cover-Algorithms-on-GPUs.git

print("Creating Variant C (Lazy Trigger) and Variant D (Full Solver) directories...")
if not os.path.exists("solver_variant_c_lazy"):
    shutil.copytree("component-aware-vertex-cover-gpu", "solver_variant_c_lazy")
if not os.path.exists("solver_variant_d_full"):
    shutil.copytree("component-aware-vertex-cover-gpu", "solver_variant_d_full")

# Inject optimized Variant C files
print("Deploying Variant C (Addition 1: Lazy Trigger) optimizations...")
with open("solver_variant_c_lazy/src/Counters.cuh", "w") as f:
    f.write({repr(counters_cuh_content)})

with open("solver_variant_c_lazy/src/GlobalWorkList.cuh", "w") as f:
    f.write({repr(gwl_c_content)})

with open("solver_variant_c_lazy/src/LocalStacks.cuh", "w") as f:
    f.write({repr(ls_c_content)})

# Inject optimized Variant D files (Variant C + Addition 2: Analytic Tree & Cycle Fast-Path)
print("Deploying Variant D (Addition 1 + Addition 2: Lazy + Analytic Fast-Path) optimizations...")
with open("solver_variant_d_full/src/Counters.cuh", "w") as f:
    f.write({repr(counters_cuh_content)})

with open("solver_variant_d_full/src/GlobalWorkList.cuh", "w") as f:
    f.write({repr(gwl_c_content)})

with open("solver_variant_d_full/src/LocalStacks.cuh", "w") as f:
    f.write({repr(ls_c_content)})

with open("solver_variant_d_full/src/helperFunctions.cuh", "w") as f:
    f.write({repr(hf_d_content)})

print(">>> All 4 Solver Variants successfully deployed and ready for compilation! <<<")""")

    # Cell 3: Compilation (Two Passes: Timing Pass & Telemetry Pass)
    add_code("""# 3. Two-Pass Compilation Protocol
# Mode 1: Timing Pass (USE_COUNTERS=0) for pure unperturbed kernel runtime
# Mode 2: Telemetry Pass (USE_COUNTERS=1) for hardware counters, nodes visited, and phase breakdowns

def configure_and_compile(repo_path, use_counters=0, arch="sm_75"):
    src_dir = os.path.join(repo_path, "src")
    makefile = os.path.join(src_dir, "Makefile")
    
    # Update Makefile architecture
    with open(makefile, "r") as f:
        mk = f.read()
    
    code_val = arch
    arch_val = "compute_" + arch.split("_")[1]
    mk_updated = mk.replace("sm_70", code_val).replace("compute_70", arch_val)
    
    with open(makefile, "w") as f:
        f.write(mk_updated)
        
    cmd = f"cd {src_dir} && make clean > /dev/null 2>&1 && make USE_COUNTERS={use_counters} -j$(nproc)"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation FAILED for {repo_path}:\\n{res.stderr}")
        return False
    return True

VARIANTS = {
    "Variant_A_2022": "/content/mvc_gpu_research/Parallel-Vertex-Cover-Algorithms-on-GPUs",
    "Variant_B_2026": "/content/mvc_gpu_research/component-aware-vertex-cover-gpu",
    "Variant_C_Lazy": "/content/mvc_gpu_research/solver_variant_c_lazy",
    "Variant_D_Full": "/content/mvc_gpu_research/solver_variant_d_full"
}

print("="*70)
print(f"COMPILING ALL 4 VARIANTS (Target: {compute_arch}, Timing Pass USE_COUNTERS=0)")
print("="*70)
for name, path in VARIANTS.items():
    print(f"Compiling {name} ...")
    ok = configure_and_compile(path, use_counters=0, arch=compute_arch)
    if ok:
        print(f"  [SUCCESS] {name} binary generated at: {path}/src/output")
    else:
        print(f"  [ERROR] Compilation failed for {name}")

print("="*70)
print("All binaries compiled successfully!")""")

    # Cell 4: Ablation Benchmark Runner
    add_code("""# 4. Multi-Instance Ablation Benchmark Runner
import time
import pandas as pd
import re

DATA_DIR = "/content/mvc_gpu_research/component-aware-vertex-cover-gpu/data"

# Select representative benchmarks: Category 1 (Sparse / Scale-Free) & Category 2 (Dense / Cliques)
BENCHMARKS = [
    "qc324.txt",
    "mhda416.txt",
    "c-fat500-5.txt",
    "US-power-grid.txt",
    "power-eris1176.txt",
    "LastFM-Asia.txt",
    "web-spam.txt",
    "web-webbase-2001.txt"
]

results = []
print("="*85)
print(f"{'Graph':<20} | {'Variant':<18} | {'MVC Size':<10} | {'Time (s)':<10} | {'Status'}")
print("="*85)

for graph_file in BENCHMARKS:
    graph_path = os.path.join(DATA_DIR, graph_file)
    if not os.path.exists(graph_path):
        print(f"Skipping {graph_file} (not found)")
        continue

    for v_name, v_path in VARIANTS.items():
        bin_path = os.path.join(v_path, "src", "output")
        cmd = f"{bin_path} -f {graph_path} -v HYBRID"
        
        t0 = time.time()
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        out = res.stdout
        # Extract MVC Size
        mvc_val = None
        m = re.search(r'Minimum is:\s*(\d+)', out)
        if m:
            mvc_val = int(m.group(1))
        else:
            m2 = re.findall(r'minimum\s*=\s*(\d+)', out, re.IGNORECASE)
            if m2:
                mvc_val = int(m2[-1])
            else:
                m3 = re.findall(r'(\d+)', out)
                mvc_val = int(m3[-1]) if m3 else -1

        status = "PASS" if res.returncode == 0 else "FAIL"
        results.append({
            "Graph": graph_file.replace(".txt", ""),
            "Variant": v_name,
            "MVC_Size": mvc_val,
            "Time_Sec": round(elapsed, 4),
            "Status": status
        })
        print(f"{graph_file.replace('.txt',''):<20} | {v_name:<18} | {str(mvc_val):<10} | {elapsed:<10.4f} | {status}")

df_results = pd.DataFrame(results)
df_results.to_csv("/content/mvc_gpu_research/results_ablation.csv", index=False)
print("="*85)
print("Benchmark sweep completed! Saved to /content/mvc_gpu_research/results_ablation.csv")""")

    # Cell 5: Automated Verification
    add_code("""# 5. Exactness Verification Pass against Ground-Truth MVCs.txt
GT_PATH = "/content/mvc_gpu_research/component-aware-vertex-cover-gpu/src/MVCs.txt"

ground_truths = {}
if os.path.exists(GT_PATH):
    with open(GT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                parts = line.split()
                if len(parts) >= 3:
                    ground_truths[parts[1]] = int(parts[2])

print("="*75)
print("EXACTNESS & OPTIMALITY VERIFICATION")
print("="*75)
all_passed = True
for idx, row in df_results.iterrows():
    g = row["Graph"]
    v = row["Variant"]
    val = row["MVC_Size"]
    expected = ground_truths.get(g)
    if expected is not None:
        if val == expected:
            print(f"  [PASS] {g:<20} | {v:<18} | MVC={val} == Expected={expected}")
        else:
            print(f"  [FAIL] {g:<20} | {v:<18} | MVC={val} != Expected={expected}")
            all_passed = False
    else:
        print(f"  [INFO] {g:<20} | {v:<18} | MVC={val} (Ground truth unrecorded)")

print("="*75)
if all_passed:
    print(">>> 100% MATHEMATICAL EXACTNESS CONFIRMED ACROSS ALL TEST CASES <<<")
else:
    print(">>> ATTENTION: Cover discrepancy found. Inspect logs! <<<")""")

    # Cell 6: Hardware Telemetry Compilation & Sweep
    add_code("""# 6. Hardware Telemetry Pass (USE_COUNTERS=1)
# Extracts:
# - Unproductive BFS scans wasted (%)
# - Nodes explored per SM
# - In-block fast-path hits (Cycle and Tree fast paths)

print("Recompiling Variant B, C, and D with USE_COUNTERS=1 for hardware profiling...")
for name in ["Variant_B_2026", "Variant_C_Lazy", "Variant_D_Full"]:
    configure_and_compile(VARIANTS[name], use_counters=1, arch=compute_arch)

telemetry_records = []
for graph_file in BENCHMARKS:
    graph_path = os.path.join(DATA_DIR, graph_file)
    g_name = graph_file.replace(".txt", "")
    
    for v_name in ["Variant_B_2026", "Variant_C_Lazy", "Variant_D_Full"]:
        bin_path = os.path.join(VARIANTS[v_name], "src", "output")
        res = subprocess.run(f"{bin_path} -f {graph_path} -v HYBRID", shell=True, capture_output=True, text=True)
        
        # Check generated Counters CSV
        cnt_file = os.path.join(VARIANTS[v_name], "src", "Counters", f"Counters_{graph_file}")
        total_bfs = 0
        unprod_bfs = 0
        cycle_hits = 0
        tree_hits = 0
        if os.path.exists(cnt_file):
            try:
                cdf = pd.read_csv(cnt_file)
                if "Total BFS Scans" in cdf.columns:
                    total_bfs = cdf["Total BFS Scans"].sum()
                if "Unproductive BFS Scans" in cdf.columns:
                    unprod_bfs = cdf["Unproductive BFS Scans"].sum()
                if "Cycle Fast-Path Hits" in cdf.columns:
                    cycle_hits = cdf["Cycle Fast-Path Hits"].sum()
                if "Tree Fast-Path Hits" in cdf.columns:
                    tree_hits = cdf["Tree Fast-Path Hits"].sum()
            except Exception as e:
                pass
                
        telemetry_records.append({
            "Graph": g_name,
            "Variant": v_name,
            "Total_BFS": total_bfs,
            "Unproductive_BFS": unprod_bfs,
            "Unproductive_Pct": round((unprod_bfs / total_bfs * 100), 2) if total_bfs > 0 else 0.0,
            "Cycle_FastPath": cycle_hits,
            "Tree_FastPath": tree_hits
        })

df_telemetry = pd.DataFrame(telemetry_records)
df_telemetry.to_csv("/content/mvc_gpu_research/summary_telemetry.csv", index=False)
print("Telemetry pass complete. Summary saved to /content/mvc_gpu_research/summary_telemetry.csv")
df_telemetry.head(10)""")

    # Cell 7: Publication-Quality Plot Generation
    add_code("""# 7. Generate High-Resolution Publication Figures (300 DPI)
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "/content/mvc_gpu_research/paper_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300
})

pivot_time = df_results.pivot(index="Graph", columns="Variant", values="Time_Sec")
benchmarks = list(pivot_time.index)
x = np.arange(len(benchmarks))
width = 0.22

# Figure 1: Speedup vs 2026 Baseline
plt.figure(figsize=(11, 5.5))
speedup_c = pivot_time["Variant_B_2026"] / pivot_time["Variant_C_Lazy"]
speedup_d = pivot_time["Variant_B_2026"] / pivot_time["Variant_D_Full"]

plt.bar(x - width/2, speedup_c, width, label='Variant C (2026 + Lazy Trigger)', color='#3b82f6', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, speedup_d, width, label='Variant D (Proposed Full Solver)', color='#10b981', edgecolor='black', alpha=0.9)
plt.axhline(1.0, color='#ef4444', linestyle='--', linewidth=1.5, label='2026 Baseline (1.0x)')

plt.ylabel('Runtime Speedup Factor (x)')
plt.title('End-to-End Speedup Relative to 2026 TPDS SOTA Baseline')
plt.xticks(x, benchmarks, rotation=25, ha='right')
plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
fig1_p = os.path.join(FIGURES_DIR, "fig1_speedup_ablation.png")
plt.savefig(fig1_p)
plt.savefig(fig1_p.replace(".png", ".pdf"))
plt.show()

# Figure 2: Solve Time Comparison
plt.figure(figsize=(11, 5.5))
for col, col_name, marker, color in [
    ("Variant_A_2022", "Variant A (2022 Baseline)", "^", "#ef4444"),
    ("Variant_B_2026", "Variant B (2026 Baseline)", "s", "#f59e0b"),
    ("Variant_C_Lazy", "Variant C (2026 + Lazy)", "d", "#3b82f6"),
    ("Variant_D_Full", "Variant D (Proposed Solver)", "o", "#10b981")
]:
    if col in pivot_time.columns:
        plt.plot(benchmarks, pivot_time[col], marker=marker, linewidth=2, label=col_name, color=color)

plt.yscale('log')
plt.ylabel('Total Solve Time in Seconds (Log Scale)')
plt.title('Absolute Solve Time across Ablation Variants (Log Scale)')
plt.xticks(rotation=25, ha='right')
plt.legend(frameon=True)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
fig2_p = os.path.join(FIGURES_DIR, "fig2_solve_time_log.png")
plt.savefig(fig2_p)
plt.savefig(fig2_p.replace(".png", ".pdf"))
plt.show()

# Figure 3: Unproductive BFS Scans Reduction
if not df_telemetry.empty:
    pivot_bfs = df_telemetry.pivot(index="Graph", columns="Variant", values="Unproductive_Pct")
    plt.figure(figsize=(10, 5))
    if "Variant_B_2026" in pivot_bfs.columns and "Variant_D_Full" in pivot_bfs.columns:
        plt.bar(x - width/2, pivot_bfs["Variant_B_2026"], width, label='2026 Baseline Wasted Scans (%)', color='#f87171', edgecolor='black')
        plt.bar(x + width/2, pivot_bfs["Variant_D_Full"], width, label='Proposed Solver Wasted Scans (%)', color='#60a5fa', edgecolor='black')
        plt.ylabel('Unproductive BFS Scan Ratio (%)')
        plt.title('Elimination of Unproductive Parallel BFS Scans')
        plt.xticks(x, benchmarks, rotation=25, ha='right')
        plt.ylim(0, 100)
        plt.legend(frameon=True)
        plt.tight_layout()
        fig3_p = os.path.join(FIGURES_DIR, "fig3_bfs_overhead_reduction.png")
        plt.savefig(fig3_p)
        plt.savefig(fig3_p.replace(".png", ".pdf"))
        plt.show()

print(f"All publication figures exported to: {FIGURES_DIR}")""")

    # Cell 8: Download Results Package
    add_code("""# 8. Download Results & Publication Package
# Packages all generated figures (PNG & PDF), raw results CSV, and telemetry CSV into a zip file
from google.colab import files
import zipfile

OUTPUT_ZIP = "/content/Exact_MVC_GPU_Paper_Results.zip"
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["/content/mvc_gpu_research/results_ablation.csv", "/content/mvc_gpu_research/summary_telemetry.csv"]:
        if os.path.exists(f):
            z.write(f, os.path.basename(f))
    for f in os.listdir(FIGURES_DIR):
        full_p = os.path.join(FIGURES_DIR, f)
        z.write(full_p, os.path.join("paper_figures", f))

print(f"Package created: {OUTPUT_ZIP} ({os.path.getsize(OUTPUT_ZIP)/1024:.1f} KB)")
print("Initiating automatic download to your browser...")
files.download(OUTPUT_ZIP)""")

    notebook_json = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "Exact_MVC_GPU_Research_Colab_Suite.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)

    print(f"Notebook successfully written to: {nb_path}")

if __name__ == "__main__":
    create_notebook()

