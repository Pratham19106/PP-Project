# Google Colab GPU Benchmark & Plotting Guide

This directory contains the automated Google Colab experimental environment to benchmark all 4 GPU solver variants, verify exactness, extract hardware telemetry, and generate publication-quality figures on an NVIDIA GPU (e.g. Free T4 on Google Colab).

---

## 🚀 3-Step Quickstart Instructions

### Step 1: Open Google Colab
1. Navigate to **[colab.research.google.com](https://colab.research.google.com/)** in your web browser.
2. In the pop-up dialog, select the **Upload** tab.
3. Drag and drop or browse to:
   ```
   d:\PP Project\evaluation\Exact_MVC_GPU_Research_Colab_Suite.ipynb
   ```

### Step 2: Enable GPU Hardware Acceleration
1. In the Colab top menu, click **Runtime** → **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (Standard free tier) or **A100 / V100** if you have Colab Pro.
3. Click **Save**.

### Step 3: Run the Entire Suite (1-Click)
1. Click **Runtime** → **Run all** (or press `Ctrl + F9`).
2. The notebook will automatically:
   - Detect the GPU architecture (`sm_75` for T4, `sm_70` for V100, `sm_80` for A100).
   - Clone the base repositories and deploy Variant C (Lazy Trigger) and Variant D (Proposed Full Solver).
   - Compile all 4 solver variants using `nvcc`.
   - Run the full ablation benchmark across scale-free and dense graph suites.
   - Assert 100% exact cover optimality against ground-truth values.
   - Profile hardware telemetry (`USE_COUNTERS=1`) for wasted BFS scans and in-block fast-path hits.
   - Render and display all 300 DPI publication plots inline.
   - Automatically download `Exact_MVC_GPU_Paper_Results.zip` to your computer containing all CSV tables, logs, and figures (PNG + vector PDF).

---

## 📦 What is in the Download Package?
When the notebook finishes, your browser will automatically download `Exact_MVC_GPU_Paper_Results.zip`, containing:
1. `results_ablation.csv` — Full wall-clock runtimes and exact MVC sizes for all 4 variants.
2. `summary_telemetry.csv` — BFS scan counts, wasted scan percentages, cycle fast-path hits, and tree fast-path hits.
3. `paper_figures/fig1_speedup_ablation.png` & `.pdf` — Runtime speedup vs. 2026 TPDS baseline.
4. `paper_figures/fig2_solve_time_log.png` & `.pdf` — Log-scale comparison across all 4 solver variants.
5. `paper_figures/fig3_bfs_overhead_reduction.png` & `.pdf` — Reduction in wasted parallel BFS passes.

---

## 🛠️ Alternative Manual Upload Bundle
If you ever wish to upload the code directly rather than having Colab clone from GitHub, the deployment zip is ready at:
```
d:\PP Project\evaluation\colab_experiment_bundle.zip (0.13 MB)
```

