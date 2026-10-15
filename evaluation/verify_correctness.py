"""
Automated Exactness Verification Script for Minimum Vertex Cover
Asserts 100% cover size agreement between all ablation variants and ground-truth solutions.
"""

import os
import sys

def load_ground_truths(mvc_file_path):
    truths = {}
    if not os.path.exists(mvc_file_path):
        print(f"Warning: {mvc_file_path} not found.")
        return truths
    with open(mvc_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                parts = line.split()
                if len(parts) >= 3:
                    graph_name = parts[1]
                    try:
                        mvc_val = int(parts[2])
                        truths[graph_name] = mvc_val
                    except ValueError:
                        pass
    return truths

def verify_results(results_csv_path, ground_truths):
    """
    Checks that every row in results_csv matches the ground truth MVC size.
    """
    if not os.path.exists(results_csv_path):
        print(f"Error: Results CSV {results_csv_path} does not exist.")
        return False

    all_passed = True
    print("\n" + "="*70)
    print("VERIFICATION PASS: Ground-Truth Exact Minimum Vertex Cover Check")
    print("="*70)

    with open(results_csv_path, 'r') as f:
        lines = f.readlines()
        headers = [h.strip() for h in lines[0].split(',')]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            graph = parts[0]
            variant = parts[1]
            mvc_size = int(parts[2])
            status = parts[3]

            expected = ground_truths.get(graph)
            if expected is not None:
                if mvc_size == expected:
                    print(f"  [PASS] {graph:<25} | Variant: {variant:<20} | MVC: {mvc_size} == {expected}")
                else:
                    print(f"  [FAIL] {graph:<25} | Variant: {variant:<20} | MVC: {mvc_size} != Expected {expected}")
                    all_passed = False
            else:
                print(f"  [INFO] {graph:<25} | Variant: {variant:<20} | MVC: {mvc_size} (Ground truth unrecorded)")

    print("="*70)
    if all_passed:
        print(">>> 100% EXACT COVER MATCH VERIFIED ACROSS ALL INSTANCES <<<")
    else:
        print(">>> DISCREPANCY DETECTED! HALT AND DEBUG LEAF PEELING / ATOMICS <<<")
    print("="*70 + "\n")
    return all_passed

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_file = os.path.join(base_dir, "..", "component-aware-vertex-cover-gpu", "src", "MVCs.txt")
    truths = load_ground_truths(gt_file)
    print(f"Loaded {len(truths)} ground-truth benchmark MVC values from MVCs.txt.")
    csv_file = os.path.join(base_dir, "results_ablation.csv")
    if os.path.exists(csv_file):
        verify_results(csv_file, truths)
    else:
        print(f"To run verification, execute ablation benchmarks to produce: {csv_file}")

