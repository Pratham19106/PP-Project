"""
Creates a compact Colab deployment zip archive containing:
- solver_variant_c_lazy (src only)
- solver_variant_d_full (src only)
- evaluation scripts
- project_netshield scripts
- verification ground-truths
"""

import os
import zipfile

def create_bundle():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_zip = os.path.join(base_dir, "evaluation", "colab_experiment_bundle.zip")
    
    print(f"Creating Colab deployment bundle: {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["solver_variant_c_lazy/src", "solver_variant_d_full/src", "evaluation", "project_netshield"]:
            abs_folder = os.path.join(base_dir, folder)
            for root, dirs, files in os.walk(abs_folder):
                # skip git, pycache, and existing zip files
                if ".git" in root or "__pycache__" in root or "figures" in root:
                    continue
                for f in files:
                    if f.endswith(".zip") or f.endswith(".png") or f.endswith(".o") or f == "output":
                        continue
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, base_dir)
                    z.write(full_p, rel_p)
                    print(f"  Added {rel_p}")
    
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\nBundle created successfully: {output_zip} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    create_bundle()

