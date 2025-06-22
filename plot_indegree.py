import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse

base_dir = "statistics/indegree/"
activations = ["relu", "tanh"]
models = ["cnn", "big"]
conditions = ["ori", "adv"]
output_dir = "statistics/indegree/res_nowd/"
os.makedirs(output_dir, exist_ok=True)


selected_classes = [0]
threshold_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def load_normalized_indegrees(path):
    all_vals = []
    for file in glob.glob(os.path.join(path, "*.npz")):
        data = np.load(file)
        if "in_degrees" in data:
            indegrees = data["in_degrees"]
            all_vals.append(indegrees)
    return all_vals
    # return np.concatenate(all_vals) if all_vals else np.array([])
    

for threshold in threshold_ratio:
    for act in activations:
        for model in models:
            for l in selected_classes:
                data_per_condition = {}

                for cond in conditions:
                    subdir = os.path.join(base_dir, act, f"{model}{cond}", f"{threshold}")
                    if not os.path.exists(subdir):
                        print(subdir)
                        continue

                    values = load_normalized_indegrees(subdir)
                    values = np.array(values)

                    if l >= len(values):
                        print(f"⚠️ Class {l} out of bounds in {subdir}")
                        continue

                    v = np.array(values[l])
                    # print(f'{model} - {act} - {cond}: {len(v[v==0])}')
                    v = v[v > 0]
                    if len(v) > 0:
                        data_per_condition[cond.upper()] = v
                    else:
                        print(f"⚠️ No data in {subdir} for class {l}")

                if not data_per_condition:
                    print(f"!! No data for class {l} — {act} | {model}")
                    continue
                
                res_path = os.path.join(output_dir, f"{threshold}")
                if not os.path.exists(res_path):
                    os.makedirs(res_path)

                # === Combined Boxplot ===
                plt.figure(figsize=(8, 6))
                plt.boxplot(
                    [data_per_condition[cond] for cond in data_per_condition],
                    labels=list(data_per_condition.keys()),
                    patch_artist=True
                )
                plt.title(f"{act.upper()} | {model.upper()} | Class {l} — Boxplot (Normalized In-Degree)")
                plt.ylabel("Normalized In-Degree")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(res_path, f"{act}_{model}_class{l}_combined_box.png"))
                plt.close()

                # # === Combined KDE Histogram ===
                # plt.figure(figsize=(8, 6))
                # for cond, values in data_per_condition.items():
                #     sns.kdeplot(values, label=cond, fill=True, common_norm=False)
                # plt.title(f"{act.upper()} | {model.upper()} | Class {l} — KDE Histogram")
                # plt.xlabel("Normalized In-Degree")
                # plt.ylabel("Density")
                # plt.legend()
                # plt.grid(True)
                # plt.tight_layout()
                # plt.savefig(os.path.join(output_dir, f"{act}_{model}_class{l}_combined_kde.png"))
                # plt.close()

                # === Combined CCDF Plot ===
                plt.figure(figsize=(8, 6))
                for cond, values in data_per_condition.items():
                    sorted_vals = np.sort(values)
                    ccdf_y = 1.0 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                    plt.plot(sorted_vals, ccdf_y, label=cond)

                plt.title(f"{act.upper()} | {model.upper()} | Class {l} — CCDF")
                plt.xlabel("Normalized In-Degree")
                plt.ylabel("CCDF")
                plt.yscale("log")
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(res_path, f"{act}_{model}_class{l}_combined_ccdf.png"))
                plt.close()

                # === Combined Histogram Plot (Bar) ===
                plt.figure(figsize=(8, 6))
                all_values = np.concatenate(list(data_per_condition.values()))
                min_val, max_val = np.min(all_values), np.max(all_values)
                bins = np.linspace(min_val, max_val, 40)

                for cond, values in data_per_condition.items():
                    plt.hist(values, bins=bins, alpha=0.6, label=cond, density=True, edgecolor='black')

                plt.title(f"{act.upper()} | {model.upper()} | Class {l} — Histogram")
                plt.xlabel("Normalized In-Degree")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, linestyle="--", linewidth=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(res_path, f"{act}_{model}_class{l}_combined_hist_bar.png"))
                plt.close()
