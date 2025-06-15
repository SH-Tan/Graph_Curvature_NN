import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_model_mean_vectors(model_dir):
    """Load all 'mean' vectors from npz files in a model directory"""
    all_means = []
    for fname in os.listdir(model_dir):
        if fname.endswith(".npz") and "in_degree_class_" in fname:
            try:
                data = np.load(os.path.join(model_dir, fname))
                all_means.append(data["mean"])
            except Exception as e:
                print(f"[Warning] Skipping {fname} in {model_dir}: {e}")
    return np.array(all_means) if all_means else None

def plot_box(models_dict, save_path=None):
    plt.figure(figsize=(14, 7))
    data = [v.flatten() for v in models_dict.values()]
    labels = list(models_dict.keys())

    plt.boxplot(data, labels=labels, patch_artist=True, vert=True)
    plt.ylabel("Curvature-Weighted In-Degree")
    plt.title("Boxplot of Mean In-Degrees Across Models")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + 'box.png')
        print(f"[Saved] {save_path}")
    else:
        plt.show()

def plot_hist(models_dict, save_path=None):
    plt.figure(figsize=(14, 7))
    bins = 30

    for label, values in models_dict.items():
        flat = values.flatten()
        plt.hist(flat, bins=bins, alpha=0.5, label=label)

    plt.xlabel("Curvature-Weighted In-Degree")
    plt.ylabel("Frequency")
    plt.title("Histogram of Mean In-Degrees Across Models")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + 'hist.png')
        print(f"[Saved] {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Directory with model subfolders (each has .npz files)")
    parser.add_argument("--mode", choices=["box", "hist"], default="box", help="Plot mode: box or hist")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the plot")
    args = parser.parse_args()

    models_dict = {}
    for subdir in sorted(os.listdir(args.base_dir)):
        full_path = os.path.join(args.base_dir, subdir)
        if not os.path.isdir(full_path):
            continue
        mean_vectors = load_model_mean_vectors(full_path)
        if mean_vectors is not None and mean_vectors.size > 0:
            models_dict[subdir] = mean_vectors.mean(axis=0)
        else:
            print(f"[Skipped] No valid data in {subdir}")

    if not models_dict:
        print("[Error] No valid model data found.")
        return

    if args.mode == "box":
        plot_box(models_dict, args.save)
    else:
        plot_hist(models_dict, args.save)

if __name__ == "__main__":
    main()
