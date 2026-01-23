import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import os
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot training & validation losses for a given model")
    parser.add_argument("model_name", type=str, help="Name of the model (used to locate CSV file)")
    args = parser.parse_args()

    # Construct path
    base_dir = os.path.join("..", "outputs", args.model_name)
    csv_path = os.path.join(base_dir, "training_loss_log.csv")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df[df["epoch"] < 50]

    # Training metrics
    train_metrics = [
        "train_loss",
        "train_z_proprio_loss", "train_standard_l2_loss",
        "train_bisim_loss", "train_bisim_z_dist", "train_bisim_r_dist",
        "train_bisim_var_loss", "train_bisim_transition_dist", "train_bisim_cov_reg"
    ]

    labels = {
        "train_loss": 'Training Loss',
        "train_z_proprio_loss": 'Proprio Loss',
        "train_standard_l2_loss": 'L2 Loss',
        "train_bisim_loss": 'Bisimulation Loss', 
        "train_bisim_z_dist": 'Bisimulation Distance',
        "train_bisim_r_dist": 'Reward Distance',
        "train_bisim_transition_dist": 'Transition Distance', 
        "train_bisim_var_loss": 'Variance Loss',
        "train_bisim_cov_reg": 'Coveriance Regularization'
    }

    plt.figure(figsize=(12,7))
    for metric in train_metrics:
        plt.plot(df["epoch"], df[metric], label=labels[metric])

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.tight_layout()
    train_png_path = os.path.join(base_dir, "train_loss_metrics.png")
    plt.savefig(train_png_path, bbox_inches='tight')
    plt.close()
    print(f"Saved training loss plot to {train_png_path}")

    # Validation metrics
    val_metrics = [
        "val_loss",
        "val_z_proprio_loss", "val_standard_l2_loss",
        "val_bisim_loss", "val_bisim_z_dist", "val_bisim_r_dist",
        "val_bisim_var_loss", "val_bisim_transition_dist", "val_bisim_cov_reg"
    ]

    plt.figure(figsize=(12,7))
    for metric in val_metrics:
        label = metric.replace("val_", "")
        plt.plot(df["epoch"], df[metric], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Metrics")
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.tight_layout()
    val_png_path = os.path.join(base_dir, "val_loss_metrics.png")
    plt.savefig(val_png_path, bbox_inches='tight')
    plt.close()
    print(f"Saved validation loss plot to {val_png_path}")

if __name__ == "__main__":
    main()