import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

    # Training metrics
    train_metrics = [
        "train_loss",
        "train_bisim_loss", "train_bisim_z_dist", "train_bisim_r_dist",
        "train_bisim_var_loss", "train_bisim_transition_dist"
    ]

    plt.figure(figsize=(12,7))
    for metric in train_metrics:
        plt.plot(df["epoch"], df[metric], label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    train_png_path = os.path.join(base_dir, "train_loss_metrics.png")
    plt.savefig(train_png_path)
    plt.close()
    print(f"Saved training loss plot to {train_png_path}")

    # Validation metrics
    val_metrics = [
        "val_loss",
        "val_bisim_loss", "val_bisim_z_dist", "val_bisim_r_dist",
        "val_bisim_var_loss", "val_bisim_transition_dist"
    ]

    plt.figure(figsize=(12,7))
    for metric in val_metrics:
        plt.plot(df["epoch"], df[metric], label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    val_png_path = os.path.join(base_dir, "val_loss_metrics.png")
    plt.savefig(val_png_path)
    plt.close()
    print(f"Saved validation loss plot to {val_png_path}")

if __name__ == "__main__":
    main()