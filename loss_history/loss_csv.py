import csv
import os

def append_loss_to_csv(epoch_log, csv_path="training_log.csv"):
    # Define header (columns)
    fieldnames = [
        "epoch",
        "train_loss", "val_loss",
        "train_bisim_loss", "train_bisim_z_dist", "train_bisim_r_dist",
        "train_bisim_var_loss", "train_bisim_transition_dist",
        "val_bisim_loss", "val_bisim_z_dist", "val_bisim_r_dist",
        "val_bisim_var_loss", "val_bisim_transition_dist"
    ]

    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Open file in append mode
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write row
        writer.writerow({
            "epoch": epoch_log.get("epoch"),
            "train_loss": epoch_log.get("train_loss"),
            "val_loss": epoch_log.get("val_loss"),
            "train_bisim_loss": epoch_log.get("train_bisim_loss"),
            "train_bisim_z_dist": epoch_log.get("train_bisim_z_dist"),
            "train_bisim_r_dist": epoch_log.get("train_bisim_r_dist"),
            "train_bisim_var_loss": epoch_log.get("train_bisim_var_loss"),
            "train_bisim_transition_dist": epoch_log.get("train_bisim_transition_dist"),
            "val_bisim_loss": epoch_log.get("val_bisim_loss"),
            "val_bisim_z_dist": epoch_log.get("val_bisim_z_dist"),
            "val_bisim_r_dist": epoch_log.get("val_bisim_r_dist"),
            "val_bisim_var_loss": epoch_log.get("val_bisim_var_loss"),
            "val_bisim_transition_dist": epoch_log.get("val_bisim_transition_dist"), 
        })