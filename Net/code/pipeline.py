import os
import pandas as pd


def create_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
    }


def update_history(history, train_metrics, val_metrics):
    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_rmse"].append(train_metrics["rmse"])
    history["val_rmse"].append(val_metrics["rmse"])


def log_epoch(epoch, total_epochs, train_metrics, val_metrics):
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train_loss={train_metrics['loss']:.3f}, train_rmse={train_metrics['rmse']:.3f} | "
        f"val_loss={val_metrics['loss']:.3f}, val_rmse={val_metrics['rmse']:.3f}"
    )