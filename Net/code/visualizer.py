import os

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def save_training_curves(history, output_dir):
    _prepare_output_dir(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    rmse_path = os.path.join(output_dir, "rmse_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_rmse"], label="train_rmse")
    plt.plot(epochs, history["val_rmse"], label="val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Validation RMSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(rmse_path)
    plt.close()

    return {
        "loss_curve": loss_path,
        "rmse_curve": rmse_path,
    }


def save_validation_plots(targets, predictions, output_dir):
    _prepare_output_dir(output_dir)
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    residuals = predictions - targets

    scatter_path = os.path.join(output_dir, "val_actual_vs_pred.png")
    min_value = min(targets.min(), predictions.min())
    max_value = max(targets.max(), predictions.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, alpha=0.6, s=18)
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Validation: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    residual_hist_path = os.path.join(output_dir, "val_residual_hist.png")
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Validation Residual Distribution")
    plt.tight_layout()
    plt.savefig(residual_hist_path)
    plt.close()

    residual_scatter_path = os.path.join(output_dir, "val_residual_scatter.png")
    plt.figure(figsize=(8, 5))
    plt.scatter(targets, residuals, alpha=0.6, s=18)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Residual")
    plt.title("Validation Residuals vs Actual")
    plt.tight_layout()
    plt.savefig(residual_scatter_path)
    plt.close()

    return {
        "actual_vs_pred": scatter_path,
        "residual_hist": residual_hist_path,
        "residual_scatter": residual_scatter_path,
    }


def save_all_plots(history, val_outputs, output_dir):
    plot_paths = {}
    plot_paths.update(save_training_curves(history, output_dir))
    plot_paths.update(
        save_validation_plots(
            val_outputs["targets"].numpy(),
            val_outputs["predictions"].numpy(),
            output_dir,
        )
    )
    return plot_paths
