import os
import pandas as pd
import torch
import torch.optim as optim

from config import EPOCHS, LEARNING_RATE, PLOTS_DIR, TEST_TARGET_PATH
from dataset import get_houseprice_dataloaders
from model import HousePriceModel
from pipeline import create_history, log_epoch, update_history
from trainer import evaluate, test, train
from visualizer import save_all_plots


def train_houseprice():
    trainloader, valloader, testloader, input_dim, test_ids = get_houseprice_dataloaders()
    model = HousePriceModel(input_dim)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = create_history()

    for epoch in range(1, EPOCHS + 1):
        train_result = train(model, trainloader, criterion, optimizer)
        val_result = evaluate(model, valloader, criterion)
        update_history(history, train_result, val_result)
        log_epoch(epoch, EPOCHS, train_result, val_result)

    val_outputs = evaluate(model, valloader, criterion, return_outputs=True)
    plot_paths = save_all_plots(history, val_outputs, PLOTS_DIR)

    predictions = test(model, testloader).numpy()
    os.makedirs(os.path.dirname(TEST_TARGET_PATH), exist_ok=True)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submission.to_csv(TEST_TARGET_PATH, index=False)

    print(f"test prediction saved to: {TEST_TARGET_PATH}")
    print(f"plots saved to: {PLOTS_DIR}")

    return model, history, plot_paths


def main():
    train_houseprice()


if __name__ == "__main__":
    main()
