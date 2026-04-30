import torch
import torch.optim as optim

from config import EPOCHS, LEARNING_RATE
from dataset import get_FashionMNIST_dataloaders
from model import FashionMNISTModel
from trainer import evaluate, test, train


def train_FashionMNIST():
    trainloader, valloader, testloader = get_FashionMNIST_dataloaders()
    model = FashionMNISTModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_result = train(model, trainloader, criterion, optimizer)
        val_result = evaluate(model, valloader, criterion)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_result['loss']:.4f}, train_acc={train_result['accuracy']:.2%} | "
            f"val_loss={val_result['loss']:.4f}, val_acc={val_result['accuracy']:.2%}"
        )

    test_result = test(model, testloader, criterion)
    print(
        f"Final test metrics | "
        f"loss={test_result['loss']:.4f}, accuracy={test_result['accuracy']:.2%}"
    )
    return model, test_result


def main():
    train_FashionMNIST()


if __name__ == "__main__":
    main()
