import torch


def _build_metrics(total_loss, total_correct, total_samples):
    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(model, trainloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for feature, target in trainloader:
        optimizer.zero_grad()
        outputs = model(feature)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == target).sum().item()
        total_samples += batch_size

    return _build_metrics(total_loss, total_correct, total_samples)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for feature, target in dataloader:
            outputs = model(feature)
            loss = criterion(outputs, target)

            batch_size = target.size(0)
            predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == target).sum().item()
            total_samples += batch_size

    return _build_metrics(total_loss, total_correct, total_samples)


def test(model, testloader, criterion):
    return evaluate(model, testloader, criterion)
