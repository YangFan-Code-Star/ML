import torch


def _compute_average_loss(total_loss, total_samples):
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def train(model, trainloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for feature, target in trainloader:
        optimizer.zero_grad()
        outputs = model(feature)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = _compute_average_loss(total_loss, total_samples)
    return {"loss": avg_loss, "rmse": avg_loss ** 0.5}


def evaluate(model, dataloader, criterion, return_outputs=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for feature, target in dataloader:
            outputs = model(feature)
            loss = criterion(outputs, target)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if return_outputs:
                predictions.append(outputs.squeeze(1).cpu())
                targets.append(target.squeeze(1).cpu())

    avg_loss = _compute_average_loss(total_loss, total_samples)
    metrics = {"loss": avg_loss, "rmse": avg_loss ** 0.5}

    if return_outputs:
        if predictions:
            metrics["predictions"] = torch.cat(predictions, dim=0)
            metrics["targets"] = torch.cat(targets, dim=0)
        else:
            metrics["predictions"] = torch.empty(0)
            metrics["targets"] = torch.empty(0)

    return metrics


def test(model, testloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for feature in testloader:
            outputs = model(feature)
            predictions.append(outputs.squeeze(1).cpu())

    if not predictions:
        return torch.empty(0)

    return torch.cat(predictions, dim=0)
