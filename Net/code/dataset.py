from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, NUM_WORKERS
from data import preprocess_data


class HousePriceDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __getitem__(self, index):
        feature = self.features[index]
        if self.targets is None:
            return feature
        target = self.targets[index]
        return feature, target

    def __len__(self):
        return len(self.features)


def get_houseprice_dataloaders():
    (train_features, train_targets, val_features, val_targets, test_features, input_dim, test_ids,) = preprocess_data()

    train_dataset = HousePriceDataset(train_features, train_targets)
    val_dataset = HousePriceDataset(val_features, val_targets)
    test_dataset = HousePriceDataset(test_features)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,)
    
    return train_loader, val_loader, test_loader, input_dim, test_ids
