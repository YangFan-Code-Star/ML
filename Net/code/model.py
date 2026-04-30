import torch
import torch.nn.functional as F

from config import HIDDEN_DIM1, HIDDEN_DIM2


class HousePriceModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, HIDDEN_DIM1)
        self.fc2 = torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.fc3 = torch.nn.Linear(HIDDEN_DIM2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
