import torch
import torch.nn as nn


class AllocationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(54, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return self.softmax(x)
