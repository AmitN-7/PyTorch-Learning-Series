import torch
import torch.nn as nn
import torchvision.transforms as transforms


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    # preidcting y variable
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
