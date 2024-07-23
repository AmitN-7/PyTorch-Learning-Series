import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_dimn, out_dimn):
        super(LinearRegressionModel, self).__init__()
        self.model = torch.nn.Linear(in_dimn, out_dimn)

    # predicting the output
    def forward(self, x):
        y_pred = self.model(x)
        return y_pred
