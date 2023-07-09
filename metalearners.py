import torch

class LogisticMetaLearner(torch.nn.Module):
    def __init__(self, inputdim, outputdim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(inputdim, outputdim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
