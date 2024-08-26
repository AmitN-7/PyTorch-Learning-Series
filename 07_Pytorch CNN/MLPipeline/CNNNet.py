from torch import nn, optim

# defining the model architecture
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 3, kernel_size=(50, 50), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3, 3)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
