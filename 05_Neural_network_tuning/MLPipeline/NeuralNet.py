from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np


class NeuralNet:

    def __init__(self, X_train, y_train, X_test, y_test, loader):

        model = self.model_arch(X_train.shape[1])

        self.train(model, X_train, loader)

        self.evaluate(model, X_test, y_test)

    def evaluate(self, model, X_test, y_test):
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model(X_test_tensor)
        yhat = list(z.argmax(1))
        y_test = list(y_test)
        print(
            "Accuracy Score of Test Data is:",
            round(accuracy_score(y_test, yhat) * 100, 2),
            "%",
        )

    def train(self, model, X_train, loader):
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        epochs = 10
        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = batch_x  # 64 batch size
                b_y = batch_y.type(torch.LongTensor)  # 64 batch size

                # Training pass
                optimizer.zero_grad()

                output = model(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                print(f"Training loss: {running_loss / len(X_train)}")

    def model_arch(self, input_size):

        # Build a feed-forward network
        hidden_sizes = [128, 64]
        output_size = 2

        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Softmax(dim=1),
        )
        print(model)
        return model
