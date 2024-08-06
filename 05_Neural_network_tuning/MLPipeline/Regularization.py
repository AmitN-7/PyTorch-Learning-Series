# imporitng necessary libraries
from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np


class Regularization:

    def __init__(self, X_train, y_train, X_test, y_test, loader):

        model_reg = self.model_arch(X_train.shape[1])

        self.train(model_reg, X_train, loader)

        self.evaluate(model_reg, X_test, y_test)

    def evaluate(self, model_reg, X_test, y_test):
        """
        Evaluate
        :param model_reg:
        :return:
        """

        # converting into tensor form
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model_reg(X_test_tensor)
        yhat = list(z.argmax(1))  # getting predicted y
        y_test = list(y_test)
        print(
            "Accuracy Score of Test Data is",
            round(accuracy_score(y_test, yhat) * 100, 2),
            "%",
        )

    def train(self, model_reg, X_train, loader):
        """
        Training
        :param model_reg:
        :return:
        """
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        # Regularization
        # Optimizers require the parameters to optimize and a learning rate
        # add l2 regularization to optimzer by just adding in a weight_decay
        optimizer = optim.Adam(model_reg.parameters(), lr=0.01, weight_decay=1e-5)
        epochs = 10
        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = batch_x  # 64 batch size
                b_y = batch_y.type(torch.LongTensor)  # 64 batch size

                # Training pass
                optimizer.zero_grad()

                output = model_reg(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                print(f"Training loss: {running_loss / len(X_train)}")

    def model_arch(self, input_size):
        """
        model arch
        :return:
        """
        # Creating a network with dropout Layers
        hidden_sizes = [128, 64]
        output_size = 2
        # Build a feed-forward network
        model_reg = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),  # 12x128
            nn.Dropout(0.2),
            # During training, 20% of the neurons will be randomly set to 0.
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # 128x64
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),  # 64x2
            nn.Softmax(dim=1),
        )
        print(model_reg)
        return model_reg
