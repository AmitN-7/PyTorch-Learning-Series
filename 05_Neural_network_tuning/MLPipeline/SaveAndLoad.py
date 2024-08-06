# ### Checkpoint (Loading and saving model)
# importing necessary libraries
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch import optim
import torch
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np


class SaveAndLoad:

    def __init__(self, X_train, y_train, X_test, y_test, loader):

        model_chk = self.model_arch(X_train.shape[1])
        print(model_chk)
        optim, path, torch = self.train(model_chk, X_train, loader)
        self.evaluate(model_chk, optim, path, torch, X_test, y_test)

    def evaluate(self, model_chk, optim, path, torch, X_test, y_test):
        """
        Evaluate the model
        :param model_chk:
        :param optim:
        :param path:
        :param torch:
        :return:
        """
        model_load = model_chk
        optimizer = optim.Adam(model_load.parameters(), lr=1e-4, weight_decay=1e-5)
        checkpoint = torch.load(path + "model_2.pt")
        model_load.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        model_load.eval()
        # converting into tensor form
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model_load(X_test_tensor)
        yhat = list(z.argmax(1))  # getting y predicted
        y_test = list(y_test)
        print(
            "Accuracy Score of Test Data is:",
            round(accuracy_score(y_test, yhat) * 100, 2),
            "%",
        )

    def train(self, model_chk, X_train, loader):
        """
        TRaining the ml model
        :param model_chk:
        :return:
        """
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        # Regularization

        # Optimizers require the parameters to optimize and a learning rate
        # add l2 regularization to optimzer by just adding in a weight_decay
        optimizer = optim.Adam(model_chk.parameters(), lr=0.01)
        epochs = 5
        path = "Output/model/"

        for e in range(epochs):
            running_loss = 0
            for step, (batch_x, batch_y) in enumerate(loader):

                b_x = batch_x  # 64 batch size
                b_y = batch_y.type(torch.LongTensor)  # 64 batch size

                # Training pass
                optimizer.zero_grad()

                output = model_chk(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": model_chk.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": running_loss,
                    },
                    path + "model_" + str(e) + ".pt",
                )
            else:
                print(f"Training loss: {running_loss / len(X_train)}")
        return optim, path, torch

    def model_arch(self, input_size):
        """
        Defining the arch
        :return:
        """
        # Creating a network with dropout Layers
        hidden_sizes = [128, 64]
        output_size = 2
        # Build a feed-forward network
        model_chk = nn.Sequential(
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
        return model_chk
