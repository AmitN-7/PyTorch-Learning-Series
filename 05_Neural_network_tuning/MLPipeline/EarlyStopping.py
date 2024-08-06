# importing necessary libraries
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import Tensor
import numpy as np


class EarlyStopping:

    def __init__(self, X_train, y_train, X_test, y_test, loader):

        model_early_stp = self.model_arch(X_train.shape[1])

        self.train(model_early_stp, X_train, loader)

        self.evaluate(model_early_stp, X_test, y_test)

    # Evaluating on test data
    def evaluate(self, model_early_stp, X_test, y_test):
        """
        Evaluating
        :param model_early_stp:
        :return:
        """
        # converting into tensor form
        X_test_tensor = Tensor(X_test)
        y_test = Tensor(np.array(y_test))
        z = model_early_stp(X_test_tensor)
        yhat = list(z.argmax(1))  # getting predicted y
        y_test = list(y_test)
        print(
            "Accuracy Score of Test Data is ",
            round(accuracy_score(y_test, yhat) * 100, 2),
            "%",
        )

    # model training
    def train(self, model_early_stp, X_train, loader):
        """
        Training model
        :return:
        """
        # Define the loss
        criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        # Regularization

        # Optimizers require the parameters to optimize and a learning rate
        # add l2 regularization to optimzer by just adding in a weight_decay
        optimizer = optim.Adam(model_early_stp.parameters(), lr=1e-4, weight_decay=1e-5)
        epochs = 100
        epochs_no_improve = 0
        early_stop = False
        min_loss = np.Inf
        iter = 0

        for e in range(epochs):
            running_loss = 0
            if early_stop:
                print("Stopped")
                break
            else:
                for step, (batch_x, batch_y) in enumerate(loader):

                    b_x = batch_x  # 64 batch size
                    b_y = batch_y.type(torch.LongTensor)  # 64 batch size

                    # Training pass
                    optimizer.zero_grad()

                    output = model_early_stp(b_x)
                    loss = criterion(output, b_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if abs(running_loss) < abs(min_loss):
                        epochs_no_improve = 0
                        min_loss = running_loss
                    else:

                        epochs_no_improve += 1
                    iter += 1

                    if e > 5 and epochs_no_improve == epochs:
                        print("Early stopping!")
                        early_stop = True
                        break
                    else:
                        continue

                else:
                    print(f"Training loss: {running_loss / len(X_train)}")

    def model_arch(self, input_size):
        """
        Model Architecture
        :return:
        """
        # Creating a network with dropout Layers
        hidden_sizes = [128, 64]
        hidden_sizes = [128, 64]
        output_size = 2
        # Build a feed-forward network
        model_early_stp = nn.Sequential(
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
        print(model_early_stp)
        return model_early_stp
