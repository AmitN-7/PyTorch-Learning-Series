import torch.nn as nn
import sklearn
import matplotlib.pyplot as plt
import torch
from .LinearRegression import LinearRegressionModel


class TrainModel:

    def __init__(self, n_features, X_train, y_train, X_test, y_test):
        in_dimn = n_features
        out_dimn = 1
        model = LinearRegressionModel(in_dimn, out_dimn)
        self.train(model, X_train, y_train)
        self.evaluate(model, X_test, y_test)

    def evaluate(self, model, X_test, y_test):
        y_pred = model(X_test).detach().numpy()
        from sklearn.metrics import r2_score, mean_squared_error

        print("------------------")
        print(f"R2 Score:{r2_score(y_test, y_pred):.4f}")
        print(f"Mean sqaured error:{mean_squared_error(y_test,y_pred):.4f}")
        from matplotlib.pyplot import figure

        figure(figsize=(20, 6), dpi=80)
        plt.plot(y_test)
        plt.xlabel("y_test")
        plt.savefig("Output/y_test.png")
        figure(figsize=(20, 6), dpi=80)
        plt.plot(y_pred, color="red")
        plt.xlabel("y_pred")
        plt.savefig("Output/y_pred.png")

    def train(self, model, X_train, y_train):
        num_epochs = 600
        learning_rate = 0.01
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch + 1) % 30 == 0:
                print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")
