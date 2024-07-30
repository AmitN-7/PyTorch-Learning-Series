from .LogisticRegression import LogisticRegression
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class TrainModel:

    def __init__(self, n_features, X_train, X_test, y_train, y_test):

        lr = LogisticRegression(n_features)
        num_epochs = 500
        learning_rate = 0.01
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)

        self.train(criterion, lr, num_epochs, optimizer, X_train, y_train)

        self.evaluate(lr, X_test, y_test)

    def evaluate(self, lr, X_test, y_test):
        with torch.no_grad():
            y_predicted = lr(X_test)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
            print("------o--------")
            print(f"accuracy is: {acc.item()*100:.4f}%")
            print("-------o--------")

        from sklearn.metrics import classification_report

        print("--Classification Matrix--")
        print(classification_report(y_test, y_predicted_cls))
        from sklearn.metrics import confusion_matrix

        print("--confusion Matrix--")
        confusion_matrix = confusion_matrix(y_test, y_predicted_cls)
        print(confusion_matrix)

    def train(self, criterion, lr, num_epochs, optimizer, X_train, y_train):
        for epoch in range(num_epochs):
            y_pred = lr(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch + 1) % 30 == 0:
                print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")
