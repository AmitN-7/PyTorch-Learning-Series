import pandas as pd
from MLPipeline.TrainModel import TrainModel
from MLPipeline.Preprocessing import Preprocessing

df = pd.read_csv("Input/data.csv")

df = Preprocessing(df).drop(["customer_id", "phone_no", "year"])

data = Preprocessing(df).dropna()

data = Preprocessing(data).scale()

data = Preprocessing(data).encode()

target_col = "no_of_days_subscribed"
X_train, X_test, y_train, y_test = Preprocessing(data).split_data(target_col)

n_features, X_train, y_train, X_test, y_test = Preprocessing(data).convert_to_tensor(
    X_train, y_train, X_test, y_test
)


TrainModel(n_features, X_train, y_train, X_test, y_test)
