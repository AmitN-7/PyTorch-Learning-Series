import pandas as pd
from MLPipeline.TrainModel import TrainModel
from MLPipeline.Preprocessing import Preprocessing


df = pd.read_csv("Input/data.csv")


cols_to_drop = ["customer_id", "phone_no", "year"]
df = Preprocessing(df).drop(cols_to_drop)

data = Preprocessing(df).apply_preprocessing()


target_col = "churn"
maj_cls = 0
min_cls = 1
data_new = Preprocessing(data).resample(target_col, maj_cls, min_cls)


target_col = "churn"
X_train, X_test, y_train, y_test = Preprocessing(data).split_data(data_new, target_col)


n_features, X_train, X_test, y_train, y_test = Preprocessing(data).convert_to_tensor(
    X_train, y_train, X_test, y_test
)


TrainModel(n_features, X_train, X_test, y_train, y_test)
