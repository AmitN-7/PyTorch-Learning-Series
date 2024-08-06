import pandas as pd

from MLPipeline.Preprocessing import Preprocessing
from MLPipeline.SaveAndLoad import SaveAndLoad
from MLPipeline.Dropout import DropoutLayer
from MLPipeline.EarlyStopping import EarlyStopping
from MLPipeline.NeuralNet import NeuralNet
from MLPipeline.Regularization import Regularization
# Reading the data 
df = pd.read_csv("Input/data.csv")

#put the names of the columns you want to drop
data = Preprocessing(df).drop(["customer_id", "phone_no", "year"])

#dropping null values
data =Preprocessing(data).dropna()

#scaling numerical features
data=Preprocessing(data).scale()

#label encoding categorical features
data=Preprocessing(data).encode()

##smote
target_col='churn' #put name of target column here 
x_smote,y_smote=Preprocessing(data).smote(target_col)


# splitting data
X_train, X_test, y_train, y_test = Preprocessing(data).split_data(x_smote,y_smote)



#loading data
loader= Preprocessing(data).data_loader(X_train,y_train)

# ### Basic Neural Net
NeuralNet(X_train, y_train, X_test, y_test, loader)

# ### Dropout
DropoutLayer(X_train, y_train, X_test, y_test, loader)

# ### Regularization
Regularization(X_train, y_train, X_test, y_test, loader)

# ### Early Stopping
EarlyStopping(X_train, y_train, X_test, y_test, loader)

# SaveAndLoad
SaveAndLoad(X_train, y_train, X_test, y_test, loader)
