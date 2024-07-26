## Linear Regression

Linear regression is used for finding linear relationship between target and one or more predictors. 
There are two types of linear regression- Simple and Multiple. Simple linear regression is useful for finding relationship between two continuous variables. One is predictor or independent variable and other is response or dependent variable. It looks for statistical relationship but not deterministic relationship. 

Y_pred = b0 + b1*X

Where, Y_pred is the value predicted by the model, b0 and b1 are the parameters,X is the input data.
In our case Y_pred is the "no_of_days_subscribed"


## Training Process Linear Regression

A typical training procedure for a Linear Regression is as follows:
- Define the Linear Regression that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule.


## Code Description


    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle


    File Name : LinearRegression.py
    File Description : Class of Linear Regression structure
    
    File Name : TrainModel.py
    File Description : Code to train and evaluate the pytorch model


## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`




