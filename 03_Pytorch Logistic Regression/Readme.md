## Logistic Regression

Logistic Regression is a statistical method used to model the relationship between a binary or categorical target variable and one or more predictor variables. The goal is to estimate the probability of a particular class or event occurring, given the predictor variables.

logistic regression uses sigmoid function to model the relationship between the predictors and the outcome. The sigmoid function maps any real-valued number to a value between 0 and 1.

The sigmoid function is defined as:

y = 1 / (1 + e^(-x))
where:

y is the predicted probability
e is the base of the natural logarithm
x is a linear combination of the predictors: x = b0 + b1*X1 + b2*X2 + ... + bn*Xn

## Intuition

Data is fed to the model using Pytorch nn module where two arguments has been passed:
First argument consists of the Input Features 
Second argument consists of the number of output we want which is 1 in our case either 0 or 1

Finally, we pass the model parameters to the Sigmoid activation function so as to classify them with a threshold of 0.5 in binary class


## Training Process Logistic Regression

A typical training procedure for a Logistic Regression is as follows:
- Define the Logistic Regression that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule.


## Code Description


    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle


    File Name : LogisticRegression.py
    File Description : Class of Logistic Regression structure
    
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



