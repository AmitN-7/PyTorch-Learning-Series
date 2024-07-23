# Pytorch

- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

## Linear Regression

Linear regression is used for finding linear relationship between target and one or more predictors. 
There are two types of linear regression- Simple and Multiple. Simple linear regression is useful for finding relationship between two continuous variables. One is predictor or independent variable and other is response or dependent variable. It looks for statistical relationship but not deterministic relationship. 

Y_pred = b0 + b1*X

Where, Y_pred is the value predicted by the model b0 and b1 are the parameters (if b1>0 then positive relationship) X is the input data.
In our case Y_pred is the body weight  and X is the height we are predicting


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
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `LinearRegression.ipynb`

