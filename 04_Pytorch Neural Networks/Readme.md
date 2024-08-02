## Neural Networks

Neural networks have been at the forefront of Artificial Intelligence research during the last few years, and have provided solutions to many difficult problems like image classification, language translation,etc. 
- PyTorch is one of the leading deep learning frameworks, being at the same time both powerful and easy to use.
- Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. 
- The networks are built from individual parts approximating neurons, typically called units or simply “neurons.” Each unit has some number of weighted inputs. 
- These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit’s output.


## Training Process Neural Nets

A typical training procedure for a neural network is as follows:
- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule.


## Code Description

    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle

    File Name : Preprocessing.py
    File Description : Code to preprocess the data
    
    File Name : Pytorch_NN.py
    File Description : Code to train and evaluate the pytorch neural network model


## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization


