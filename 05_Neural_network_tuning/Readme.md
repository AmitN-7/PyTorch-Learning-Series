## Neural Networks

Neural networks have been at the forefront of Artificial Intelligence research during the last few years, and have provided solutions to many difficult problems like image classification, language translation or Alpha Go. 
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

## Neural Net Hyperparameters

- Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel.
During training, some number of layer outputs are randomly ignored or “dropped out.” This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. In effect, each update to a layer during training is performed with a different “view” of the configured layer.
Dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.

- The amount that the weights are updated during training is referred to as the step size or the “learning rate.”
The learning rate controls how quickly the model is adapted to the problem. Smaller learning rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs.

- Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. This in turn improves the model’s performance on the unseen data as well.
A modern approach to reducing generalization error is to use a larger model that may be required to use regularization during training that keeps the weights of the model small. These techniques not only reduce overfitting, but they can also lead to faster optimization of the model and better overall performance

- Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset.
The idea is very simple. The model tries to chase the loss function crazily on the training data, by tuning the parameters.

- The ModelCheckpoint callback class allows you to define where to checkpoint the model weights, how the file should named and under what circumstances to make a checkpoint of the model



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


