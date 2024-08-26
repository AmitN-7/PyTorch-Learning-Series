## Convolutional Neural Net

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. 
While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. 
Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.


## Code Description

    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle


    File Name : CNNNet.py
    File Description : Class of CNN structure
    
    File Name : TrainModel.py
    File Description : Code to train and evaluate the pytorch model


    File Name : CreateDataset.py
    File Description : Code to load and transform the dataset. 
    
    Link to dataset: https://drive.google.com/file/d/1zGi0IQPIP3S2lVIKmND6oEA81nnJIuJw/view


### Modular code

- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization



