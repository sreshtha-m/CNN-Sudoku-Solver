
# CNN Sudoku Solver

### Project Content

|         File         |   Description                         |
| -------------------- | ------------------------------------- |
| data_preprocesses.py | Contains data preprocessing scripts   |
| main.py              | Notebook for running and testing      |
| model.py             | Implementation of the neural network  |

### Description 

This project can solve sudoku puzzles using a convolutional neural network. 

#### Dataset 

https://www.kaggle.com/datasets/bryanpark/sudoku

#### The Model 

The model created uses 5 convolutional layers with the activation function of softplus and a kernel size of 3 by 3 and this was done using the framework keras. 

* The use of "Batch Normalization" between each layer drastically decreases the number of epochs required to train the architecture by instantly normalising between layers. 
* The Softplus activation function works similarly to the use of a ReLu activation function as seen in the table below where I tested and trained some activation functions:

![Batch 32](https://user-images.githubusercontent.com/115587363/208547436-b9311676-0406-4365-8553-d66970456e73.jpg)
 
* This is due to the similarity in shape of ReLu and Softplus. However in certain studies and in many cases, ReLu is the more reliable activation and will give a more accurate result and a lower loss. 


#### Limitations 
Unfortunately due to the CPU being used, the model was not able to be trained on more than one epoch due to the large dataset. 

