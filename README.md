# PyTorch Tornado Detection 
This project uses deep learning to detect tornados in volumetric level 2 radar data.
Before this project there was no publicly available dataset for finding tornado in radar data.
This project involved creating the dataset.
The data loading pipeline is very robust and is highly multithreaded.
The model consists of one 3D CNN layer followed by four 2D CNN layers.
The loss function is custom and allows the model to pinpoint tornados in the data without the exact locations of the tornados ever being specified in the training data.
* dataset.py contains the data loading pipeline
* tornado_data.py contains helper methods for using historical tornado records
* download_radar_data.py is a script for downloading all of the level 2 radar data containing tornados from 2013 - 2022
* model.py contains the model and loss function
* train.py is a script that runs the training loop
* evaluate.py a script that contains tools for evaluating the model

## Running
First download and build the dependencies  
Run download_radar_data.py to get the raw radar data. This comes out to about 113 GB and takes several hours.  
Run train.py to start the training  
Use tensorboard to monitor the training and stop train.py when it has finished converging  
The model will be saved to saved_model.pt every 500 steps


## Dependencies
openstorm_radar_py is a requirement for running this project. 
This project and its dependencies should all be placed in the same directory.  
Ex:  
```
Folder  
  ├PyTorchTornadoDetection
  ├OpenStorm  
  └openstorm_radar_py  
```
Run `git clone https://github.com/JordanSchlick/PyTorchTornadoDetection` and `git clone https://github.com/JordanSchlick/openstorm_radar_py` and `git clone https://github.com/JordanSchlick/OpenStorm` in the same directory do download them all.  
Your system needs to be have an environment capable of build native python modules.  
Run `python setup.py build` inside openstorm_radar_py to build the module.

This project depends on PyTorch, pandas, numpy, matplotlib, and boto3
