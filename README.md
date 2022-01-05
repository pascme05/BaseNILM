# Introduction
BaseNILM is a tool for solving the energy dissagregation problem. It aims to give a baseline systems for both new and experience researchers within the area of energy disaggregation and Non-Intrusive Load Monitoring. For a full description please see the provided documentation in BaseNILM\docu.

# Publication
The BaseNILM toolkit is part of the following NILM survey paper and tries to replicate the presented architectures and disaggregation approaches. Please cite the following paper when using the BaseNILM toolkit:

Furthermore, please do also cite the corresponding publicly available datasets. For a complete list of all publicly available datasets please see the NILM survey paper.

# Depedencies
The BaseNILM Toolkit was implemented using the following dependencies:
- Python 3.8
- Tensorflow 2.5.0
- Keras 2.4.3

For GPU based calculations CUDA in combination cuDNN has been used, utilizing the Nvidia RTX 3000 series for calculation. The following version have been tested and proven to work with the BaseNILM tookit:
- CUDA 11.4
- DNN 8.2.4
- Driver 472.39

# Usage
For a first test run use start.py to train, test and plot a 10-fold cross validation using the REDD-2 dataset with four loads. If you dont want to train simply set setup_Exp['train']=0 as the models for the example test run are already stored in BaseNILM\mdl. For changing parameters and adapting the parameters please refer to the documentation in BaseNILM\docu.

# Development
If you want to contribute to the BaseNILM toolkit, please contact me via: 
