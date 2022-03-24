# Introduction
BaseNILM is a tool for solving the energy dissagregation problem. 
It aims to give a baseline systems for both new and experienced researchers within 
the area of energy disaggregation and Non-Intrusive Load Monitoring. 
For a full description please see the provided documentation in BaseNILM \docu.

# Publication
The BaseNILM toolkit is part of the following NILM survey paper and tries to 
replicate the presented architectures and disaggregation approaches. 
Please cite the following paper when using the BaseNILM toolkit:

P. A. Schirmer and I. Mporas, Non-Intrusive Load Monitoring: A Review

Furthermore, please do also cite the corresponding publicly available datasets. 
For a complete list of all publicly available datasets please see the NILM 
survey paper.

# Datasets
An overview of a few datasets and their locations:

1) REFIT:  https://www.refitsmarthomes.org/datasets/
2) UKDALE: https://jack-kelly.com/data/
3) REDD:   http://redd.csail.mit.edu/
4) AMPds2: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FIE0S4

# Dependencies
The BaseNILM Toolkit was implemented using the following dependencies:
- Python 3.8
- Tensorflow 2.5.0
- Keras 2.4.3

For GPU based calculations CUDA in combination cuDNN has been used, 
utilizing the Nvidia RTX 3000 series for calculation. 
The following version have been tested and proven to work with the BaseNILM toolkit:
- CUDA 11.4
- DNN 8.2.4
- Driver 472.39

# Usage
For a first test run use start.py to train, test and plot a 
5-fold cross validation using the AMPds2 dataset with five loads (deferrable loads). 
If you don't want to train simply set 'setup_Exp['train']=0' as the models 
for the example test run are already stored in BaseNILM \mdl. 
For changing parameters and adapting the parameters please refer to 
the documentation in BaseNILM \docu.

# Results
For the setup described in usage the results can be found below. For all other
results please refer to the documentation in BaseNILM \docu.



# Development
As failure and mistakes are inextricably linked to human nature, the toolkit is obviously not perfect, 
thus suggestions and constructive feedback are always welcome. If you want to contribute to the BaseNILM 
toolkit or spotted any mistake, please contact me via: p.schirmer@herts.ac.uk

