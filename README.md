# Introduction
BaseNILM is a tool for solving the energy disaggregation problem. 
It aims to provide a baseline systems for both new and experienced researchers within 
the area of energy disaggregation and Non-Intrusive Load Monitoring. 
For a full description please see the provided documentation under BaseNILM \docu.

# Publication
The BaseNILM toolkit is part of the following NILM survey paper and tries to 
replicate the presented architectures and disaggregation approaches. 
Please cite the following paper when using the BaseNILM toolkit:

P. A. Schirmer and I. Mporas, "Non-Intrusive Load Monitoring: A Review," in IEEE Transactions on Smart Grid, 2022, doi: 10.1109/TSG.2022.3189598

(https://ieeexplore.ieee.org/document/9820770)

Furthermore, please do also cite the corresponding publicly available datasets. 
As well as [1] when using the data balance option, [2] when using the WaveNet pytorch 
implementations and [3] when using the DSC implementation. For a complete list of all 
publicly available datasets please see the NILM survey paper.

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

For GPU based calculations CUDA in combination with cuDNN has been used, 
utilizing the Nvidia RTX 3000 series for calculation. 
The following versions have been tested and proven to work with the BaseNILM toolkit:
- CUDA 11.4
- DNN 8.2.4
- Driver 472.39

# Usage
For a first test run use start.py to train, test and plot a 
10-fold cross validation using the AMPds2 dataset with five loads (deferrable loads). 
If you don't want to train simply set 'setup_Exp['train']=0' as the models 
for the example test run are already stored in BaseNILM \mdl. 
For changing parameters and adapting the parameters please refer to 
the documentation in BaseNILM \docu.

# Results
For the setup described in usage the results can be found below. For all other
results please refer to the documentation in BaseNILM \docu.

	|          |    FINITE STATES   |          POWER ESTIMATION         |   PERCENT OF TOTAL  |
	| item ID  | ACCURACY | F-SCORE | E-ACCURACY |   RMSE   |    MAE    |    EST    |  TRUTH  |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| DWE      |   97.22% |  95.86% |   72.37%   |    0.87% |    0.12%  |    2.51%  |   5.54% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| FRE      |   99.98% |  99.97% |   95.60%   |    0.24% |    0.13%  |   35.58%  |  36.53% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| HPE      |   99.99% |  99.99% |   97.80%   |    0.58% |    0.06%  |   37.60%  |  37.55% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| WOE      |   99.39% |  99.31% |   95.79%   |    0.57% |    0.02%  |    6.51%  |   6.77% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	| CDE      |   99.93% |  99.93% |   98.04%   |    0.69% |    0.02%  |   13.81%  |  13.61% |
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	|----------|----------|---------|------------|----------|-----------|-----------|---------|
	|    AVG   |   99.30% |  99.01% |   95.55%   |    1.26% |    0.07%  |   96.01%  | 100.00% |

# Development
As failure and mistakes are inextricably linked to human nature, the toolkit is obviously not perfect, 
thus suggestions and constructive feedback are always welcome. If you want to contribute to the BaseNILM 
toolkit or spotted any mistake, please contact me via: p.schirmer@herts.ac.uk

# License
The software framework is provided under the MIT license.

# Cite

P. A. Schirmer and I. Mporas, "Non-Intrusive Load Monitoring: A Review," in IEEE Transactions on Smart Grid, 2022, doi: 10.1109/TSG.2022.3189598

(https://ieeexplore.ieee.org/document/9820770)

# References
[1] Pan, Yungang, et al. "Sequence-to-subsequence learning with conditional gan for power disaggregation." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.

[2] Jiang, Jie, et al. "Deep Learning-Based Energy Disaggregation and On/Off Detection of Household Appliances." ACM Transactions on Knowledge Discovery from Data (TKDD) 15.3 (2021): 1-21.

[3] Batra, Nipun, et al. "Towards reproducible state-of-the-art energy disaggregation." Proceedings of the 6th ACM international conference on systems for energy-efficient buildings, cities, and transportation. 2019.
