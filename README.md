# DeepCDA

###########################################################################################
The source code for                                                                                             
                                                                                                                
DeepCDA: Deep Cross-Domain Compound-Protein Affinity Prediction through LSTM and Convolutional Neural Network   
###########################################################################################

# Requirements

Python 3.6

Tensorflow 1.11

Keras 2.2.4

Scipy 1.3.2

numpy


# Data


Download the data from the following link

https://drive.google.com/open?id=1B72WnWMbywxK2M9RntquRWQ3cm6U9YoW

Download the folded data from the following link:

https://drive.google.com/open?id=15KotSJWknMOAnHM68RpOh_rqMISsMwsE

# Usage

First, a feature encoder for training data should be learned and saved in a file with the name 'model_name'(use the folded data).

To do this, use the following instruction (with your appropriate hyper parameters):

python Feature_Encoder.py --data_path Davis_Dataset_folded.mat --num_filters 32 64 --smiles_filter_length 4 6 8 --protein_filter_length 8 12 --learning_rate 0.001 --embedding_size 256 --num_epochs 100 --batch_size 256 --model_name davis_model


then, the learned model is used t0 learn a feature encoder for test data (this step is for domain adaptation):

python DomainAdaptation.py --source_data_path davis.mat --target_data_path bindingdb_kinase_ki.mat --model_name davis_model --num_classification_layers 5 --num_epochs 1

