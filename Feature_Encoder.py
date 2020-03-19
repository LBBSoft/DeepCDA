
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
import random as rn
from keras import backend as K
from itertools import product
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten,Lambda, TimeDistributed, LSTM, Add
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
import sys, pickle, os
import math, json, time
from keras.regularizers import l2
import decimal
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
import argparse



def argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_path',
      type=str,
      default='Davis_Dataset_folded.mat',
      help='Path to the dataset (a .mat file)'
  )
  parser.add_argument(
      '--num_filters',
      type=int,
      nargs='+',
      help='number of filter for convolutional layers'
  )
  parser.add_argument(
      '--smiles_filter_lengths',
      type=int,
      nargs='+',
      help='the size of filters for convolutional layers of smiles encoder'
  )
  parser.add_argument(
      '--protein_filter_lengths',
      type=int,
      nargs='+',
      help='the size of filters for convolutional layers of protein encoder'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      nargs='+',
      help='Learning Rate'
  )
  parser.add_argument(
      '--embedding_size',
      type=int,
      nargs='+',
      help='the embedding size for embedding layers'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=100,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=256,
      help='Batch size'
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='model',
      help='model_name'
  )

  flags, unparsed = parser.parse_known_args()

  return flags



protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
                "U": 19, "T": 20, "W": 21, 
                "V": 22, "Y": 23, "X": 24, 
                "Z": 25 }

protein_dict_len = 25

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

smiles_dict_len = 64

protein_max_len=1000
smiles_max_len=100

def coeff_fun_prot(x):
    import tensorflow as tf
    import keras
        
    tmp_a_1=tf.keras.backend.mean(x[0],axis=-1,keepdims=True)
    tmp_a_1=tf.nn.softmax(tmp_a_1)
    tmp=tf.tile(tmp_a_1,(1,1,keras.backend.int_shape(x[1])[2]))
    return tf.multiply(x[1],tmp)
    
def att_func(x):
    import tensorflow as tf
    import keras
    
    tmp_a_2=tf.keras.backend.permute_dimensions(x[1],(0,2,1))
    mean_all=tf.keras.backend.sigmoid(tf.keras.backend.batch_dot(tf.keras.backend.mean(x[0],axis=1,keepdims=True),tf.keras.backend.mean(tmp_a_2,axis=-1,keepdims=True)))
    tmp_a=tf.keras.backend.sigmoid(tf.keras.backend.batch_dot(x[0],tmp_a_2))*mean_all
    #tmp_a=tf.nn.softmax(tmp_a)
    return tmp_a
def coeff_fun_lig(x):
    import tensorflow as tf
    import keras
    tmp1=tf.keras.backend.permute_dimensions(x[0],(0,2,1))
    tmp_a_1=tf.keras.backend.mean(tmp1,axis=-1,keepdims=True)
    tmp_a_1=tf.nn.softmax(tmp_a_1)
    tmp=tf.tile(tmp_a_1,(1,1,keras.backend.int_shape(x[1])[2]))
    return tf.multiply(x[1],tmp)


TABSY = "\t"
figdir = "figures/"

def feature_extraction_model(FLAGS, embedding_size, num_filters, protein_filter_lengths, smiles_filter_lengths):
    
    Drug_input = Input(shape=(smiles_max_len,), dtype='int32',name='drug_input') 
    Protein_input = Input(shape=(protein_max_len,), dtype='int32',name='protein_input')

    encode_smiles = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding')(Drug_input) 
    encode_smiles = Conv1D(filters=num_filters, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_smiles')(encode_smiles)
    # encode_smiles = MaxPooling1D(2)(encode_smiles)  # test it is effectuve or not
    encode_smiles = Conv1D(filters=num_filters*2, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_smiles')(encode_smiles)
    #encode_smiles = MaxPooling1D(2)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*3, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_smiles')(encode_smiles)
    #encode_smiles = MaxPooling1D(2)(encode_smiles)
    smile_lstm = LSTM(num_filters*3,return_sequences='True', name='lstm_smiles')(encode_smiles)

    
    encode_protein = Embedding(input_dim=protein_dict_len+1, output_dim = embedding_size, input_length=protein_max_len, name='protein_embedding')(Protein_input)
    encode_protein = Conv1D(filters=num_filters, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_prot')(encode_protein)
    #encode_protein = MaxPooling1D(2)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*2, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_prot')(encode_protein)
    #encode_protein = MaxPooling1D(2)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*3, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_prot')(encode_protein)
    #encode_protein = MaxPooling1D(2)(encode_protein)
    protein_lstm = LSTM(num_filters*3,return_sequences='True', name='lstm_prot')(encode_protein) 
    
    att_tmp=TimeDistributed(Dense(num_filters*3, kernel_initializer='normal',kernel_regularizer=l2(0.01),use_bias=False, name='TD_lstm'))(protein_lstm)
    #att_tmp=Dropout(0.1)(att_tmp)
    att=Lambda(att_func)([att_tmp,smile_lstm])
    protein_lstm=Lambda(coeff_fun_prot)([att,protein_lstm])
    smile_lstm=Lambda(coeff_fun_lig)([att,smile_lstm])
    
    att_tmp2=TimeDistributed(Dense(num_filters*3,  kernel_initializer='normal',kernel_regularizer=l2(0.01),use_bias=False, name='TD_conv'))(encode_protein)
    # att_tmp2=Dropout(0.5)(att_tmp2)
    att2=Lambda(att_func)([att_tmp2,encode_smiles])
    encode_protein=Lambda(coeff_fun_prot)([att2,encode_protein])
    encode_smiles=Lambda(coeff_fun_lig)([att2,encode_smiles])
    
    
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    
    encode_protein1 = GlobalMaxPooling1D()(protein_lstm)
    encode_smiles1 = GlobalMaxPooling1D()(smile_lstm)
    
    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) 
    encode_interaction1 = keras.layers.concatenate([encode_smiles1, encode_protein1], axis=-1)  # skip connection
    encode_interaction =Add()([encode_interaction,encode_interaction1])

    # Fully connected 
    FC1 = Dense(1024, activation='relu', name='dense1')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu', name='dense2')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu', name='dense3')(FC2)
    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal', name='dense4')(FC2) # if you want train model for active/inactive set activation='sigmoid'

    model = Model(inputs=[Drug_input, Protein_input], outputs=[predictions])

    print(model.summary())

    return model



def crossfold_validation( flags):
    hyper_param_1 = flags.num_filters                 
    hyper_param_2 = flags.smiles_filter_lengths      
    hyper_param_3 = flags.protein_filter_lengths    
    hyper_param_4 = flags.learning_rate 
    hyper_param_5 = flags.embedding_size 
    epoch = flags.num_epochs       
    batchs_size = flags.batch_size    
    fold_nums=5
    params_list=[]
    
    print(flags.data_path)
    out_s=sio.loadmat(flags.data_path)
    train_folds_drugs, train_folds_proteins,  train_folds_affinity  = out_s['train_folds_drugs'], out_s['train_folds_proteins'], out_s['train_folds_affinity']
    test_folds_drugs, test_folds_proteins,  test_folds_affinity  = out_s['test_folds_drugs'], out_s['test_folds_proteins'], out_s['test_folds_affinity']
    val_folds_drugs, val_folds_proteins,  val_folds_affinity  = out_s['val_folds_drugs'], out_s['val_folds_proteins'], out_s['val_folds_affinity']
    
    h = len(hyper_param_1) * len(hyper_param_2) * len(hyper_param_3) * len(hyper_param_4) * len(hyper_param_5)

    cindex_perfs = [[0 for x in range(fold_nums)] for y in range(h)] 
    losses = [[0 for x in range(fold_nums)] for y in range(h)] 
    
    for fold in range(fold_nums):
    
        # train, val and test index
        train_drugs=np.array(train_folds_drugs[fold])
        train_proteins=np.array(train_folds_proteins[fold])
        train_affinity=np.array(train_folds_affinity[fold])
        
        val_drugs=np.array(val_folds_drugs[fold])
        val_proteins=np.array(val_folds_proteins[fold])
        val_affinity=np.array(val_folds_affinity[fold])
        
        test_drugs=np.array(test_folds_drugs[fold])
        test_proteins=np.array(test_folds_proteins[fold])
        test_affinity=np.array(test_folds_affinity[fold])
        
        params_list=[]
        iterator=0
        for ind_1 in range(len(hyper_param_1)): 
            num_filters = hyper_param_1[ind_1]
            for ind_2 in range(len(hyper_param_2)): 
                smiles_filter_lengths = hyper_param_2[ind_2]
                for ind_3 in range(len(hyper_param_3)):
                    protein_filter_lengths = hyper_param_3[ind_3]
                    for ind_4 in range(len(hyper_param_4)):
                         learning_rate = hyper_param_4[ind_4]
                         for ind_5 in range(len(hyper_param_5)):
                             embedding_size = hyper_param_5[ind_5]
                             
                             model = feature_extraction_model(flags, embedding_size, num_filters, protein_filter_lengths, smiles_filter_lengths)
                             adam=Adam(lr=learning_rate)
                             model.compile(optimizer=adam, loss='mean_squared_error', metrics=[cindex_score]) 
                             es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                             gridres = model.fit(([np.array(train_drugs),np.array(train_proteins) ]), np.array(train_affinity), batch_size=batchs_size, epochs=epoch, 
                             validation_data=( ([np.array(val_drugs), np.array(val_proteins) ]), np.array(val_affinity)),  shuffle=False, callbacks=[es] )
                             loss_mse, cindex_perf = model.evaluate(([np.array(val_drugs),np.array(val_proteins) ]), np.array(val_affinity), verbose=0)
                             
                             model.save_weights(flags.model_name+'.h5')

                             # Save the model architecture
                             with open(flags.model_name+'.json', 'w') as f:
                                 f.write(model.to_json())
                             losses[iterator][fold]=loss_mse
                             cindex_perfs[iterator][fold]=cindex_perf
                             iterator=iterator+1
                             param_list=[num_filters,smiles_filter_lengths,protein_filter_lengths,learning_rate,embedding_size];
                             params_list.append(param_list)
    mean_param_list=np.mean(np.array(cindex_perfs),axis=1)
    best_set_param_idx=np.argmax(mean_param_list)
    best_set_param=params_list[best_set_param_idx]
    
    # save the model
    
    model.save_weights(flags.model_name+'.h5')

    # Save the model architecture
    with open(flags.model_name+'.json', 'w') as f:
         f.write(model.to_json())
    
    return losses, cindex_perfs

    


def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select
       

def run_DeepCDA( flags ): 
    crossfold_validation( flags)

if __name__=="__main__":
    flags = argparser()
    run_DeepCDA( flags )

