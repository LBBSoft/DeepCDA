import os
import csv
import gzip
import pprint
import collections
import operator 
import pandas as pd

import keras
from keras.layers import Conv2D, MaxPooling2D,Multiply, Flatten, Dense, Input,Dropout,LSTM, Reshape, RepeatVector, Permute,Concatenate
from keras.models import Model,clone_model
import numpy as np
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras import regularizers
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from zipfile import ZipFile 

import numpy as np
import math
import scipy.io as sio
import pickle
import math
import sklearn

import argparse



def argparser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--source_data_path',
      type=str,
      default='Davis_Dataset_folded.mat',
      help='Path to the dataset (a .mat file)'
  )
  parser.add_argument(
      '--target_data_path',
      type=str,
      default='Davis_Dataset_folded.mat',
      help='Path to the dataset (a .mat file)'
  )
  parser.add_argument(
      '--num_classification_layers',
      type=int,
      default=100,
      help='Number of classification layers in feature encoder training model.'
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='model',
      help='model_name'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=30,
      help='number of epochs'
  )

  flags, unparsed = parser.parse_known_args()

  return flags



def generate_data_our(XD,XP,batch_size):
    i = 0
    

    while True:
        input1 = []
        input2=[]
        output=[]
        batch_counter=0
        while batch_counter<batch_size:
            if i == len(XD):
                i = 0
            
            batch_counter += 1
                    
            input1.append(XD[i])
            input2.append(XP[i])
                    
            i += 1
        input1=np.array(input1)
        input2=np.array(input2)
        yield [np.array(input1) ,np.array(input2)]
def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select
    
def DomainAdaptation(flags):

    # load source dataset (training dataset)
    out_s=sio.loadmat(flags.source_data_path)
    train_drugs, train_prots,  train_Y  = out_s['XD_train'], out_s['XT_train'], out_s['y_train']
    train_Y =train_Y [0]

    # load target dataset (test dataset)    
    out_s=sio.loadmat(flags.target_data_path)
    val_drugs, val_prots,  val_Y= out_s['XD_train'], out_s['XT_train'], out_s['y_train']
    val_Y= val_Y[0]
      

    # Load the saved architecture 
    with open(flags.model_name+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights (the model was leaned on training data)
    model.load_weights(flags.model_name+'.h5')
    
    adam=Adam(lr=0.0001 )


    model.compile(optimizer=adam,loss='mse',metrics=['mse',cindex_score])

    Source_encoder = Model(inputs=model.input , outputs=model.layers[-1*flags.num_classification_layers].output )
    print(len(model.layers))
    for i in range(len(model.layers)-flags.num_classification_layers):
        wei=model.layers[i].get_weights()
        print(i)
        Source_encoder.layers[i].set_weights(wei)

    Target_encoder= clone_model(Source_encoder)
    Target_encoder.set_weights(Source_encoder.get_weights())
    Target_encoder.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])

    #design the discriminator of domain adaptation network

    Discrim_layer1 = Dense(128, activation='relu',kernel_initializer='normal')
    dout1=Dropout(0.3,name='dout1')
    Discrim_layer2 = Dense(128, activation='relu',kernel_initializer='normal')#,kernel_regularizer=regularizers.l2(0.01))
    dout2=Dropout(0.3,name='dout2')
    Discrim_output = Dense(1, activation='sigmoid',kernel_initializer='normal')#,kernel_regularizer=regularizers.l2(0.01))
     
    #train the domain adaptation network 
    Discrim_layer_one_T= Discrim_layer1(Target_encoder.output)
    dout1_out=dout1(Discrim_layer_one_T)
    Discrim_layer_two_T=Discrim_layer2(dout1_out)
    dout2_out=dout2(Discrim_layer_two_T)
    Discrim_layer_output_T=Discrim_output(dout2_out)

    adam=Adam(lr=0.00001 )   # find the appropriate learning rate

    net1=Model(inputs=Target_encoder.input, outputs=Discrim_layer_output_T)
    #net1.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])
    for i in range(len(Target_encoder.layers)):
        wei=Target_encoder.layers[i].get_weights()
        net1.layers[i].set_weights(wei)


    for layer in net1.layers[:-1*flags.num_classification_layers]:
        layer.trainable=False
    net1.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])
    #net1.layers[-1].trainable=True
    #net1.layers[-2].trainable=True
    #net1.layers[-3].trainable=True

    input_mapp=Input(shape=(1024,),name='input_map')
    Discrim_layer_one_S= Discrim_layer1(input_mapp)
    dout1_out_s=dout1(Discrim_layer_one_S)
    Discrim_layer_two_S=Discrim_layer2(dout1_out_s)
    dout2_out_s=dout2(Discrim_layer_two_S)
    Discrim_layer_output_S=Discrim_output(dout2_out_s)

    net2=Model(inputs=input_mapp, outputs=Discrim_layer_output_S)
    #net2.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy')
    adam1=Adam(lr=0.00004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)  # find the appropriate learning rate
    net2.compile(loss="binary_crossentropy",optimizer=adam1,metrics=['accuracy'])


    predicted_ic50_target = Dense(1, activation='sigmoid')(Target_encoder.output)
    wei=model.layers[-1].get_weights()

    epoches=1
    batch_size=256
    #print(net3.summary())
    for i in range(len(net1.layers)-5):
        wei=net1.layers[i].get_weights()
        model.layers[i].set_weights(wei)

    #print(model.evaluate_generator(generate_data(bindingdb_generator_test,batch_size),steps = len(bindingdb_generator_test)/batch_size))
    #print(model.evaluate_generator(generate_data(bindingdb_generator_test,batch_size),steps = len(bindingdb_generator_test)/batch_size))
    #print(model.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), batch_size=256))

    from scipy.spatial import distance
            

    print(model.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), batch_size=256))

    for iteration in range(flags.num_epochs):
        count_btch=0
        for stepp,(source,target) in  enumerate(zip(generate_data_our(train_drugs,train_prots,1024), generate_data_our(val_drugs,val_prots,1024))):
             #print(source)
             #print(type(source))
             #print(len(source[1]))i
             print(stepp)
             if iteration==1 & stepp>10:
                 adam=Adam(lr=0.000001 )
             #tf.keras.backend.clear_session()
             for i in range(len(model.layers)-flags.num_classification_layers):
                 wei=model.layers[i].get_weights()
                 Target_encoder.layers[i].set_weights(wei)
             Target_encoder.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])
             feat_source=Source_encoder.predict(source, batch_size=256)
             feat_target=Target_encoder.predict(target, batch_size=256)
             ddist=distance.cdist(feat_source,feat_target,'cosine')
             tmp_samp=np.amax(np.eye(1024)-(1-ddist),axis=1)
             tmp_samp=tmp_samp.reshape((feat_source.shape[0],1))   
             print(np.amax(tmp_samp))      
             print(np.amin(tmp_samp))
             
             
             lbl_source=np.ones((feat_source.shape[0],1))
             lbl_target=np.zeros((feat_target.shape[0],1))
             feat_all=np.concatenate((feat_source,feat_target), axis=0)
             lbl_all=np.concatenate((lbl_source,lbl_target), axis=0)
             
             # apply some noise on labels
             chosen_idx=np.random.choice(int(feat_source.shape[0]+feat_target.shape[0]), int(np.floor(0.05*(feat_source.shape[0]+feat_target.shape[0]))), replace=False)
             for k_idx in range(len(chosen_idx)):
                 if lbl_all[chosen_idx[k_idx]]==0:
                    lbl_all[chosen_idx[k_idx]]=1 
                 else:
                    lbl_all[chosen_idx[k_idx]]=0
             # make soft labels        
             chosen_idx=np.random.choice(int(feat_source.shape[0]+feat_target.shape[0]), int(np.floor(0.5*(feat_source.shape[0]+feat_target.shape[0]))), replace=False)
             for k_idx in range(len(chosen_idx)):
                 if lbl_all[chosen_idx[k_idx]]==0:
                    lbl_all[chosen_idx[k_idx]]=0.1
                 else:
                    lbl_all[chosen_idx[k_idx]]=0.9
             tmp_samp = np.exp(tmp_samp)/sum(np.exp(tmp_samp))
             sample_weight=np.concatenate((tmp_samp,np.exp(lbl_source)/sum(np.exp(lbl_source))),axis=0)
             sample_weight=sample_weight.flatten()
            
             net2.fit(x=feat_all,y=lbl_all,batch_size=1024,epochs=2,sample_weight=sample_weight,shuffle=True)
             for layer in net1.layers:
                 layer.trainable=True
             for ind in range(flags.num_classification_layers):
                 net1.layers[-1*(ind+1)].trainable=False
             net1.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])
             net1.fit(target, np.ones((np.array(target[0]).shape[0],1)), batch_size=256,epochs =  2)
             for i in range(len(net1.layers)-flags.num_classification_layers):
                  wei=net1.layers[i].get_weights()
                  model.layers[i].set_weights(wei)
             if stepp%10==0:
                  print(model.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), batch_size=256))
             #   Save the model weights                    
             model.save_weights('DeepCDA_weights'+str(iteration)+'.h5')

             # Save the model architecture
             with open('DeepCDA_architecture'+str(iteration)+'.json', 'w') as f:
                  f.write(model.to_json())


if __name__=="__main__":
    flags = argparser()
    DomainAdaptation(flags)