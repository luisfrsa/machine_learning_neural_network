import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
#pip install keras
from keras.models import Sequential
from keras.layers import Dense
import sys


#python testes_luis.py 32,64,128,64,32,2 1000 5 mean_squared_error elu sgd
#python testes_luis.py 32,32,2 10 50 mean_squared_error elu sgd


param_Dense = sys.argv[1]
param_epochs = int(sys.argv[2])
param_batch_size = int(sys.argv[3])
param_loss = sys.argv[4]
param_activation = sys.argv[5]
param_optimizer = sys.argv[6]
param_new = sys.argv[7]

if(param_loss=='mean_squared_logarithmic_error'):
    print("(Param loss escolhido: mean_squared_logarithmic_error")
    param_loss_value = tf.keras.losses.mean_squared_logarithmic_error;
elif(param_loss=='mean_squared_error'):    
    print("(Param loss escolhido: mean_squared_error")
    param_loss_value = tf.keras.losses.mean_squared_error;
elif(param_loss=='squared_hinge'):    
    print("(Param loss escolhido: squared_hinge")
    param_loss_value = tf.keras.losses.squared_hinge;

if(param_optimizer=='sgd'):
    print("(Param optimizer escolhido: sgd")
    param_optimizer_value = tf.keras.optimizers.SGD(1e-2)
elif(param_optimizer=='rms'):  
    print("(Param optimizer escolhido: rms")
    param_optimizer_value = tf.keras.optimizers.RMSprop(1e-2)
elif(param_optimizer=='adadelta'):  
    print("(Param optimizer escolhido: adadelta")
    param_optimizer_value = tf.keras.optimizers.Adadelta(1e-2)

if(param_activation=='elu'):
    print("(Param activation escolhido: elu")
    param_activation_value = tf.nn.elu
elif(param_activation=='selu'):    
    print("(Param activation escolhido: selu")
    param_activation_value = tf.nn.selu   
elif(param_activation=='crelu'):    
    print("(Param activation escolhido: crelu")
    param_activation_value = tf.nn.crelu     
    
if(param_new=='1'):
   param_new_value = 1
elif(param_new=='0'):
    param_new_value = 0

filename="model/result_"+str(param_Dense)+"_"+ str(param_epochs)+"_20_"+ str(param_loss)+"_"+ str(param_activation)+"_"+str(param_optimizer)+"_1.model"

split_dense = param_Dense.split(',')
split_dense_len = len(split_dense);

# import transformations as trafo

LABEL_COL_AND_SHAPE = 20
MALE_ARRAY = [0,1]
FEMALE_ARRAY = [1,0]


# loaded_csv = np.array(pd.read_csv('voice_tiny.csv'))
loaded_csv = np.array(pd.read_csv('voice.csv'))
string_labels = np.array(loaded_csv[:, [LABEL_COL_AND_SHAPE]])
formated_labels = np.array(map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,string_labels))
loaded_without_label = np.delete(loaded_csv, LABEL_COL_AND_SHAPE, axis=1)

normalized_data = preprocessing.normalize(loaded_without_label, norm='l2')

if(param_new_value==1):
    model = tf.keras.Sequential()
    for i in range(split_dense_len):
        if(i == 0):
            model.add(tf.keras.layers.Dense(int(split_dense[i]), activation=param_activation_value, input_shape=(LABEL_COL_AND_SHAPE,)))
        elif(i == split_dense_len-1):
            model.add(tf.keras.layers.Dense(int(split_dense[i]), activation='softmax'))
        else:
            model.add(tf.keras.layers.Dense(int(split_dense[i]), activation=param_activation_value))
else:    
    model = tf.keras.models.load_model(filename)#= keras_model()        


# model.add(tf.keras.layers.Dense(32, activation=param_activation_value, input_shape=(20,)))
# model.add(tf.keras.layers.Dense(64, activation=param_activation_value))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(
    loss=param_loss_value,
    optimizer=param_optimizer_value,
    metrics=['accuracy']
)

# model.compile(
#     loss=param_loss_value,
#     optimizer=tf.keras.optimizers.SGD(1e-2),
#     metrics=['accuracy']
# )

model.fit(
    normalized_data,
    formated_labels,
    epochs=param_epochs,
    shuffle=True,
    batch_size=param_batch_size

)

model.save(filename)

