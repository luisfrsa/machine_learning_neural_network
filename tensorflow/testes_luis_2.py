import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
import sys


#python testes_luis.py 32,64,128,64,32,2 1000 5 mean_squared_error elu
#python testes_luis.py 32,32,2 100 5 mean_squared_error elu
param_Dense = sys.argv[1]
param_epochs = int(sys.argv[2])
param_batch_size = int(sys.argv[3])
param_loss = sys.argv[4]
param_activation = sys.argv[5]

split_dense = param_Dense.split(',')
split_dense_len = len(split_dense);

if(param_loss=='mean_squared_logarithmic_error'):
    param_loss_value = tf.keras.losses.mean_squared_logarithmic_error;
elif(param_loss=='mean_squared_error'):    
    param_loss_value = tf.keras.losses.mean_squared_error;
elif(param_loss=='squared_hinge'):    
    param_loss_value = tf.keras.losses.squared_hinge;

if(param_activation=='elu'):
    param_activation_value = tf.nn.elu;
elif(param_activation=='selu'):    
    param_activation_value = tf.nn.selu;    


LABEL_COL_AND_SHAPE = 20
MALE_ARRAY = [0,1]
FEMALE_ARRAY = [1,0]
        
loaded_csv = np.array(pd.read_csv('voice_tiny.csv'))

male_female_labels = np.array(loaded_csv[:, [LABEL_COL_AND_SHAPE]])
formated_labels = map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,male_female_labels)
data_without_gender = np.delete(loaded_csv, LABEL_COL_AND_SHAPE, axis=1)

normalized_layer = preprocessing.normalize(data_without_gender, norm='l2')


model = tf.keras.Sequential()
# for i in range(split_dense_len):
#     if(i == 0):
#         model.add(tf.keras.layers.Dense(int(split_dense[i]), activation=param_activation_value, input_shape=(LABEL_COL_AND_SHAPE,)))
#     elif(i == split_dense_len-1):
#         model.add(tf.keras.layers.Dense(int(split_dense[i]), activation='softmax'))
#     else:
#          model.add(tf.keras.layers.Dense(int(split_dense[i]), activation=param_activation_value))

model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu, input_shape=(20,)))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
# model.compile(
#     loss=param_loss_value,
#     optimizer=tf.keras.optimizers.SGD(1e-2),
#     metrics=['accuracy']
# )

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.SGD(1e-2),
    metrics=['accuracy']
)

# model.fit(
#     data_without_gender,
#     normalized_layer,
#     epochs=param_epochs,
#     shuffle=True,
#     batch_size=5
# )
model.fit(
    data_without_gender,
    normalized_layer,
    epochs=1000,
    shuffle=True,
    batch_size=5

)

model.save('ia.model')

