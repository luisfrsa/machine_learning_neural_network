import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense
import sys


#python testes_luis.py 32,64,128,64,32,2 1000 5 mean_squared_error elu sgd
#python testes_luis.py 32,32,2 10 50 mean_squared_error elu sgd


param_Dense = sys.argv[1]
# comeca baixo,e aumenta
param_epochs = int(sys.argv[2]) 
# comeca alto e diminuiria
param_batch_size = int(sys.argv[3])
param_loss = sys.argv[4] 
param_activation = sys.argv[5]
param_optimizer = sys.argv[6] # comeca alto e diminui inicia com 1e-2, 1e-3, 1e-4
param_new = sys.argv[7]
param_opt = sys.argv[8]
param_finish = sys.argv[9]

if(param_opt=="1"):
    param_opt_rate = 1e-1;
elif(param_opt=="2"):
    param_opt_rate = 1e-2;     
elif(param_opt=="3"):
    param_opt_rate = 1e-3;
elif(param_opt=="4"):
    param_opt_rate = 1e-4;
elif(param_opt=="5"):
    param_opt_rate = 5e-5;
elif(param_opt=="6"):
    param_opt_rate = 1e-5;    
       


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
    param_optimizer_value = tf.keras.optimizers.SGD(param_opt_rate)
elif(param_optimizer=='rms'):  
    print("(Param optimizer escolhido: rms")
    param_optimizer_value = tf.keras.optimizers.RMSprop(param_opt_rate)
elif(param_optimizer=='adadelta'):  
    print("(Param optimizer escolhido: adadelta")
    param_optimizer_value = tf.keras.optimizers.Adadelta(param_opt_rate)

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

filename="model/result.model"

split_dense = param_Dense.split(',')
split_dense_len = len(split_dense);

# import transformations as trafo

LABEL_COL_AND_SHAPE = 20
MALE_ARRAY = [0,1]
FEMALE_ARRAY = [1,0]


# loaded_csv = np.array(pd.read_csv('voice_tiny.csv'))
train_loaded_csv = np.array(pd.read_csv('train_voice.csv'))
train_string_labels = np.array(train_loaded_csv[:, [LABEL_COL_AND_SHAPE]])
train_formated_labels = np.array(map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,train_string_labels))
train_loaded_without_label = np.delete(train_loaded_csv, LABEL_COL_AND_SHAPE, axis=1)
train_normalized_data = preprocessing.normalize(train_loaded_without_label, norm='l2')

test_loaded_csv = np.array(pd.read_csv('test_voice.csv'))
test_string_labels = np.array(test_loaded_csv[:, [LABEL_COL_AND_SHAPE]])
test_formated_labels = np.array(map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,test_string_labels))
test_loaded_without_label = np.delete(test_loaded_csv, LABEL_COL_AND_SHAPE, axis=1)
test_normalized_data = preprocessing.normalize(test_loaded_without_label, norm='l2')

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
    model = tf.keras.models.load_model(filename)       


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
if(param_finish=="0"):
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/' +filename, histogram_freq=0, write_graph=True, write_images=False)
    model.fit(
        train_normalized_data,
        train_formated_labels,
        epochs=param_epochs,
        shuffle=True,
        batch_size=param_batch_size,
        callbacks=[tensorboard],
        validation_data=(test_normalized_data,test_formated_labels)
    )
    model.save(filename)
   


