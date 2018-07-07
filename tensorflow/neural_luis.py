import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import sys

def get_param_rate(param_opt): 
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
    return param_opt_rate;

def get_param_loss(param_loss):
    if(param_loss=='mean_squared_logarithmic_error'):
        print("(Param loss escolhido: mean_squared_logarithmic_error");
        param_loss_value = tf.keras.losses.mean_squared_logarithmic_error;
    elif(param_loss=='mean_squared_error'):    
        print("(Param loss escolhido: mean_squared_error");
        param_loss_value = tf.keras.losses.mean_squared_error;
    elif(param_loss=='squared_hinge'):    
        print("(Param loss escolhido: squared_hinge");
        param_loss_value = tf.keras.losses.squared_hinge;
    return param_loss_value;

def get_param_optimizer(param_optimizer,param_opt_rate):
    if(param_optimizer=='sgd'):
        print("(Param optimizer escolhido: sgd")
        param_optimizer_value = tf.keras.optimizers.SGD(param_opt_rate);
    elif(param_optimizer=='rms'):  
        print("(Param optimizer escolhido: rms")
        param_optimizer_value = tf.keras.optimizers.RMSprop(param_opt_rate);
    elif(param_optimizer=='adadelta'):  
        print("(Param optimizer escolhido: adadelta");
        param_optimizer_value = tf.keras.optimizers.Adadelta(param_opt_rate);
    return param_optimizer_value;

def get_param_activation(param_activation):
    if(param_activation=='elu'):
        print("(Param activation escolhido: elu");
        param_activation_value = tf.nn.elu;
    elif(param_activation=='selu'):    
        print("(Param activation escolhido: selu");
        param_activation_value = tf.nn.selu;
    elif(param_activation=='crelu'):    
        print("(Param activation escolhido: crelu");
        param_activation_value = tf.nn.crelu; 
    return param_activation_value;

def get_param_new_value(param_new):    
    if(param_new=='1'):
        param_new_value = 1;
    elif(param_new=='0'):
        param_new_value = 0;
    return param_new_value;

def get_values(csvFile):  
    loaded_csv = np.array(pd.read_csv(csvFile))
    string_labels = np.array(loaded_csv[:, [LABEL_COL_AND_SHAPE]])
    formated_labels = np.array(map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,string_labels))
    loaded_without_label = np.delete(loaded_csv, LABEL_COL_AND_SHAPE, axis=1)
    #normalized_data = preprocessing.normalize(loaded_without_label, norm='l2')
    normalized_data = preprocessing.MinMaxScaler().fit_transform(loaded_without_label)
    return normalized_data,formated_labels;

param_dense = sys.argv[1]
param_epochs = int(sys.argv[2]) 
param_batch_size = int(sys.argv[3])
param_loss = sys.argv[4] 
param_activation = sys.argv[5]
param_optimizer = sys.argv[6]
param_new = sys.argv[7]
param_opt = sys.argv[8]
  
     
LABEL_COL_AND_SHAPE = 20
MALE_ARRAY = [0,1]
FEMALE_ARRAY = [1,0]

param_opt_value = get_param_rate(param_opt);
param_loss_value = get_param_loss(param_loss);
param_optimizer_value = get_param_optimizer(param_optimizer,param_opt_value);
param_activation_value = get_param_activation(param_activation);
param_new_value = get_param_new_value(param_new);


filename="model/result.model"

train_normalized_data,train_formated_labels  = get_values('train_voice.csv')
test_normalized_data,test_formated_labels  = get_values('test_voice.csv')

split_dense = param_dense.split(',')
split_dense_len = len(split_dense);

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
   


