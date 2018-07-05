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

if(param_opt=="2"):
    param_opt_rate = 1e-2;
elif(param_opt=="3"):
    param_opt_rate = 1e-3;
elif(param_opt=="4"):
    param_opt_rate = 1e-4;


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
        normalized_data,
        formated_labels,
        epochs=param_epochs,
        shuffle=True,
        batch_size=param_batch_size,
        callbacks=[tensorboard],
        # validation_data=(test_x,test_y)
    )
    model.save(filename)
else:
    model.predict(normalized_data)


