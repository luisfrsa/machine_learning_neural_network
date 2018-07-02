import sys
import tensorflow as tf
print(sys.argv)
#python a.py 32,64,128,64,32,2 SGD 1000 5 mean_squared_error elu
param_Dense = sys.argv[1]
param_optimizer = sys.argv[2]
param_epochs = sys.argv[3]
param_batch_size = sys.argv[4]
param_loss = sys.argv[5]
param_activation = sys.argv[5]


# print(param_Dense)
# print(param_optimizer)
# print(param_epochs)
# print(param_batch_size)
# print(param_loss)


split_dense = param_Dense.split(',')
split_dense_len = len(split_dense);

param_loss_value=''
param_activation_value = ''
if(param_optimizer=='mean_squared_logarithmic_error'):
    param_loss_value = tf.keras.losses.mean_squared_logarithmic_error;
elif(param_optimizer=='mean_squared_error'):    
    param_loss_value = tf.keras.losses.mean_squared_error;
elif(param_optimizer=='squared_hinge'):    
    param_loss_value = tf.keras.losses.squared_hinge;

if(param_activation=='elu'):
    param_activation_value = tf.tf.nn.elu;
elif(param_activation=='selu'):    
    param_activation_value = tf.tf.nn.selu;

print(param_loss_value)
print(param_activation_value)
print("aaaaa")


# for i in range(split_dense_len):
#     if(i == 0):
#         model.add(tf.keras.layers.Dense(split_dense[i], activation=param_activation_value, input_shape=(LABEL_COL_AND_SHAPE,)))
#     elif(i == split_dense_len-1):
#         model.add(tf.keras.layers.Dense(split_dense[i], activation='softmax'))
#     else:
#          model.add(tf.keras.layers.Dense(split_dense[i], activation=param_activation_value))