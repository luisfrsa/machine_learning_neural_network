import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
#pip install keras
from keras.models import Sequential
from keras.layers import Dense
import sys

print(sys.argv)


# import transformations as trafo

LABEL_COL_AND_SHAPE = 20
MALE_ARRAY = [0,1]
FEMALE_ARRAY = [1,0]


a = np.array(pd.read_csv('voice_tiny.csv'))
labels = a[:, [LABEL_COL_AND_SHAPE]]
a = np.delete(a, LABEL_COL_AND_SHAPE, axis=1)

y = []
for label in labels:
    if label == 'male':
        y.append(MALE_ARRAY)
    else:
        y.append(FEMALE_ARRAY)
b =  np.array(y)
        
data = np.array(pd.read_csv('voice_tiny.csv'))
# np.random.shuffle(data)
# print("SUFFLE")
# print(data)


labels = np.array(data[:, [LABEL_COL_AND_SHAPE]])
formated_labels = map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,labels)
data2 = np.delete(data, LABEL_COL_AND_SHAPE, axis=1)
# normalize_array =  map(normalize,data2)
# print(normalize_array)

X_normalized = preprocessing.normalize(data2, norm='l2')

c = X_normalized
d = np.array(formated_labels)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu, input_shape=(20,)))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
# model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.SGD(1e-2),
    metrics=['accuracy']
)

model.fit(
    a,
    d,
    epochs=100,
    shuffle=True,
    batch_size=5

)

model.save('ia.model')


# model = Sequential()

# model.add(Dense(units=64, activation='relu',  input_shape=(LABEL_COL_AND_SHAPE,)))
# model.add(Dense(units=10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
           
# model.fit(X_normalized, formated_labels, epochs=5, batch_size=5)

# model.save('ia.model')


# normed_matrix = normalize(data2)
# max_value = data2.max()
# normalized = map(lambda array:map(lambda elem:elem+99,array),data2)
# normalize_array =  map(lambda array:math.sqrt(reduce(lambda total,elem:math.sqrt(elem)+total, array)),data2)
# print(normalize_array)
                
# normalized = map((lambda array:
#                 euclidian = math.sqrt(reduce(lambda elem,accum:accum+math.sqrt(elem), array))
#                 # return 1
#                 return map(lambda elem:
#                     elem/euclidian
#                 ,array)
                
#             ,data2))
# normalized = map(lambda elem:elem/max_value, data2)
# normalized = map(map(lambda x:x/max_value, data2))
# print(normalized)



# print(trafo.unit_vector(data2, axis=1))
# max = (map(max,map(max, data)))
# print (max)
#a if condition else b
#m = map(max,(map(lambda elem: 0 if isinstance(elem, basestring) else elem , data)))
# m = map(max,(map(lambda elem: 0 if isinstance(elem, basestring) else elem , data)))


# flat = data.flatten()
# m = max(flat)
# print ("Full---------")
# print(data)
# print("Flat---------")
# print(flat)
# print ("M---------")
# print(m)


#isinstance(o, basestring)
# max = print(map(list, data))
# a = ["foo", "bar", "baz"]
# c = (list(map(list, a)))
# print (c)

# (lambda x: (x%2 == 0) , my_list)