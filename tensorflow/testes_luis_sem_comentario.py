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


loaded_csv = np.array(pd.read_csv('voice_tiny.csv'))
string_labels = np.array(loaded_csv[:, [LABEL_COL_AND_SHAPE]])
formated_labels = np.array(map(lambda label:  MALE_ARRAY if label=="male" else FEMALE_ARRAY,string_labels))
loaded_without_label = np.delete(loaded_csv, LABEL_COL_AND_SHAPE, axis=1)

normalized_data = preprocessing.normalize(loaded_without_label, norm='l2')



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
    normalized_data,
    formated_labels,
    epochs=100,
    shuffle=True,
    batch_size=5

)

model.save('ia.model')

