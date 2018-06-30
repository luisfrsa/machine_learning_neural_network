# import tensorflow as tf
import pandas as pd
import numpy as np

data = np.array(pd.read_csv('voice_tiny.csv'))

LABEL_COL = 20

labels = data[:, [LABEL_COL]]

print(labels)

data = np.delete(data, LABEL_COL, axis=1)

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