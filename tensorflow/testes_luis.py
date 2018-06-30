# import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
# import transformations as trafo
data = np.array(pd.read_csv('voice_tiny_tiny.csv'))

LABEL_COL = 3

labels = data[:, [LABEL_COL]]
# print("LABELL")
# print(labels)

data2 = np.delete(data, LABEL_COL, axis=1)
# print("DATA")

# print(data2)
print(data2)
print("normalized")
# normalize_array =  map(normalize,data2)
# print(normalize_array)

X_normalized = preprocessing.normalize(data2, norm='l2')
print(X_normalized)
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