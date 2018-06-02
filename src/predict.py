

from keras.applications.vgg16 import decode_predictions
import numpy as np


#3.01e-9, 2.427e-11,... 
a = np.genfromtxt("/home/dcody/CS179homework/final/outputs/predictions.txt", delimiter = ',')
a = a[~np.isnan(a)]
print("Length of a is: ", len(a))


print(decode_predictions(np.expand_dims(a, axis = 0),top=5)[0])
print(decode_predictions(np.expand_dims(a[::-1], axis = 0),top=5)[0])