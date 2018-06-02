import numpy as np
from PIL import Image

from keras import backend
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing import image

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


sz = 224

a = np.genfromtxt("elephant.txt", delimiter = ',')
a = a[~np.isnan(a)]
a = np.reshape(a, (3, sz, sz))

a = np.swapaxes(a,2,1)
a = np.swapaxes(a,2,0)
imsave( "image.jpeg", a)
x = np.expand_dims(a, axis = 0)
print("Length of a is: ", len(a))


model = VGG16(weights='imagenet')
preds = model.predict(x)
print(preds)
print('Predicted:', decode_predictions(preds, top=5)[0])


#EXPORT WEIGHTS
model.save_weights("weights.h5")










