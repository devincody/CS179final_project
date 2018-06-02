from keras.preprocessing.image import load_img, img_to_array#, save_img
import numpy as np
from scipy.misc import imsave

from keras.applications import vgg16

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    # if K.image_data_format() == 'channels_first':
    x = x.reshape((3, img_nrows, img_ncols))
    x = x.transpose((1, 2, 0))
    # else:
    #     x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


base_image_path = "HawaiiResized.jpg"
style_reference_image_path = "style2.jpg"
combo_image_path = "combo.jpg"

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 224#img_nrows
img_ncols = 224#int(width * img_nrows / height)


## ID
a = preprocess_image("elephant.png")
f = open("images/elephant.txt", 'w+')
for i in range(3):                                    
    for j in range(img_nrows):                              
            for k in range(img_ncols):   #rows first (i.e. row major order)                   
                f.write(str(a[0][j][k][i]) + "\n")
f.close()

## CONTENT
a = preprocess_image(base_image_path)
f = open("images/content.txt", 'w+')
for i in range(3):                                    
    for j in range(img_nrows):                              
            for k in range(img_ncols):                      
                f.write(str(a[0][j][k][i]) + "\n")
f.close()

## STYLE
a = preprocess_image(style_reference_image_path)
f = open("images/style.txt", 'w+')
for i in range(3):                                    
    for j in range(img_nrows):                              
            for k in range(img_ncols):                      
                f.write(str(a[0][j][k][i]) + "\n")
f.close()

## COMBINATION
a = preprocess_image("rand.jpg")
f = open("images/rand.txt", 'w+')
for i in range(3):                                    
    for j in range(img_nrows):                              
            for k in range(img_ncols):                      
                f.write(str(a[0][j][k][i]) + "\n")
f.close()


