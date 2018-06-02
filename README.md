README

# A GPU implemention of the neural Algorithm of Artistic Style

## Overview of the Algorithm

The Neural Algorithm of Artistic Style, first publicized by Gatys et al. in their 2015 paper,
is an incredibly powerful method for combining the style of one image onto the content of
another. The fundamental assumption of the theory is that while the content of an image is
preserved in the layer features of a convolutional neural network, the "style" or something
similar to style is encoded in the gram matrix of the feature layers. To create a "combined"
image we start with a style image, a content image, and a "white noise" random image. Then
define loss functions at various feature layers which penalize the random image if the mean
squared error between layer output and the content image's output. Similarly, the random
image is again penalized if the gram matrix of the layer output of the random image is
different from the output of the "style" image. To make the "random" image look more like 
the style and content images, we can backpropagate the error to change the image. This process
is repeated many times until the random noise is turned into an image.

In relation to our class, convolutional neural networks have seen great speed-ups by executing
on GPUs. Since this algorithm uses a convolutional neural network, the algorithm will benefit
greatly from implementation on a GPU.

## Theory of Implementation

To implement this algorithm, we essentially need to do 3 things:

1. Obtain and implement a working convolutional neural network. Here we use VGG16.
2. Implement the necessary loss functions.
3. Run gradient descent on the image until it converges

### VGG16

VGG16 is an award-winning convolutional neural network that has been train to identitfy 1000
classes of objects. The fact that it was trained is vital to our algorithm since training is
what enables the layer features to encode certain information about the image. VGG

### Loss Metrics

### Gradient Descent

## Results

### implementation of VGG16

One of the main objectives of this project was to implement VGG16 using the infrastructure that
was give to us during labs 5 and 6. Although I was able to sucessfully steal the weight matricies
from python keras, and 

### Checking functions

### Loss functions

### Utility code


## Discussion

### Sucesses

### Failures

### Future Work
