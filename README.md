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
what enables the layer features to encode certain information about the image. VGG16 is a
particular neural network topology which consists of 16 convolutional/dense layers with varying
numbers of parameters. The chief difficulty of this part of the project was obtaining and using
the network weights provided by the network designers.

### Loss Metrics

As discussed previously, although many neural networks use a single loss function, this
used multiple loss functions which are evaluated a multiple layers throughout the back
propagation process. In each of these pre-defined loss-layers, we store a matrix of elements
which represent the feature outputs from either the content or the style images. For example,
in the layer named "block5conv2" we store a n * c * w * h-sized array which holds the content
image's feature outputs. Or in the layer "block5conv1", we store an n * n array which holds
the gram matrix of the feature outputs for the style image.

To implement these, I defined a flag variable (bool style_transfer) which told the program
whether to save the feature output to the loss_metrics variable, to save the gram matrix of
the feature output to the loss_metrics variable, to evaluate the style/content loss during
backpropagation, or to just run and make predictions normally.

### Gradient Descent

This ended up being easier than expected. All I had to do was alter the backward_propagation
function for the input to push the gradient all the way to the input image. Lastly, by making
small adjustments to the train_on_batch() function, and calling it multiple times I could
make many updates to the image.

## Results

### implementation of VGG16

One of the main objectives of this project was to implement VGG16 using the infrastructure that
was give to us during labs 5 and 6. Although I was able to sucessfully steal the weight matricies
from python keras, store them in a reasonable format (.h5), and upload them sucessfully to the
weights matricies as defined in the layers, getting reasonable predictions out of the CNN was 
still very difficult. Ultimately, the error is likely a data formatting issue somewhere a long
the road. 

Code highlights:
1. src/h5_utils.cpp
	- C++ scripts that were developed to read in the .h5 files into arrays before they were
	  cudaMalloc'ed to their proper places in the weights matricies. 
2. src/layers.cpp
	- Extended the Conv2D class to accept a padding parameter as required by the VGG16 CNN.
		- line
	- Extended the Layer::init_weights_biases() funciton to allow for random or initialized
	  weight matricies. Utilized the custom functions written in h5_utils.cpp to initialize
	  the matricies with previously computed values. cudaMalloc'd the .h5 values into the
	  appropriate weight/bias matricies.
	 	- line ~ 244
3. src/main.cpp
	- implemented all 16 layers of VGG16
		- line ~ 71
4. src/model.cpp
	- Extended Model::add so that every layer will keep track of a layer_name to identify
	  itself with. 
	  	- line ~ 60

2. vgg16/vgg.py
	- python code which implements the vgg neural network.
	- allows us to save the vgg16 neural network weights into an .h5 data format
3. vgg16/rewrite.py
	- a partial rewriting of the keras style transfer code.
	- submitted as the cpu implementation


### Checking functions

Often, I would want to check the values of certain arrays on the GPU to make sure the values
where being written correctly to memory, so I wrote several utilities to 

1. src/model.cpp
	- 
### Loss functions

### Utility code


## Discussion

### Sucesses

If there's anything I've learnt

### Failures

### Future Work
