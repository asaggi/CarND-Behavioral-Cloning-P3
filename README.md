## Behavioral Cloning

[//]: # (Image References)

[image1]: ./images/augment.jpg "Augmentation"
[image2]: ./images/crop.jpg "Cropping"
[image3]: ./images/preprocess.jpg "Preprocessing"
[image4]: ./images/randomBrightness.jpg "Random Brightness"
[image5]: ./images/randomFlip.jpg "Random Flip"
[image6]: ./images/randomTranslate.jpg "Random Translate"
[image7]: ./examples/placeholder_small.png "Flipped Image"

The goals / steps of this project are the following:   
* Use the simulator to collect data of good driving behavior.  
* Build, a convolution neural network in Keras that predicts steering angles from images.  
* Train and validate the model with a training and validation set.  
* Test that the model successfully drives around track one without leaving the road.  
* Summarize the results with a written report.  

In this project, I used the default images provided for training as my data set.
I used the NVIDIA model, as it's supposed to work well in this situation.

## Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

## Data Preprocessing

* Normalization of Images (To make gradients work better)
* The images are cropped so that the model won’t be trained with the sky and the car front parts, here is an example of a cropped image (60:25):

	![alt text][image2]

## Model Training
I used the following augumentation technique to generate unlimited number of images:

* Left image, steering angle is adjusted by +0.2
* Right image, steering angle is adjusted by -0.2
* Randomly flip image left/right, here is an example of a flipped image

	![alt text][image5]

* Randomly choose left, right or center images.
* Randomly altering image brightness (lighter or darker), here is an example of a an image with reduced brightness:    

	![alt text][image4]

## Training, Validation and Test
I splitted the images into train and validation set in order to measure the performance at every epoch. Testing was done using the simulator.

* I chose MSE for the loss function to measure how close the model predicts to the given steering angle for each image.
* I used Adam optimizer for optimization with learning rate of 1.0e-4 which is smaller than the default of 1.0e-3. 
* I used ModelCheckpoint from Keras to save the model.

## Submission
**My project includes the following files:**.   
- model.py containing the script to create and train the model.   
- drive.py for driving the car in autonomous mode.   
- model.h5 containing a trained convolution neural network   
- writeup_report.md summarizing the results.  
- preprocess.py, helper functions for model.py

## Architecture

### An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20). 

### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 22). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 32-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 39).

## Final Model Architecture

The design of the network is based on the NVIDIA model.

I've added the following adjustments to the model.

* Used Lambda layer to normalized input images to avoid saturation and make gradients work better.
* Added an additional dropout layer to avoid overfitting after the convolution layers.
* Include  ELU for activation function for every layer except for the output layer to introduce non-linearity.  

The model looks like as follows: 
---  
*  Image normalization.  
*  Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU.  
*  Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU.  
*  Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU.  
*  Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU    
*  Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU     
*  Drop out (0.5)
*  Fully connected: neurons: 100, activation: ELU.  
*  Fully connected: neurons: 50, activation: ELU.  
*  Fully connected: neurons: 10, activation: ELU.  
*  Fully connected: neurons: 1 (output).  


## OUTPUT

The model can drive the course without bumping into the side ways.


## References
**NVIDIA**:   [Self Driving Car] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)