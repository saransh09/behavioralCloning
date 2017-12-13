# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 - contains the video of one lap of autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

> I have first used the Lambda layer to Normalize the images, this is so that random brightness or color patters do not affect the training process
> Then I have cropped the images, from the top and the bottom, so that the training process only focuses on the road and not any other random noise
> Following that I have used the Nvidia's model for Self Driving Cars
> This was the architecture that I use
  1) Convolution Layer - 24 kernels of 5x5
  2) ELU activation layer
  3) Convolution Layer - 36 filters of 5x5
  4) ELU activation layer
  5) Convolution Layer - 48 filters of 5x5
  6) ELU Activation Layer
  7) Flatten Layer
  8) Densely connected layer with 100 output channel
  9) ELU Activation layer
  10) Densely connected layer with 50 output channel
  11) ELU Activation layer
  12) Densely connected layer with one output - This is our actual output
  
#### 2. Attempts to reduce overfitting in the model

> In the beginning itself, I have subsample the training and the validation data, so that we can use it to tune the hyperparameters in the model

> Custom generator functions were written and the generated data was shuffled adequately so that it does not induce any bias

> Also, I have used the l2 regularization technique in the convolution layers and the densely connected layers so that I could avoid overfitting

> The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

> The model used an adam optimizer was used in the model

> I used it to generate 4*(total_training_size) of the images for each epoch of training

> the l2_regularization_constant was set at 0.001

> I used the MSE (Mean squared error) loss

#### 4. Appropriate training data

For training, I used a specific technique, 

#Track1
1) Counter Clockwise - One Lap of Smooth driving, one lap of corener recovery

2) Clockwise - One Lap of Smooth Driving

#Track2

1) One lap of counter clockwise smooth driving

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the approach somewhat similar to what was shown in the videos,

> I first trained on a single lap, and started by just using one dense layer

> From that I experimented by adding more Dense Layers

> Running preprocessing steps like normalization, cropping

> Then using a simple one convolution layer followed by two Dense Layer network

> Finally arriving to the Nvidia architecture, which actually yielded the best results

#### 2. Final Model Architecture

> This was the final architecture that I used:
  1) Convolution Layer - 24 kernels of 5x5
  2) ELU activation layer
  3) Convolution Layer - 36 filters of 5x5
  4) ELU activation layer
  5) Convolution Layer - 48 filters of 5x5
  6) ELU Activation Layer
  7) Flatten Layer
  8) Densely connected layer with 100 output channel
  9) ELU Activation layer
  10) Densely connected layer with 50 output channel
  11) ELU Activation layer
  12) Densely connected layer with one output - This is our actual output

I added the ELU activation layers as suggested to me in the evaluation of the previous project

#### 3. Creation of the Training Set & Training Process

> So as the dataset was of images, I initially used just the center images, and the results were pretty much evident of the fact that there was a need of using more images and if needed creating more images

> This is exactly what I did, I used all three of the center, left and the right images to train. Added to that I created one augmented image per three images of a single time instance, by horizontle flipping the center image
This was done primarily so that I decrease the bias induced due to my direction of driving on the track (clockwise or counter clockwise)

> Then I normalized the images and mean centered them, so that they are no longer susceptible to random noise due to the brightness patches

> For training, I used a specific technique, 

#Track1
1) Counter Clockwise - One Lap of Smooth driving, one lap of corener recovery

2) Clockwise - One Lap of Smooth Driving

#Track2

1) One lap of counter clockwise smooth driving

This way I was able to generate all the necessary data to make the car drive properly on the track
