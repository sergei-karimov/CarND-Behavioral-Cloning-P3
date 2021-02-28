# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./NV_NET.png "Model Visualization"
[image2]: ./CENTER_IMAGE.jpg "Grayscaling"
[image4]: ./RECOVERY_1.jpg "Recovery Image"
[image5]: ./RECOVERY_2.jpg "Recovery Image"
[image6]: ./NORMAL.jpg "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 64 (model.py lines 75-92) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, several data augmentation techniques like flipping images horizontally as well as using left and right images to help the model generalize were used. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Another technique to reduce overfitting was to introduce dropout in the network, with a dropout probabilities of 0.2 and 0.3.
#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).
##### Hyperparameters
```python
EPOCH = 50
TEST_SIZE = 0.2
BATCH_SIZE = 32
```
#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set. A validation loss was calculated as the mean absolute error between the true and predicted steering angles for the validation set and this was used to monitor progress of training.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a good model was to use the Nvidia architecture since it has been proven to be very successful in self-driving car tasks. The architecture was also recommended in the lessons and it's adapted for this use case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

I first tried training the model with the network described in the lessons but could not get the model to drive correctly. The next step was to add more convolutions and layer activations, which still did not get the model to do what was expected. After many attempts, the nvidia architecture was adapted for this project and this worked. The architecture worked in creating a model that could drive around the track.
#### 2. Final Model Architecture

The final model architecture (model.py lines 75-92) consisted of a convolution neural network.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles.
After the collection process, I had 33442 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
