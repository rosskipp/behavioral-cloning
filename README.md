# **Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "Model Architecture"
[image2]: ./images/center_drive.png "Center driving"
[image3]: ./images/placeholder_small.png "Recovery Image"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"

## Rubric Points

---

### Files and Code

#### Files

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summary the results

#### Usage

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

and if you'd like to record a video, you can run:

```sh
python drive.py model.h5 name_of_folder
```

Then run the video.py script on that folder to assemble the video

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I had enough memory to not need to chunk the data for training, so you won't see any generators for training.


### Model Architecture and Training Strategy

#### Model Architecture

My model is based on the NVIDIA network described in the lectures. After seeing really good performance with this network on a limited set of data, it seemed like a great choice for the application. It's a deep, convolutional neural network, and here is the architecture of the network.

* Image normalization
* 2D Convolution: 5x5 kernel, strides: 2x2, filter: 24, activation: RELU
* 2D Convolution: 5x5 kernel, strides: 2x2, filter: 36, activation: RELU
* 2D Convolution: 5x5 kernel, strides: 2x2, filter: 48, activation: RELU
* 2D Convolution: 3x3 kernel, filter: 64, activation: RELU
* 2D Convolution: 3x3 kernel, filter: 64, activation: RELU
* Flatten
* Fully connected: 100
* Fully connected: 50
* Fully connected: 10
* Fully connected: 1 (output)


The first layer of the network performs image normalization so all image values are in the range of -0.5 to 0.5.

The next part of the network are the convolutional layers. These serve the purpose of feature extraction in the network. The first three layers use a 2x2 strided convolutions with a 5×5 kernel. The last two layers are non-strided convolutions with a 3×3 kernel size. The model introduces nonlinearity using RELU activations for each of the convolutional layers.

The final layers of the network are fully connected, first with 100 neurons, then 50, 10 and finally the output. The reason for this design is discussed in the NVIDIA paper as follows:

> The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.

[LINK TO PAPER](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The final model architecture is implimented in the `nvidiaNetwork` function from `model.py` lines 71-86.

Here is a visualization of the architecture from the NVIDIA paper mentioned earlier:

![alt text][image1]


#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).


#### Solution Design Approach

The overall strategy for deriving a model architecture was to create a network that took in images from the vehicle cameras and outputted a steer angle for that given image. This would allow new images from the vehicle to be given a steer angle.

My first step was to use the LeNet network as I thought this model might be appropriate because it was introduced for classifying images, and this seemed to be a classification problem - given an image, output a steer angle. I didn't have the best of luck with the LeNet architecture, so I started looking at the

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to inclued dropout layers between the fully connected layers before the output of the network.  I only ran the network for 3 Epochs. Looking at the loss outputs of the training routine, I probably could have reduced this to two epochs as well.

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The behavior of the final trained network, in that it can successfully drive the track, and also doesn't seem to act erratically to any given input is a good sign that there isn't too much overfitting.


#### Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I drove the 1st track only in my training data. I put in two smooth laps of driving in the center of the lane in the forwards direction, a lap of recovery driving from the edges, and also some more sweeps of the corners.

Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


The data was shuffled before being used for training, and also split into a 20% validation set and 80% training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
