# **Traffic Sign Recognition**

## Project Summary Document/Report

### This document is the writeup for the project Traffic Sign Recognition

---

**The Traffic Sign Recognition Project is built based on the project code template**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/Histogram_TrainingData.png "Histogram"
[image2]: ./writeup_images/SignSample.png "Sign Image Sample"
[image3]: ./writeup_images/Preprocessing.png "Before_After_Preprocessing"
[image4]: ./new-traffic-sign/1x.png "Traffic Sign 1"
[image5]: ./new-traffic-sign/2x.png "Traffic Sign 2"
[image6]: ./new-traffic-sign/3x.png "Traffic Sign 3"
[image7]: ./new-traffic-sign/5x.png "Traffic Sign 4"
[image8]: ./new-traffic-sign/6x.png "Traffic Sign 5"
[image9]: ./writeup_images/Top5Guess.png "Top 5 Guess"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The following figure shows the distribution on the number of images for each type of traffic sign. It could be seen that some signs have more training data but some signs have less training data. 

![The plot about the histogram of the traffic sign images][image1]

A random pick of the traffic sign image shows below. It could be seen that the grayscale is a good way to extract feature on the shape of the traffic sign.

![A sample on the traffic sign image][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I convert the image from BGR to YUV, using cv2.cvtColor function. And then, I apply the edge enhancement by doing the subtraction of Gaussion blur, in order to have better contrast on the traffic sign's edge.

Here is an example of a traffic sign image before and after the preprocessing.

![Image showing before and after preprocessing][image3]

I didn't try to add more data or data augumentation, since the results are already good after training the model with the preprocessed data.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Preprocessed image   					| 
| Convolution 5x5     	| shape = (5,5,1,108), outputs 28x28x108 	    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108 				|
| Convolution 5x5	    | output 10x10x200      						|
| RELU          		|         									    |
| Max pooling			| 2x2 stride,  outputs 5x5x200         		    |
| Flatten				| output 5000									|
| Fully connected		| output 120									|
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| Fully connected       | output 43                                     |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an BATCH_SIZE of 128, and EPOCHS of 100. I chose the learning rate to be 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.971
* test set accuracy of 0.955


If a well known architecture was chosen:
* What architecture was chosen? LeCun's Model was chosen.
* Why did you believe it would be relevant to the traffic sign application?
  In the LeCun's paper, the data used to train the model is the traffic sign. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 In the paper, the author did extensive comparison on the following architecture parameteres:
 1. Number of features at each stages: 108 - 200 seems the best
 2. Single or Multi-scale features
 3. Classifier architecture: signal layer vs. 2-layer
 4. Color: YUV or Y only
 5. Different Learning rate

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because because the shape in the center is neither letter nor numbers. The sign does not have distinguished shape. The second one is to test if the classifier could differentiate the speed limit sign with different limit numbers. The third one is to see if the classifier could see the boundry between the white sign and the white background. The fourth one is to test if the direction of the arrow could be differentiated. The fifth one is to test the sign's color differentiated from the background as well as the detection of the sign's arrow direction. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way (11)     | Right-of-way (11)  							| 
| 30 Speed (1)     		| 30 Speed (1)										|
| Priority Road			| Priority Road											|
| Keep right	      	| Keep right					 				|
| Turn left ahead		| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.955

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following is the result showing the top 5 softmax probabilities for each image:

![alt text][image9]




