# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because having a three channel image makes classifying harder.


As a last step, I normalized the image data because it makes the numbers and results make more sense, human-wise, and easier to interpret 

I decided to generate additional data because there were classes that are very 'underfit'; Very low on count compared to others, 
thus making the data unbalanced which impacts the predictions and learning process, what the model learns and what it ignores.  

To add more data to the the data set, I used the following techniques; 
Rotation and Translation 

 Rotation was done three times with three different rotation-angles. 90, 180, 270
Translation was done only once by moving pixels by 5 on X and Y axis each. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Average pooling	    | 2x2 stride, VALID padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride,  VALID padding, outputs 10x10x16. |
| RELU                  |                                               |
| Average Pooling       | 2x2 stride, VALID padding, outputs 5x5x16     |
| Convolution 5x5       | 1x1 stride, VALID padding, outputs 1x1x400    |
| RELU                  |                                               |
| Flattened             | input 1x1x400, ouput 400                      |
| Fully Connected		| input 400, output 200       					|
| Fully Connected       | input 200, output 120                         |
| Fully Connected 		| input 120, output 84 							|
| Fully Connected 		| input 84, output 43							|
| Softmax				| \      										|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
batch size was changed to 64 and epochs to 40; after using many sizes and numbers, this was best combination to get the best prediction I could achieve. 
Adam optimizer was used, it's easy to implement and tunes itself thanks to adaptive learning rates.  


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 99.9%
* validation set accuracy of ? 94.4% 
* test set accuracy of ? 92.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet-5
* What were some problems with the initial architecture? it could not achieve the desired minimum accuracy
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

whilst 99 and 94.4 are not much of difference inbetween, but I think classes/labels needed better balance
I have added another Convolution layer, dropout layers, and a fulyl connected layer, which could improve the accuracy from (89% to 94.5%)

* Which parameters were tuned? How were they adjusted and why? dropout set to 0.5 on training and 1 on validation. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? 

If a well known architecture was chosen:
* What architecture was chosen? LeNet-5 but I had to adjust it
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Left      		| 0 		Dangerous curve to the left   		| 
| Turn Right     		| 0 		Slippery road						|
| Double Curve			| 1												|
| Double Curve	      	| 0			End of all speed and passing limits	|
| Road Work				| 1     										|
| Pedestrians			| 1												|
| Wild Animal Crossing  | 1												|
| Road narrows on right | 1												|
| Right-of-way next-int.| 1												|
| Stop 					| 0												|
| Priority Road			| 1												|
| Yield 				| 1												|


The model was able to correctly guess 8 of the 12 traffic signs, which gives an accuracy of 66%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Dangerous curve to the left  					| 
| 7.097439e-11   		| Slippery road									|
| 7.755852e-13			| Double curve									|
| 3.2228455e-16	      	| End of all speed and passing limits			|
| 2.5027635e-17			| Wild animals crossing     					|





For the 2nd image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999981        		| Dangerous Curve to the right  				| 
| 1.9025533e-06    		| Children Crossing								|
| 2.7679055e-08			| Slippery road									|
| 1.3563364e-10	      	| End of no passing					 			|
| 8.625412e-11			| Traffic signals      							|


For the 3nd image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Double curve  								| 
| 5.512785e-14     		| Slippery road									|
| 1.5251313e-16			| Wild animals crossing							|
| 1.2279856e-16	      	| Dangerous curve to the left					|
| 5.6617363e-22			| Beware of ice/snow      						|

For the 4rd image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children crossing   							| 
| 1.1717271e-28     	| Bicycles crossing 							|
| 2.805951e-29			| Turn left ahead								|
| 2.3754153e-32	      	| Slippery road					 				|
| 2.2535324e-32			| Beware of ice/snow      						|

For the 5th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work   									| 
| 3.1820002e-09     	| Yield 										|
| 1.5448344e-09			| Beware of ice/snow							|
| 5.927037e-10	      	| No passing for vehicles over 3.5 metric tons	|
| 2.98145e-10			| Keep right      								|

For the 6th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999783         	| Pedestrians   								| 
| 1.15530775e-05    	| Road narrows on the right 					|
| 7.685829e-06			| General caution								|
| 2.4603014e-06      	| Traffic signals					 			|
| 1.2935358e-10			| Dangerous curve to the left      				|

For the 7th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0          			| Wild animals crossing   						| 
| 8.469257e-25      	| Double curve 									|
| 4.6397588e-26			| Traffic signals								|
| 3.8626825e-27 	    | Yield					 						|
| 1.0920853e-30			| Dangerous curve to the left     				|

For the 8th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999542        		| Road narrows on the right   					| 
| 4.5733716e-05   		| Bicycles crossing								|
| 5.4125508e-08			| Pedestrians									|
| 1.6433732e-09    		| Double curve					 				|
| 2.9807334e-10			| Wild animals crossing      					|

For the 9th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Right-of-way at the next intersection  		| 
| 2.0024686e-32     	| Beware of ice/snow 							|
| 4.987838e-37			| Double curve									|
| 0.0	      			| Speed limit (20km/h					 		|
| 0.0				    | Speed limit (30km/h      						|

For the 10th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.8488767         	| Traffic signals  								| 
| 0.04398035     		| Pedestrians									|
| 0.021856831			| Speed limit (120km/h)							|
| 0.011224006	      	| Speed limit (70km/h)					 		|
| 0.010943037			| No vehicles     								|

For the 11th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   								| 
| 1.6743965e-31     	| No vehicles 									|
| 1.252755e-31			| End of all speed and passing limits			|
| 1.179285e-32	      	| Wild animals crossing					 		|
| 1.0063582e-34			| Speed limit (50km/h)     						|

For the 12st image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Yield  										| 
| 1.274425e-14     		| Bumpy road 									|
| 1.1283608e-22			| Bicycles crossing								|
| 3.614842e-25      	| Ahead only					 				|
| 6.9406874e-30			| Wild animals crossing     					|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


