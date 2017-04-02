# Traffic Sign Recognition

## Will Rifenburgh

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./download.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./websearch_signs/german_signs_5/dangerous_curve_to_the_right20.jpg "Traffic Sign 1"
[image5]: ./websearch_signs/german_signs_5/roundabout_40.jpg "Traffic Sign 2"
[image6]: ./websearch_signs/german_signs_5/speed_limit_60kmph_3.jpg "Traffic Sign 3"
[image7]: ./websearch_signs/german_signs_5/stop_14.jpg "Traffic Sign 4"
[image8]: ./websearch_signs/german_signs_5/yield_13.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/wmrifenb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Here is a link to my [project html](https://github.com/wmrifenb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library and the python csv library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

I basically picked a sign image at random and displayed it with its csv file enumeration. Running the cell repeatedly allowed me to explore the images and their respective classifcations. The following is an example of what you  would see:

13
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cells of the IPython notebook.

I chose to keep all three channels because converting to grayscale would have lead to loss of information that is otherwise useful in training the the neural network to classify the signs.

As first step, I attempted to normalize image data by converting the range of values from 0-255 to 0.0-1.0. This proved to hurt the convergence of the neural network so I abondoned it.

I chose to leave the image data unchanged.


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for training, validation and testing is in the seventh cell of the notebook. The code is largely taken form the LeNet lab notebook with minor changes. Changes include switching to 3 chanels, changing one hot encoding to account for the now 43 output possibilities and increasing the learning rate to 0.001

To cross validate my model, I simply used the data found in valid.p against the data found in train.p 


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  | 400 |
| Fully connected		| 120        									|
| RELU | |
| Fully connected | 84 |
| RELU | |
| Fully connnected | 43 |
| Softmax				|        									|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used an the AdamOptimizer as in the LeNet lab. I kept the same 128 sample batch size but increased the epochs to 30. 10 epochs was sufficient enought to achieve 94% accuracy on the validation data. Inceasing to 30 was done to ensure the neural network would refine itself enough to satisfy the 93% accuracy criteria on the test data. 30 epochs brought the accuracy to 95.3% on the validation data.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eigth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 95.3% 
* test set accuracy of 93.1%

I chose the LeNet archetecture because it handled optical character recognition well in the LeNet lab. Traffic sign identification is not much more than OCR with three color channels as opposed to just grayscale.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is slightly rotated.

All of the images might be difficult because they have varying brightness and zoom level.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the ninth and tenth cells of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Dangerous curve to the right     		|    Speed limit (30km/h)								| 
| Roundabout mandatory      			|  				Roundabout mandatory  				|
|	Speed limit (60km/h)			| 						Speed limit (50km/h)  			|
| Stop      		| Stop	 				|
| Yield			| Yield	      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image (Dangerous curve to the right), the top five soft max probabilities were:

| Probability|Prediction|
|:--------------------:|:---------------------------:|
| 0.999998 | Speed limit (30km/h) |
| 1.98308e-06 | Speed limit (100km/h) |
| 3.32381e-08 | Right-of-way at the next intersection |
| 5.93201e-10 | Speed limit (120km/h) |
| 9.44289e-11 | Speed limit (80km/h) |

For the second image (Roundabout mandatory) 

| Probability|Prediction|
|:--------------------:|:---------------------------:|
| 0.999955 | Roundabout mandatory |
| 3.2755e-05 | No vehicles |
| 1.15368e-05 | Speed limit (100km/h) |
| 5.35198e-07 | Keep right |
| 1.45116e-07 | Turn right ahead |

For the third image (Speed limit (60km/h))

| Probability|Prediction|
|:--------------------:|:---------------------------:|
| 1.0 | Speed limit (50km/h) |
| 3.47253e-11 | Speed limit (30km/h) |
| 7.77497e-14 | Wild animals crossing |
| 3.83347e-18 | Yield |
| 4.25921e-24 | Speed limit (80km/h) |

For the fourth image (Stop)

| Probability|Prediction|
|:--------------------:|:---------------------------:|
| 1.0 | Stop |
| 2.7326e-10 | Speed limit (30km/h) |
| 7.01483e-11 | Speed limit (80km/h) |
| 8.99677e-13 | No entry |
| 1.60252e-13 | Speed limit (60km/h) |

For the fifth image (Yield)

| Probability|Prediction|
|:--------------------:|:---------------------------:|
| 1.0 | Yield |
| 0.0 | Speed limit (20km/h) |
| 0.0 | Speed limit (30km/h) |
| 0.0 | Speed limit (50km/h) |
| 0.0 | Speed limit (60km/h) |

The accuracy of the model on the new websearch images was 60% compared to 93% of the test data.
Considering that the 'Dangerous curve to right' sign image was taken at an odd angle its not suprising that the model failed to correctly classify it. If I had done some more preprocessing to challenge the neural network more, it may have succeeded in classifying this image.

The estimate for the third image was close. The model guessed the 50kmph sign when it was actually the 60kmph sign. Strangely though it 60kmph didnt make the top five guesses.

The LeNet model was very confident in its guesses - including the wrong ones.
