#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set
* The size of test set
* The shape of a traffic sign image is
* The number of unique classes/labels in the data set

All of these statistics are displayed under the second code cell

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. The first plot on the screen uses a random number generator to verify that the image data is being read in from the .p files correctly. The second is a bar chart showing the distribution of the different classes within the datasets. This shows that there are many examples in some classes and relatively few in others. This can be seen as a weakness of the final model because there will be a skew toward the classes with more data. The outputs of these visualizations are below the third and fourth code cells respectively.



###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I normalized the data by using a x = (x-128)/128 for the training, validation and test sets.
I tried different methods of normalizing the data like histogram normalization and pca whitening but using the simple formula above gave comparable results and was simpler to implement.

I chose not to convert the images to grayscale because the images that the system would see in the real world would be in color. Also the loss of information might have caused my model to overfit.  

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code in the fifth code cell is where I did my preprocessing and normalization the goal of these steps was to get mean of the data to have a mean that is close to zero the output of this is displayed below the cell with a mean of -0.354 I decided that I was satisfied with the distribution of the data.  


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch normalization
| RELU					    |
| Max pooling	      | 2x2 stride, valid padding  outputs 14x14x6 	|
| Convolution 5x5	  | 1x1 stride, valid padding, output 10x10x6
| Batch normalization
| RELU				        							
|	Max Pooling				|												|
| Flatten						|												|
| Dropout           | Dropout probability of 0.5
| Fully Connected
| RELU
| Fully Connected
| RELU
| Fully Connected


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook.

To train the model, I used an a batch size of 128 run for 100 epochs

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 87.7%
* test set accuracy of 94.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The architecture was the first tried. There were 2 Batch normalization layers and there was also a Dropout Layer added. These were added and the accuracy of the model increased from ~50% to 94% when run on the final test set. The batch normalization allowed for the normalization of the data within a given batch which allows the model to center the data while it is processing.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The LeNet architecture was chosen. Given that the LeNet architecture is one of the best known architectures for image classification I decided that it would be relevant


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

They are located under the 11th cell
.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:-----------------:|:---------------------------------------------:|
| Keep Right     		| Keep Right   									    |
| Right of Way      | Right of Way at Intersection			|
| Priority Road			| Priority Road 										|
| 60 km/h	      		| 60 km/h				 				            |
| 30 km/h		        | 30 km/h    							          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ~95%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			    | Keep Right   									    |
| .0     				        | Turn Left Ahead 									|
| .0					          | 30 km/h										        |




For the second image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			    | Right of Way  									  |
| .0     				        | Beware Ice								        |
| .0					          | Children Crossing		              |

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the third image, the model is sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			    | Priority Road  									  |
| .0     				        | 100 km/h							            |
| .0					          | No passing vehicles over 3.5 metric tons 		              |
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the fourth image, the model is relatively sure that this is a stop sign (probability of 0.75), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.75         			    | 60 km/h							              |
| 0.15    				      | Wild Animal Crossing							|
| 0.10					        | No passing vehicles over 3.5 metric tons	              |
For the fifth image, the model is sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			    | 30 km/h  									        |
| .0     				        | 50 km/h								            |
| .0					          | 80 km/h		                        |
