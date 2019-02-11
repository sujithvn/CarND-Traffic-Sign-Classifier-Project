# **Traffic Sign Recognition using Deep Learning** 

## Writeup

### Submitted as part of Udacity Self Driving Car Engineer Nanodegree.
###### Sujith V Neelakandan
---

**Build a Traffic Sign Recognition Project**

**The steps of this project are the following: **
* Step 0: Load The Data
* Step 1: Dataset Summary & Exploration
  * 1.1 Basic Summary of the Data Set
  * 1.2 A quick visualization of the dataset
  * 1.3 Distribution of each CLASS in the Train/Test/Validate datasets
* Step 2: Pre-process the Data Set
  * 2.1 Shuffle the dataset
  * 2.2 Normalization of the images along with histogram equalization
  * 2.3 Generating variants of the image using Augmentation
  * 2.4 Generating new image data via Normalization and Augmentation
* Step 3: Design and Test the Model Architecture
  * 3.1 Model Architecture
  * 3.2 Training Pipeline
  * 3.3 Model Evaluation
  * 3.4 Train, Validate and Test the Model
* Step 4: Test a Model on New Images
  * 4.1 Load and Output the Images
  * 4.2 Predict the Sign Type for Each Image
  * 4.3 Analyze Performance
  * 4.4 Output Top 5 Softmax Probabilities For Each Image
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./writeup_images/single.png "Single"
[image2]: ./writeup_images/grid_43.png "Grid"
[image3]: ./writeup_images/histo_train.png "Histo_Train"
[image4]: ./writeup_images/histo_valid.png "Histo_Valid"
[image5]: ./writeup_images/histo_test.png "Histo_Test"
[image6]: ./writeup_images/normalized.png "Normalized"
[image7]: ./writeup_images/augmented.png "Augmented"
[image8]: ./writeup_images/new_5.png "Traffic Sign 5"
[image9]: ./writeup_images/new_speed30_probs.png "new_speed30_probs"
[image10]: ./writeup_images/new_ahead_probs.png "new_ahead_probs"
[image11]: ./writeup_images/new_aheadleft_probs.png "new_aheadleft_probs"
[image12]: ./writeup_images/new_NOVeh_probs.png "new_NOVeh_probs"
[image13]: ./writeup_images/new_gencaution_probs.png "new_gencaution_probs"


##### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)



#### Data Set Summary & Exploration

##### 1. Here is a basic summary of the data set.

Basic len(), unique() and subsetting functions were used to get a basic summary of the datasets. Here are some quick facts about the same: *Refer code section 1.1*

* The size of training set is: **34799**
* The size of the validation set is: **4410**
* The size of test set is: **12630**
* The shape of a traffic sign image is: **(32, 32, 3)**
* The number of unique classes/labels in the data set is: **43**

##### 2. Include an exploratory visualization of the dataset.

Let us quickly see a random sample image from the training dataset: *Refer code section 1.2*
![SingleImage][image1]

Now let us glance through an image each from the 43 classes:
![Grid43][image2]

Here is an exploratory visualization of the data set. The are the histograms showing the distribution of the various classes in the three datasets: *Refer code section 1.3*

![Histogram-Training][image3]  ![Histogram-Validation][image3]  ![Histogram-Test][image5]



#### Pre-processing, Design and Test a Model Architecture

##### 1. Pre-processing of Data

As a first step, I decided to shuffle the training dataset so that thre is no specific order of classes that may impact the learning.  *Refer the code section 2.1*

###### Normalization

It is a good practice to normalize the input data so as to bring down the high range of its parameters. I used histogram equilzation followed by Normalization.  *Refer the code section 2.2*

Here is an example of a traffic sign image before and after Normalization.

![Normalized][image6]


###### Augmentation

Since the data distribution among the different classes were not uniform, we would face a situation of imbalanced classes. This would result in our model predicting a more frequent class when there probabilities are not much different. This cann be fixed by providing more data input for the sparse classes.

I decided to make a balanced dataset of 2000 entries per class. To add more data to the the data set, I used the techniques such as adjusting brightness,  rotation, translation & affine-transform. *Refer the code sections 2.3 and 2.4.*

Here is an example of an original image and an augmented image:

![Augmented][image7]

##### 2. Design and Test Model Architecture

The **LeNet-5** implementation shown in the classroom at the end of the CNN lesson is what is used as a starting point. The LeNet-5 solution is quite robust and we can expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93.

My final model consisted of the following layers:



|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x3 RGB image             |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride, valid padding, outputs 5x5x16 |
|     Flatten     |                Output 400                |
| Fully connected |                Output 120                |
|      RELU       |                                          |
| Fully connected |                Output  84                |
|      RELU       |                                          |
| Fully connected |                Output  43                |
|     Softmax     |                                          |



##### 3.2. Training your model.

To train the model, calculated 'logits' are evaluated for the least cross-entropy. *AdamOptimizer* is generally recommended to have better results then SGD. *Refer code section 3.2*

#### 4. Training & Evaluation of the model.

##### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of : 0.832  (This was because we were evaluating against the original training set)
* validation set accuracy of: 0.948
* test set accuracy of: 0.935

Fine tuning of the architecture:
* I stuck with the LeNet architecture, because it had a high level of accuracy as mentioned in the original research paper. 
* I tried DROPOUT of 0.5 in the Fully connected layers, but could not get the expected results.
* The Learning rate was adjusted twice in the training process. I switched to 0.0001 at 50% of the epochs and switched to 0.00001 at 75% of the epochs.


#### Test a Model on New Images

##### 1. Five German traffic signs found on the web. 

Here are five German traffic signs that I found on the web:

![new5][image8]

##### 2. The model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
| Speed limit (30km/h) | Speed limit (30km/h) |
|      Ahead only      |      Ahead only      |
| Go straight or left  |  Beware of ice/snow  |
|     No vehicles      |     No vehicles      |
|   General caution    |   General caution    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of **80%**. This compares favorably to the accuracy on the test set of 0.935

##### 3. Describing how certain the model is when predicting on each of the five new images.

We verify this by looking at the softmax probabilities for each prediction.  Refer code section 4.4 in Python Notebook.

###### First image

The model is completely sure that this is a _Speed limit (30km/h)_ sign (probability of 1.0), and the image is that of a _Speed limit (30km/h)_ sign. The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|     1.0     | Speed limit (30km/h) |
|     .0      | Speed limit (20km/h) |
|     .0      | Speed limit (50km/h) |
|     .0      | Speed limit (70km/h) |
|     .0      | Speed limit (80km/h) |
![speed30][image9]

---

###### Second image
| Probability |      Prediction      |
| :---------: | :------------------: |
|     1.0     |      Ahead only      |
|     .0      |   Turn left ahead    |
|     .0      |   Turn right ahead   |
|     .0      | Go straight or right |
|     .0      | Go straight or left  |
![ahead][image10]

---

###### Third image 
Model got this one **WRONG**. It gave almost 0.00% probability to the correct sign of _Go straight or left_.  

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    .9014    |      Beware of ice/snow      |
|    .0982    |       Turn left ahead        |
|    .0003    | Dangerous curve to the right |
|    .0001    |          Ahead only          |
|    .0000    |     Go straight or left      |
![aheadleft][image11] 

---

###### Fourth image 
| Probability |             Prediction              |
| :---------: | :---------------------------------: |
|     1.0     |             No vehicles             |
|     .0      |          End of no passing          |
|     .0      |        Speed limit (80km/h)         |
|     .0      |                Yield                |
|     .0      | End of all speed and passing limits |
![novehicle][image12]

---


###### Fifth image
| Probability |        Prediction         |
| :---------: | :-----------------------: |
|     1.0     |      General caution      |
|     .0      |        Pedestrians        |
|     .0      |        Bumpy road         |
|     .0      |      Traffic signals      |
|     .0      | Road narrows on the right |
![gencaution][image13]

---

#### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
###### This optional step was not performed at this time due to time constraint (deadline) in submitting this project. This step will be re-visited later


