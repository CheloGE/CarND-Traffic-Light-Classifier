# CarND-Traffic-Light-Classifier
This project aims to classify traffic lights coming from a simulator 

Author: Marcelo Garcia

[//]: # (Image References)

[figure1]: ./figures/Labels.JPG "Labels"
[figure2]: ./figures/preprocess.JPG "preprocess"
[figure3]: ./figures/preprocess2.JPG "preprocess"
[figure4]: ./figures/distribution.JPG "distribution dataset"
[figure5]: ./figures/CNN.JPG "CNN"
[figure6]: ./figures/greenTest.JPG "greenTest"


### Dataset

Data was taken using CARLA simulator as shown in the GIF image below:

<div class="wrap">
    <img src=".\figures\Simulator.gif" />
    <br clear="all" />
</div>

As illustrated above, the end goal of the traffic light classifier is to integrate it into a ROS pipeline that drives a car along a highway. 

The dataset was taken frame by frame and labeled manually. 

The different classes that were labeled are the following:

![][figure1]

The dataset distribution was not balanced though, as shown below:

![][figure4]

As the data was not balanced we performed a stratified cross validation to validate the model afterwards.

### Preprocess data

* Images were standardized and resized
* Images were converted to gray to threshold using Otsu's Binarization. The idea is to get the edges in the image.
* Use the thresholded image as a mask
* Convert image to HSV color space to get the value channel and threshold values in the range of green, red, and yellow. 

After the threshold we had something like below:

![][figure2]

Not all images were that clean though. Some other examples are shown below:

![][figure3]

### Model build

Even though the images were not completely clear they were still handled pretty good by a CNN. The CNN structure has the following structure:

![][figure5]

The model parameters were:

* Loss= categorical_crossentropy
* Optimizer: ADAM with a learning rate decay starting on 1e-3
* Regularization techniques:
    * Early stop
    * dropout layers

Finally, the model classified correctly the images with an accuracy of 0.91 on validation dataset. Nevertheless, an F1-score would've been more suitable as this dataset was imbalanced. A test in random images of class "green" visually confirmed the performance of the network, as illustrated below:

![][figure6]


