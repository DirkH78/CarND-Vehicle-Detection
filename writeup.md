## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image_0.png
[image2]: ./output_images/image_1.png
[image3]: ./output_images/image_2.png
[image4]: ./output_images/image_3.png
[image5]: ./output_images/image_4.png
[image6]: ./output_images/image_5.png
[image7]: ./examples/image0000.png
[image8]: ./examples/image73.png
[image9]: ./examples/HOG.png
[image10]: ./examples/HeatMaps.PNG
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `VehicleDetection.py` (in lines 102 through 132).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image7]
![alt text][image8]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(32, 32)` and `cells_per_block=(2, 2)`:

![alt text][image7]
![alt text][image9]

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters above were finally used since they showed the best results. An additional parameters is: `hog_channel = 2`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (in lines 303 through 309) using a grid search (cross validation) to identify the best suited parameters for C. For training and testing the feature data was split and scaled by the function `createScaledTestTrainData`. The accuracy was 98%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Choosing `y_start_stop = [350, None]`, `xy_window = (64, 64)` and `xy_overlap = (0.7, 0.7)` provided the best results:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

The feature vectors didn't just include the HOG-features. A color histogram feature (`color_hist`) and a spatial feature vector (`bin_spatial`) were appended as well to increase the quality of the identification process.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. A gaussian blur effect was used to increase the significance of positive identifications.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of test images:

![alt text][image10]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To increase the quality of the video stream based detection, it could be helpful to implement a heatmap which will "burn out" over time. Thus the information of positive identifications could be saved for the following frame. The data base of vehicles (and non vehicles) needs to be widened so that bicicles, trucks and pedastrians, animals,... could be identified as well.

