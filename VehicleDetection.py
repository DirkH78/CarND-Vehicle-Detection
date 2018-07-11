import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

###Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
scale = 1.5

#Helper Functions
def ImportPicturesFromFolder(folder, color_space='RGB', show=False):
    # create image array
    i=0
    imglist=[]
    for fname in glob.glob(folder, recursive=True):
        imglist.append(np.array(mpimg.imread(fname)))
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2YCrCb)
        else: imglist[i] = np.copy(imglist[i]) 
        # normalize
        #if np.amax(imglist[i]) <= 1:
        #    imglist[i] = np.round(imglist[i]*255)
        if show == True:
                plt.imshow(imglist[i])
                plt.show()
                print("Imported RGB-image No.:", i+1, "/Shape:", imglist[i].shape, "/Max:", np.amax(imglist[i]))
        i=i+1
    return imglist

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256), show=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if show == True:
            # Generating bin centers
            bin_edges = channel1_hist[1]
            bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
            # Plot a figure with all three bar charts
            fig = plt.figure(figsize=(12,3))
            plt.subplot(131)
            plt.bar(bin_centers, channel1_hist[0])
            plt.xlim(0, bins_range[1])
            plt.title('R Histogram')
            plt.subplot(132)
            plt.bar(bin_centers, channel2_hist[0])
            plt.xlim(0, bins_range[1])
            plt.title('G Histogram')
            plt.subplot(133)
            plt.bar(bin_centers, channel3_hist[0])
            plt.xlim(0, bins_range[1])
            plt.title('B Histogram')
            plt.show()
            plt.close(fig)
    return hist_features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32), show=False):
    # Use cv2.resize().ravel() to create the feature vector
    featureimg = cv2.resize(img, size)
    features = featureimg.ravel()
    # Return the feature vector
    if show == True:
        plt.imshow(featureimg)
        plt.show()
    return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        plt.imshow(hog_image)
        plt.show()
        return features
    # Otherwise call with one output
    else:     
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(imglst, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for i in range(len(imglst)):
        # Read in each one by one
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(imglst[i], spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(imglst[i], hist_bins, hist_range)
        # get HOG features
        hog_features = get_hog_features(imglst[i], orient, pix_per_cell, cell_per_block, vis, feature_vec)
        hog_features = np.ravel(hog_features)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

def createScaledTestTrainData(car_features, notcar_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
 
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
     
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
         
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

testimglist=ImportPicturesFromFolder("./test_images/*jpg", color_space=color_space, show=False)
#cars=ImportPicturesFromFolder("./training_images/vehicles/**/*png", color_space=color_space, show=False)
#notcars=ImportPicturesFromFolder("./training_images/non-vehicles/**/*png", color_space=color_space, show=False)

## Testing and Debugging
cars=ImportPicturesFromFolder("./lim_training_images/vehicles/**/*png", color_space=color_space, show=False)
notcars=ImportPicturesFromFolder("./lim_training_images/non-vehicles/**/*png", color_space=color_space, show=False)
#testimg=cars[0]
#hist_features = color_hist(testimg, 32, (0, round(np.amax(testimg))), show=True)
#bin_spatial(testimg, size=(32, 32), show=True)
#features = get_hog_features(testimg, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)

## Find Cars
carfeatures = extract_features(cars, spatial_size, hist_bins, (0, 1), orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
notcarfeatures = extract_features(notcars, spatial_size, hist_bins, (0, 1), orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
X_train, X_test, y_train, y_test=createScaledTestTrainData(carfeatures, notcarfeatures)
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
print('My SVC predicts: ', svc.predict(X_test[0:round(len(X_test)*0.1)]))
print('For labels: ', y_test[0:round(len(y_test)*0.1)])