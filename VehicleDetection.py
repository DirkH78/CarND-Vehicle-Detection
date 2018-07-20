import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mpy
import glob
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from sklearn.model_selection import GridSearchCV

###Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 32 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 2 # Can be 0, 1, 2 or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
y_start_stop = [350, None] # Min and max in y to search in slide_window()
#scale = 1.5
hist_range = (0, 1)
xy_window = (64, 64)
xy_overlap = (0.7, 0.7)
threshhold = 5.1

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
                        hog_channel, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img[:,:,hog_channel], orientations=orient,
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
        features = hog(img[:,:,hog_channel], orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(imglst, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, vis=False, feature_vec=False):
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
        hog_features = get_hog_features(imglst[i], orient, pix_per_cell, cell_per_block, hog_channel, vis, feature_vec)
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
    return X_scaler, X_train, X_test, y_train, y_test

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions), 
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched   
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
             
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
    
def search_windows(img, windows, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel):
 
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        test_img = test_img/256
        test_img = test_img.reshape(1,test_img.shape[0], test_img.shape[1], test_img.shape[2])
        #4) Extract features for that window using single_img_features()
        features = extract_features(test_img, spatial_size, hist_bins, (0, 1), orient, pix_per_cell, cell_per_block, hog_channel, vis=False, feature_vec=False)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
 
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
     
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
 
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def process_image(img):
    
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(img, windows, svc, X_scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel) 
    #boxImg = draw_boxes(testimglist[i], hot_windows, color=(0, 0, 255), thick=6)
    #plt.imshow(boxImg)
    #plt.show()
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
         
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshhold)
    # enlarge positives
    heat = gaussian_blur(heat*2, 51)

    # Visualize the heatmap when displaying   
    heatmap = np.clip(heat, 0, 255)
     
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    img = draw_labeled_bboxes(np.copy(img), labels)
    
    plt.imshow(img)
    plt.show()
    return img

# Import data files
testimglist=ImportPicturesFromFolder("./test_images/*jpg", color_space=color_space, show=False)
cars=ImportPicturesFromFolder("./training_images/vehicles/**/*png", color_space=color_space, show=False)
notcars=ImportPicturesFromFolder("./training_images/non-vehicles/**/*png", color_space=color_space, show=False)

## Find Cars
carfeatures = extract_features(cars, spatial_size, hist_bins, (0, 1), orient, pix_per_cell, cell_per_block, hog_channel, vis=False, feature_vec=False)
notcarfeatures = extract_features(notcars, spatial_size, hist_bins, (0, 1), orient, pix_per_cell, cell_per_block, hog_channel, vis=False, feature_vec=False)
X_scaler, X_train, X_test, y_train, y_test=createScaledTestTrainData(carfeatures, notcarfeatures)
# Use a linear SVC (support vector classifier) identify best suited parameters by grid search
parameters = {'C':[1, 3, 5, 7, 9, 11, 13, 15]}
vc = LinearSVC()
svc = GridSearchCV(vc, parameters)
#svc = LinearSVC()

# Train the SVC
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
#print('My SVC predicts: ', svc.predict(X_test[0:round(len(X_test)*0.1)]))
#print('For labels: ', y_test[0:round(len(y_test)*0.1)])


for i in range(len(testimglist)):
    windows = slide_window(testimglist[i], x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(testimglist[i], windows, svc, X_scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel) 
    heat = np.zeros_like(testimglist[i][:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
         
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshhold)
    # enlarge positives
    heat = gaussian_blur(heat*2, 51)
    # Visualize the heatmap when displaying   
    heatmap = np.clip(heat, 0, 255) 
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    img = draw_labeled_bboxes(np.copy(testimglist[i]), labels)
    mpimg.imsave('output_images/image_'+str(i)+'.png',img)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()

##Process video file
#prVid = mpy.VideoFileClip("project_video.mp4")
#heat_n_minus_one = 0
#processedPrVid = prVid.fl_image(process_image)
#processedPrVid.write_videofile("project_video_output.mp4", audio=False)