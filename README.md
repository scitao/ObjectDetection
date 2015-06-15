## Object detection framework in python

### Introduction
Detecting objects is an open computer vision problem. Prior to the days of using Deep learning for object detection, Deformable Part Models (DPM) [1] were the backbone for state of the art detectors. DPMs themselves arose from prior work of Navneet Dalal and Bill Triggs. In their landmark paper for pedestrian detection [2], Dalal and Triggs proposed a sliding window based object detector, that was trained on HOG features of the object, with positive instance features coming from manually annotated images of the object (with bounding boxes) and negative instances coming from randomly sampling bounding boxes in which the object was not present. Their detector was shown to work really well, despite it’s simplicity. This code implements a ‘car’ detector that is very similar to the Dalal and Triggs model, with few additional features. 

### Critical features
* For a given input category (object), multiple detectors are trained based on aspect ratio of the bounding boxes. In this example, 2 types of detectors are trained for car detection - front view and side view. This has been shown to be one of the main reasons why DPMs worked as well as they did - images of a single object can have vastly different appearances, based on the viewpoint/pose and so it is better to train multiple detectors
* The detectors are trained at multiple template sizes. During test phase there is no need for image pyramids. Multi-sized detectors are run on the same input image and predictions from all of them are combined
* Best parameters for the linear SVM classifier built on HOG features are found through grid search and the results are cross validated
* After the sliding window detections, the predictions are cleaned through an automated non maximal suppression process. Mean shift clustering is done to find unique bounding box for each instance of the 'object' in the input image

### Requirements
0. Anaconda python (or) the rest of the following libraries 
0. Python 2.7
0. Scikit learn
0. Numpy
0. Matplotlib
0. Skimage

**Note:** OpenCV can definitely be used instead of skimage, but it has been purposefully avoided. Getting OpenCV configured can be a pain. 

**Download the following**
* Training data from http://imagenet.stanford.edu/internal/car196/car_ims.tgz and extract it within the code’s main folder
* A subset of PASCAL VOC 2007, for negative instances of the object category, download from: http://bit.ly/1BcuwwF

### Running the code
* The actual time for training is reasonably long when using 10K > images for +ve instances and a similar amount of -ve instances. So, the features have been precomputed and the network has been trained. The code test.py loads the trained classifiers and makes predictions on the input image. it can be run by executing:
python test.py test_image.jpg  

* To train the detectors from scratch and make predictions on a new image, execute the following line:
python car_detection.py paths.txt bb_box.txt negative_paths.txt test_image.jpg

**Warning**: training will take a lot of time. This is since we are extracting features for 20k+ images for 3 different HOG template sizes and training a scikit learn classifier on each of the 3 templates (2 detectors * 3 templates). An interesting note is that scikit’s classifier slows down when requesting for prediction probabilities, besides just being able to predict on test images. Also, we train a detector with multiple parameters configurations(3) and cross validate it (10 folds) to find best detector (6 detectors * 10 fold CV * 3 parameters on grid - essentially 180 classifiers are being trained).

The performance of the detectors are not top notch, they could be better with better grid search over SVM parameters (Both gaussian and linear kernels, with more ‘cost’ parameter variations). Another important aspect is to do **hard negative mining**, which hasn’t been implemented in this code. 

### Acknowledgements 
* Carnegie Mellon’s computer vision course, most of the ideas spawned from those lectures
* Thanks to Stanford AI Lab’s car dataset
* PASCAL VOC dataset - used to generate negative training instances

### References
1. Felzenszwalb, Pedro, David McAllester, and Deva Ramanan. "A discriminatively trained, multiscale, deformable part model." Computer Vision and Pattern Recognition, 2008. CVPR 2008. IEEE Conference on. IEEE, 2008.
2. Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005. 