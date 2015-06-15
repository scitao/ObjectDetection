"""
Object detection framework:
Code to train an object detector based on HOG features and SVM, preditions are 
made through sliding wondow detectors

Inputs:
1. Txt file containing +ve training image paths. These images must contain at least 
	one instance of the 'object' of interest
2. Txt file containing bounding box annotations for the images in argument 1, the 
	annotations must be of the form x1,y1,x2,y2
3. Txt file containing -ve training image paths. These images must NOT contain even 
	one instance of the 'object' of interest
4. Path to test image

Output:
1. If the input is a single image, the detected object is displayed 

TO DO: Hard negative mining
"""

import sys
import numpy as np
from Object_type import *
import clean_detections
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import skimage.transform as tr
from sklearn.externals import joblib

if len(sys.argv) <5:
	print """Error: At least 4 arguments needed: 
			 1. Txt file containing +ve training image paths.
			 2. Txt file containing bounding box annotations for the images in argument 1
			 3. Txt file containing -ve training image paths.
			 4. Path to test image
	"""
	sys.exit()
else:
	positive_paths = sys.argv[1]
	bbox_paths     = sys.argv[2]
	negative_paths = sys.argv[3]
	test_image 	   = sys.argv[4] 


#Read the image paths
image_paths = open(positive_paths).read().split('\n')
image_paths = image_paths

#Read the bounding boxes
boxes = open(bbox_paths).read().split('\n')
boxes = boxes

#Storing the bounding boxes in numpy array of the form x1,y1,x2,y2
bb_boxes = np.zeros((len(boxes),4))
for i in range(len(boxes)):	
	bb_boxes[i,:] = [int(x) for x in boxes[i].split(',')]


#Computing the aspect ratios
aspect_ratio = []
area = []
for i in range(len(boxes)):
	ratio = (bb_boxes[i,2]-bb_boxes[i,0])/float(bb_boxes[i,3]-bb_boxes[i,1])
	aspect_ratio.append(ratio)

## Visualize aspect ratio distribution
#hisT = np.histogram(np.array(aspect_ratio),bins = [x/100.0 for x in range(0,450,45)])
#plt.scatter(hisT[1][:-1], hisT[0], c='r')
#plt.show()
threshold = 1.5

#Invoking 'ObjectType' class, that holds: file paths, bounding boxes, 
#computes hog features, trains classifier for each category of cars
front_cars = ObjectType('Front_cars')
side_cars  = ObjectType('Side_cars')

#Splitting the data based on aspect ratio
for i in range(len(boxes)):

	if aspect_ratio[i] < threshold:
		#Front facing cars
		front_cars.file_paths.append(image_paths[i])
		front_cars.bbox.append( list(bb_boxes[i,:]) )
	else:
		#Side facing cars
		side_cars.file_paths.append(image_paths[i])
		side_cars.bbox.append( list(bb_boxes[i,:]) )

#Nagative samples
non_cars = ObjectType('Non_cars')

#Image paths for negative samples
non_cars.file_paths = open(negative_paths).read().split('\n')

#Generate bounding boxes from non car images for negative samples
num_boxes = 60 #Number of boxes for each non car image
non_cars.generate_random_boxes(num_boxes)

#Compute hog template and detectors in different sizes 
#Avoids image pyramids when testing
template_sizes = [(64,64),(128,128),(256,256)]

#Train detector - repeat for multiple template sizes
for i in range(len(template_sizes)):

	size = template_sizes[i]
	
	print "Computing features for template size ", size

	#Compute features for negative instances
	features_2 = non_cars.compute_features(size)

	#Train the detector - Linear svm
	#Best SVM params through cross validated grid search
	clf_front = front_cars.train_classifier(features_2,size)
	clf_side = side_cars.train_classifier(features_2,size)

	#Save the detectors during training time
	temp_str = 'FeaturesAndClassifiers/front_'+str(size[0])+'.pkl'
	joblib.dump(clf_front, temp_str, compress=9)

	temp_str = 'FeaturesAndClassifiers/side_'+str(size[0])+'.pkl'
	joblib.dump(clf_side, temp_str, compress=9)



#Test time ----- replace code for runing on cluster
#Read input image and resize it
im  = io.imread(test_image, as_grey=True)
im = tr.resize(im, (256,256))


detector_types = ['front_','side_']
predictions = []
for i in range(len(template_sizes)):

	size = template_sizes[i]

	print "Running sliding window detectors for template size ", size

	for name in detector_types:

		#Load the trained detector
		temp_str = 'FeaturesAndClassifiers/'+name +str(size[0])+'.pkl'
		detector = joblib.load(temp_str)

		#Run the detectors on the image - sliding window
		predictions.extend( list(clean_detections.detection(im,size, detector)))
del im

#Non - maximal supression to get the best bounding box from overlaping ones
K =1; bandwidth = 30
predictions_final = clean_detections.nms(predictions, bandwidth,K)


#Display RGB image with bounding box
im  = io.imread(test_image)
im = tr.resize(im, (256,256))
temp = np.copy(im)

#Draw a bounding box in iamge, hacky workaround to avoid using opencv funtion/dependencies
for i in range(len(predictions_final)):

	x1,y1,x2,y2 = predictions_final[i]

	im[y1:y2,x1:x2,:] = [255,0,0]
	im[(y1+2):(y2-2), (x1+2):(x2-2), :] = temp[(y1+2):(y2-2), (x1+2):(x2-2), :]

io.imshow(im)
io.show()

