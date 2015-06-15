"""
Short version of object detection framework - Refer to car_detection.py for full documentation

Input:
1. Path to test image

Output:
1. If the input is a single image, the detected object is displayed 

"""

import sys
import numpy as np
import clean_detections
import skimage
import skimage.io as io
import skimage.transform as tr
from sklearn.externals import joblib

if len(sys.argv) <2:
	print """Error: At least 1 argument needed: 
			 1. Path to test image
	"""
	sys.exit()
else:
	test_image 	   = sys.argv[1] 

#Test time ----- replace code for runing on cluster
#Read input image and resize it
print "Reading and resizing input image"
im  = io.imread(test_image, as_grey=True)
im = tr.resize(im, (256,256))

#HOG template sizes 
template_sizes = [(64,64),(128,128),(256,256)]

#Slinging window detectors
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
print "Predicted bounding box: ", predictions_final

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