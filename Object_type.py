import random
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
import skimage.io as io
import skimage.transform as tr
from skimage.feature import hog


class ObjectType:
	"""
		A class that holds image file paths, bounding box information, HOG features, an object detector,
		for a specific 'ObjectType' such as dog or cat etc..

		Methods:
		1. compute_features    	 : Computes HOG features for all images belonging to the specific object, 
								   given the HOG template size
		2. train_classifier 	 : Trains a classfier based on HOG features. The features from this category 
								   itself is used as +ve sample, and the method takes in -ve sample 
								   features as it's argument
		3. generate_random_boxes : Generates a set of random bounding boxes for a given image. Useful for 
								   generating multiple negative training instances	from a single image
	"""


	def __init__(self, name):
		self._group_name = name
		self.file_paths = []
		self.bbox = []

	def compute_features(self,size):
		"""
			Computes HOG features for images in self.file_paths
			Argument 'size' - denotes the size of HOG template
		"""

		print "Computing features for "+self._group_name

		#HOG Feature dimension varies based on size
		if size[0] == 64:
			N_feat = 128
		elif size[0] == 128:
			N_feat = 512
		else:
			N_feat = 2048

		#Place holder for features
		features = np.zeros((len(self.file_paths),N_feat))

		for i in range(len(self.file_paths)):
			#Read the image
			im  = io.imread(self.file_paths[i], as_grey=True)

			#extract ROI - the car
			x1,y1,x2,y2 = self.bbox[i]
			roi = im[y1:y2,x1:x2]

			#Resize image according to given size
			im_new = tr.resize(roi, size) 

			#Compute HOG
			features[i,:] = hog(im_new, orientations=8, pixels_per_cell=(16, 16),
                    			cells_per_block=(1, 1))

		#Save the features, saves future computation
		t_str = 'FeaturesAndClassifiers/features_'+str(size[0])+self._group_name+'.npy' 
		np.save(t_str,features)

		return features

	def train_classifier(self,features_2,size):

		"""
			Trains an SVM classfier based on HOG features of the same class as 
			positive instances and negative instances given as input.

			Arguments:
			1. features_2 - Precomputed HOG features for negative instance [size: N x D]
			2. size       - size of the HOG template (used for feature computation) 
		"""

		#Compute features +ve instances
		features_1 = self.compute_features(size)

		#Form labels
		target = []
		for i in range(len(features_1)+len(features_2)):
			if i <len(features_1):
				target.append(1)
			else:
				target.append(0)

		#Train data - concatenate +ve and -ve instances
		data = np.vstack((features_1,features_2))

		#Spliting the data into train and test
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.4, random_state=0)

		#Grid of parameters - to search for the most suitable parameters
		parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
		               'C': [1, 100, 1000]},
		              {'kernel': ['linear'], 'C': [1,100,1000]}]

		#Grid search + Cross validation
		clf  = GridSearchCV(svm.SVC(probability=True), parameters, scoring='accuracy', cv=10) 
		clf.fit(X_train, y_train)

		#Best accuracy
		print "Accuracy: ", accuracy_score(y_test, clf.predict(X_test))

		#Returns trained classifier
		return clf

	def generate_random_boxes(self, number):

		"""
			Generates random bounding boxes for all images in self.file_paths, useful for
			generating negative training instances

			Argument number - denotes the number of bounding boxes to be generated
		"""

		#Iterating through all images 
		new_paths = []
		for path in self.file_paths:

			#Read the image to get image shape
			im = io.imread(path)
			x,y = im.shape[1],im.shape[0]

			#Set the bounding box template size
			size = 256

			#If the image is smaller than the window size, we can't generate 100s of
			#bounding boxes, we eliminate them
			if (im.shape[1] < 300) or (im.shape[0] < 300):
				continue

			#Generate random positions for bounding boxes
			x_s  = [random.randint(0,(x-size-2))  for i in range(number)]
			y_s  = [random.randint(0,(y-size-2))  for j in range(number)]

			
			#Save the bounding boxes for each image
			for i in range(len(x_s)):
				#Same image for all 'n' bounding boxes
				new_paths.append(path)

				self.bbox.append(list([x_s[i],y_s[i],x_s[i]+size,y_s[i]+size]))

		self.file_paths = new_paths



