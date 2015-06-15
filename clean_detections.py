import numpy as np
from numpy import matlib
from skimage.feature import hog

#Sliding window detection
def detection(im,size_temp, detector):
    """
        Performs sliding window detection using trained classifier for the given input image

        Arguments
        1. im         - Input image (grey scale)
        2. size_temp  - Size of the HOG template to use for the given detector
        3. detector   - Pretrained detector used to classify the presence or absence of the 'object'
    """

    predictions = []
    step_size = 5 #Step size of sliding window
     
    #Iterating through all possible positions of the tempalte on the image
    y_max = im.shape[0] - size_temp[0]
    x_max = im.shape[1] - size_temp[1]    
    for y in xrange(0, y_max, step_size):
        for x in xrange(0, x_max, step_size):
            
            #Region of interest for each iteration
            roi = im[y:y+size_temp[0], x:x+size_temp[1]]

            #Compute HOG features 
            feat = hog(roi,orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1))
            
            #Classify whether the box has the object or not
            result = detector.predict(feat)
            prob   = detector.predict_proba(feat) #Measure of confidence in the classifier's decision

            if result ==1:
                predictions.append(list([x,y,x+size_temp[1],y+size_temp[0], prob[0,1]]))
    return predictions


#Non maximal supression
def nms(predictions,bandwidth=30,K=2):
    """
        Non maximal supression to get rid of overlapping detection windows:
        Given a set of candidate bounding boxes, this function employs mean shift clustering
        to get possible unique locations for the object. Further, it chooses the number of classes
        permissible in the final image and outputs the most confident locations

        Arguments:
        1. predictions - 
        1. bandwidth   - permissble bandwidth for each cluster center in mean shift clustering
        2. K           - expected maximum number of 'object' instances in the given image
    """

    #Mean shift clustering
    stop_thresh = bandwidth*0.001
    CCenters,CMemberships = MeanShift(predictions,bandwidth,stop_thresh)

    #Find candidate bounding boxes that are assocciated with maximum confidence 'score'
    all_boxes = np.array(predictions)
    scores = all_boxes[:,-1]
    indices = np.argsort(scores)[::-1];

    #Isolate the final best possible bounding box locations
    count=0
    class_list = []
    final_boxes = []
    for i in range(len(indices)):
        temp = indices[i]

        #Only one bounding box for each 'cluster' of bounding boxes - ensure no overlap in bboxes
        if CMemberships[temp] in class_list:
            continue
        else:
            count +=1
            class_list.append(CMemberships[temp])
            cl = int( CMemberships[temp])
            final_boxes.append(list(CCenters[cl,:]))

        #Do not go more than the maximum expected number of object instances in the image
        if count >= K:
            break

    return final_boxes
    


#Mean shift clustering
def MeanShift(data,bandwidth,stop_thresh):

    """
        Mean Shift clustering - Clusters various bounding boxes to form unique groups amoing them. 
        Can be used to cluster any set of data points in general, only criteria is that each data 
        point must be assocciated with a 'weight'. The number of clusters and the cluster means are 
        computed automatically. 

        Arguments:
        1. data         - Input data to cluster. Shape: N x D+1, the last column represents the 'weight'
        2. bandwidth    - Controls the radius of each cluster center
        3. stop_thresh  - Criteria for the iterative procedure to stop
    """

    #Copying parameters
    data = np.array(data)
    h = bandwidth
    N = data.shape[0]
    F = data.shape[1]-1

    #Inital value of thresh, for the while loop to run
    thresh = stop_thresh +0.002

    #Copying probabilities - we consider it weights in mean shift clustering
    wt = data[:,-1]

    #Actual data
    data = data[:,0:-1];

    #Duplicate of data - Deep copy
    new_data = np.copy(data)

    #Initializing
    CMemberships = np.zeros((N,1))
    M_x = np.zeros((N,F))
    err = np.zeros((N,1))

    #Iterating until cluster centers converge
    while thresh > stop_thresh:

        #For each data point, computing the new center, within its window
        for i in range(N):

            #Distances between each data center current data center 'i'
            new_data_mat = np.matlib.repmat(new_data[i,:],N,1)
            dDist = np.sqrt(np.sum(np.square(new_data_mat-data),axis=1))

            #Data centers that are within the window/bandwidth of the current 
            #data center 'i'
            temp_id = np.where( dDist <= h )
            idx = temp_id[0]

            #Computing the incremental change, to be added to old data center
            M_x[i,:] = np.dot(wt[idx], data[idx,:])/float(np.sum(wt[idx])) 

            #Computing the new error/change, to check for convergence
            err[i] = np.sum(np.square(  M_x[i,:] - new_data[i,:]  ))
            del idx

        #New centers
        new_data = np.copy(M_x)

        #New error
        thresh = np.sum(err);

    #Finding all unique cluster centers
    new_data = np.round(new_data)

    #Finding all unique cluster centers
    temp = np.ascontiguousarray(new_data).view(np.dtype((np.void, new_data.dtype.itemsize * new_data.shape[1])))
    _, idx = np.unique(temp, return_index=True)

    #Unique centers
    CCenters = new_data[idx]

    #Assigning all data points to nearest cluster center
    for i in range(N):

        data_mat = np.matlib.repmat(data[i,:],CCenters.shape[0],1)
        dDist = np.sum(np.square(data_mat - CCenters),axis=1)

        CMemberships[i,0] = np.argmin(dDist);

    return CCenters,CMemberships
