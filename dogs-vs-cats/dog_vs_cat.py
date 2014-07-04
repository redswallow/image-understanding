import os
import multiprocessing
import re
import time
import random
from glob import glob
import itertools
import pickle
import numpy as np
import cv2
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model.logistic import LogisticRegression

def resize(dir):
    image_filenames = glob('{}/*.jpg'.format(dir))
    for f in image_filenames:
        f_dir = f.split('/')
        image = cv2.resize(cv2.imread(f),(250,250))
        cv2.imwrite('train_resized/%s' % f_dir[1], image)

def build_file_list(dir):
    """ Given a directory, it builds a shuffled list of the file """
    random.seed(42)
    image_filenames = glob('{}/*.jpg'.format(dir))
    image_filenames.sort()
    random.shuffle(image_filenames)
    return image_filenames

def build_labels(file_list, n_samples = None):
    """ build the labels from the filenames:
    cats corresponds to a 1, dogs corresonds to a -1 """
    if(n_samples == None): 
        n_samples = len(file_list)
    n_samples = max(n_samples,len(file_list))
    file_list = file_list[:n_samples]
    y = np.zeros(n_samples,dtype = np.int32)
    for (i,f) in enumerate(file_list):
        if "dog" in str(f):
            y[i] = -1
        else:
            y[i] = 1
    return y

def file_to_rgb(filename):
    """ return an image in rgb format:
    a gray scale image will be converted,
    a rgb image will be left untouched """
    bild = cv2.imread(filename)
    if (bild.ndim==2):
        #rgb_bild= skimage.color.gray2rgb(bild)
        rgb_bild = cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR) 
    else:
        rgb_bild = bild
    return rgb_bild

def hsv_to_feature(hsv,N,C_h,C_s,C_v):
    res = np.zeros((N, N, C_h, C_s, C_v))
    cell_size= 250/N
    h_range = np.arange(0.0,1.0,1.0/C_h)
    h_range = np.append(h_range,1.0)
    s_range = np.arange(0.0,1.0,1.0/C_s)
    s_range = np.append(s_range,1.0)
    v_range = np.arange(0.0,1.0,1.0/C_v)
    v_range = np.append(v_range,1.0)
    for i in xrange(N):
        for j in xrange(N):
            cell= hsv[i*cell_size:i*cell_size+cell_size,j*cell_size:j*cell_size+cell_size,:]
            for h in range(C_h):
                h_cell = np.logical_and(cell[:,:,0]>=h_range[h],cell[:,:,0]<h_range[h+1])
                for s in range(C_s):
                    s_cell = np.logical_and(cell[:,:,1]>=s_range[s],cell[:,:,1]<s_range[s+1])
                    for v in range(C_v):
                        v_cell = np.logical_and(cell[:,:,2]>=v_range[v],cell[:,:,2]<v_range[v+1])
                        gesamt = np.logical_and(np.logical_and(h_cell,s_cell),v_cell)
                        res[i,j,h,s,v] = gesamt.any()
    return np.asarray(res).reshape(-1)



def build_color_featurevector(pars):
    filename,N,C_h,C_s,C_v =pars
    rgb_bild = file_to_rgb(filename)
    return hsv_to_feature(cv2.cvtColor(rgb_bild, cv2.COLOR_RGB2HSV), N, C_h, C_s, C_v)

def build_color_featurematrix(file_list,N,C_h,C_s,C_v):
    pool = multiprocessing.Pool(2)
    x = [(f,N,C_h,C_s,C_v) for f in file_list]
    print x[:10]
    res = pool.map(build_color_featurevector, x)
    return np.array(res)

def build_color_feature_matrices_or_load(file_list):
    try:
        F1 = np.load("F1.npy")
    except IOError:
        F1 = build_color_featurematrix(file_list,1,10,10,10)
    try:
        F2 = np.load("F2.npy")
    except IOError:
        F2 = build_color_featurematrix(file_list,3,10,8,8)
    try:
        F3 = np.load("F3.npy")
    except IOError:
        F3 = build_color_featurematrix(file_list,5,10,6,6)
    return F1,F2,F3

def classify_color_feature(F,y):
    start = time.time()
    clf = svm.SVC(kernel='rbf',gamma=0.001)
    scores = cross_validation.cross_val_score(clf, F, y, cv=5,n_jobs=-1)
    time_diff = time.time() - start
    print "Accuracy: %.1f  +- %.1f   (calculated in %.1f seconds)" \
    % (np.mean(scores)*100,np.std(scores)*100,time_diff)

#resize("train")
file_list = build_file_list("train_resized")
pickle.dump(file_list, open("file_list.pkl","wb"))
y=build_labels(file_list, n_samples = None)
np.save('y',y)

file_list = pickle.load(open("file_list.pkl","rb"))
print len(file_list)

F1,F2,F3 = build_color_feature_matrices_or_load(file_list[:10000])
print len(F1)
np.save("F1",F1)
np.save("F2",F2)
np.save("F3",F3)

union = np.hstack((F1,F2,F3))
classify_color_feature(F1[:5000],y[:5000])
classify_color_feature(F2[:5000],y[:5000])
classify_color_feature(F3[:5000],y[:5000])
classify_color_feature(F1[:10000],y[:10000])
classify_color_feature(F2[:10000],y[:10000])
classify_color_feature(F3[:10000],y[:10000])
