#!/usr/bin/python

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from multiprocessing import Pool
import time
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report

def import_files(dog_img_list_file, cat_img_list_file, test_img_list_file):

    with open(dog_img_list_file) as dog_img_list:
         dog_imgs  = [line.rstrip('\n') for line in dog_img_list]
    with open(cat_img_list_file) as cat_img_list:
         cat_imgs  = [line.rstrip('\n') for line in cat_img_list]
    with open(test_img_list_file) as test_img_list:
         test_imgs  = [line.rstrip('\n') for line in test_img_list]

    return dog_imgs, cat_imgs, test_imgs

def map_sift_desc(img):

    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(gray, None)

    return desc

def reduce_sift_desc(mapping):

    return reduce(lambda x, y: np.concatenate((x, y), axis = 0), mapping)

def get_hist_feature(sift_features, predicted_labels):
    feature_num = [f.shape[0] for f in sift_features]
    hist = np.zeros(shape = (len(feature_num), 1000))
    for i, num in enumerate(feature_num):
        labels = predicted_labels[:num]
        for label in labels:
            hist[i, label] = hist[i, label] + 1
        predicted_labels = predicted_labels[num:]
    return hist

def classify(train_features, train_labels, test_features):
    clf = svm.SVC(C = 0.005, kernel = 'linear', )
    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    return predicted_labels

def main():

    p = Pool(8)

    #dog_images, cat_images, test_images = import_files('small_dog_img',
    #'small_cat_img', 'small_test_img')
    dog_images, cat_images, test_images = import_files('dog_img', 'cat_img',
    'test_img')
    n_dog = len(dog_images)
    n_cat = len(cat_images)
    n_train = n_dog + n_cat
    n_test = len(test_images)
    all_images = np.concatenate((dog_images, cat_images, test_images), axis = 0)
    n_all = all_images.shape[0]
    sift_start = time.time()
    sift_features = p.map(map_sift_desc, all_images)
    sift_end = time.time()
    print (sift_end - sift_start)*1000
    train_sift_features = reduce_sift_desc(sift_features[: n_train])
    test_sift_features = reduce_sift_desc(sift_features[n_train :])
    kmeans_start = time.time()
    kmeans = MiniBatchKMeans(n_clusters = 1000, batch_size = 1000, max_iter = 250)
    kmeans.fit(train_sift_features)
    train_predicted_labels = kmeans.predict(train_sift_features)
    test_predicted_labels = kmeans.predict(test_sift_features)
    kmeans_end = time.time()
    print (kmeans_end - kmeans_start)*1000
    '''
    hist_start = time.time()
    train_hist_features = get_hist_feature(sift_features[: n_train],
            train_predicted_labels)
    test_hist_features = get_hist_feature(sift_features[n_train :],
            test_predicted_labels)
   #hist_end = time.time()
   #print (hist_end - hist_start)*1000
    train_labels = np.concatenate((np.ones(n_dog), np.zeros(n_cat)), axis = 0)
   #svm_start = time.time()
    pred = classify(train_hist_features, train_labels, test_hist_features)
   #svm_end = time.time()
   #print (svm_end - svm_start)*1000
  # true = np.loadtxt('small_test_label')
  # print classification_report(true, pred)
  # print confusion_matrix(true, pred)
    out = pd.DataFrame(pred, columns = ['label'])
    out = out.astype(int)
    out.index += 1
    out.to_csv('sub1.csv', index_label = 'id')
    '''

if __name__ == "__main__":
    main()
