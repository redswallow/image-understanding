#!/usr/bin/python

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from multiprocessing import Pool
from itertools import islice
import time

def import_files(dog_img_list_file, cat_img_list_file):
    dog_imgs  = ["train/dog.%s.jpg" %i for i in xrange(1,16)]
    cat_imgs  = ["train/cat.%s.jpg" %i for i in xrange(1,16)]

    return dog_imgs, cat_imgs

def map_sift_desc(img):
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(gray, None)
    return desc

def map_surf_desc(img):
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF()
    kp, desc = surf.detect(gray, None, useProvidedKeypoints = False)
    return desc


def online_mbk(sift_features):
    n = sift_features.shape[0]
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters = 400, batch_size = 400, max_iter = 100, random_state = rng, verbose = True)
    index = 0
    for _ in range(3):
        sift_features = shuffle(sift_features, n_samples = int(round(n*0.1)), random_state = rng)
        i = iter(sift_features)
        while True:
            index += 1
            print index*2500
            sublist = list(islice(i, 2500))
            if len(sublist) > 0:
                sublist = np.vstack(sublist)
                kmeans.partial_fit(sublist)
            else:
                break

    print "finished training"
    predicted_labels = kmeans.predict(sift_features)
    return predicted_labels

def get_hist_feature(sift_features, predicted_labels):
    feature_num = [f.shape[0] for f in sift_features]
    hist = np.zeros(shape = (len(feature_num), 400))
    for i, num in enumerate(feature_num):
        labels = predicted_labels[:num]
        for label in labels:
            hist[i, label] = hist[i, label] + 1
        predicted_labels = predicted_labels[num:]
    return hist


def main():
    n_cpu = 2
    p = Pool(n_cpu)

    dog_images, cat_images = import_files('dog_img', 'cat_img')
    n_dog = len(dog_images)
    n_cat = len(cat_images)
    n_all = n_dog + n_cat
    all_images = np.concatenate((dog_images, cat_images), axis = 0)
    all_labels = np.concatenate((np.ones(n_dog), np.zeros(n_cat)), axis = 0)
    print "begin sift feature extraction"
    sift_start = time.time()
    sift_features = p.map(map_sift_desc, all_images)
    sift_end = time.time()
    print (sift_end - sift_start)
    print "stacking features"
    stack_start = time.time()
    all_sift_features = np.vstack(sift_features)
    stack_end = time.time()
    print (stack_end - stack_start)
    print "begin mini batch kmeans"
    kmeans_start = time.time()
    all_predicted_labels = online_mbk(all_sift_features)
    kmeans_end = time.time()
    print (kmeans_end - kmeans_start)
    print "begin histogram of features"
    hist_start = time.time()
    all_hist_features = get_hist_feature(sift_features,
            all_predicted_labels)
    hist_end = time.time()
    print (hist_end - hist_start)

    X_train, X_test, Y_train, Y_test = train_test_split(all_hist_features, all_labels, test_size = 0.2, random_state = 50)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
    svc = svm.SVC()
    print "begin grid search with cross validation"
    grid_start = time.time()
    clf = grid_search.GridSearchCV(svc, tuned_parameters, cv = 5, n_jobs = n_cpu)
    clf.fit(X_train, Y_train)
    grid_end = time.time()
    print (grid_end - grid_start)
    print clf.best_estimator_
    print clf.best_score_
    print clf.best_params_
    print "begin fitting test data"
    fit_start = time.time()
    clf_best = svm.SVC(C = 1, gamma = 0.0001, kernel = 'rbf')
    clf_best.fit(X_train, Y_train)
    Y_pred1 = clf_best.predict(X_train)
    Y_pred2 = clf_best.predict(X_test)
    fit_end = time.time()
    print (fit_end - fit_start)
    print classification_report(Y_train, Y_pred1)
    print classification_report(Y_test, Y_pred2)
    print clf_best.score(X_train, Y_train)
    print clf_best.score(X_test, Y_test)


if __name__ == "__main__":
    main()
