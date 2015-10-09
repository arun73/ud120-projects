#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
import numpy as np

alg = GaussianNB()

t0 = time()
alg.fit(features_train, labels_train)
print "Time for training: ", time()-t0, "s"

t1 = time()
labels_pred = alg.predict(features_test)
print "Time for prediction: ", time()-t1, "s"

print("Accuracy: " + str(np.mean(labels_pred == labels_test)))

#########################################################


