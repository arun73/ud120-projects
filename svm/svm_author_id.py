#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

clf = SVC(kernel="rbf", C=10000)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)

print "Time for training: ", time()-t0, "s"


t1 = time()
preds = clf.predict(features_test)
print "Time for prediction: ", time()-t1, "s"

print "Accuracy: ", accuracy_score(labels_test, preds)


print "Prediction for element #10: ", preds[10]
print "Prediction for element #26: ", preds[26]
print "Prediction for element #50: ", preds[50]

print "Number of test cases predicted as Chris(1): ", np.sum(preds)
#########################################################
