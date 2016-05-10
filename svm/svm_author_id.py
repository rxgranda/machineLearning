#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

from sklearn import svm

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)
#clf.predict([[2., 2.]])
# ESTO ES PARA SELECCIONAR SOLO EL 1 % DEL DATASET
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#Linear KErnel
#clf = svm.LinearSVC()

# Kernel RBF
clf = svm.SVC(kernel='rbf', C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

print "Score: ",clf.score(features_test, labels_test)

t0 = time()
#print clf.predict([features_test[50]])
count=0
for key in features_test:
    prediction=clf.predict([key])
    if prediction[0]==1:
        count = count+1
print "TOtal 1: ", count, "from: ", len(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

#print clf.score(features_test,labels_test)

#########################################################
