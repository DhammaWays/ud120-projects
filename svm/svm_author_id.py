#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
#sys.path.append("../tools/")
sys.path.append("..\\tools\\")
sys.path.append("..\\choose_your_own\\")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from class_vis import prettyPicture
import matplotlib.pyplot as plt

#C = 0.0, 100., 1000., and 10000.
clf = SVC(kernel="rbf", C=10000.)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

print "Accuracy:", accuracy_score(labels_test, pred)

print "Number of Chris predicted:", sum(pred)


#prettyPicture(clf, features_train, labels_train)
#plt.show()

#########################################################


