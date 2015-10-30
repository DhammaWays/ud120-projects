#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

# split into training and test data
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


### it's all yours from here forward!  

### your code goes here ###
def getDecTreeAccuracy( **kwargs ):
    from sklearn import tree
    from sklearn.metrics import accuracy_score, precision_score, recall_score    
    clf = tree.DecisionTreeClassifier( **kwargs )
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict( features_test ) 
    print "Test Data: Predicted number of person of interest ", len(pred[pred == 1.0]), "out of total persons", len(pred)
    return [accuracy_score(labels_test, pred), precision_score(labels_test, pred), recall_score(labels_test, pred)]

#accuracy = getDecTreeAccuracy( min_samples_split=50 )
accuracy = getDecTreeAccuracy()
print "Accuracy, Precision, Recall scores:", accuracy
