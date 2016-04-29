#!/usr/bin/python

# Lekhraj Sharma
# Data Analyst Nanodegree
# P5: Machine Learning Final Project
# Person of Interest Identifier for Enron DataSet
# April 2016

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Tried following feature lists for feature selection using GaussianNB
# Performance with GaussioanNB is given just below easch feature combination
#
#features_list = ['poi','salary'] # You will need to use more features
#Accuracy: 0.71875 Precision: 0.25 Recall: 0.142857142857

#features_list = ['poi','salary', 'total_payments', "bonus", 
# "exercised_stock_options", "total_stock_value", "long_term_incentive",
# "expenses", "from_this_person_to_poi",  "from_poi_to_this_person",
# "shared_receipt_with_poi"  ]
#Accuracy: 0.895833333333 Precision: 0.6 Recall: 0.5

#features_list = ['poi','salary', 'total_payments', "from_this_person_to_poi",
# "from_poi_to_this_person"  ] 
#Accuracy: 0.863636363636 Precision: 0.0 Recall: 0.0

#features_list = ['poi','salary', 'total_stock_value', 
# "from_this_person_to_poi", "from_poi_to_this_person"  ]
#Accuracy: 0.818181818182 Precision: 0.2 Recall: 0.2 

#features_list = ['poi','salary', 'total_stock_value', 
# "from_this_person_to_poi", "from_poi_to_this_person",
# "shared_receipt_with_poi"  ] 
#Accuracy: 0.840909090909 Precision: 0.4 Recall: 0.333333333333

#features_list = ['poi','salary', "bonus", "from_this_person_to_poi",
# "from_poi_to_this_person", "shared_receipt_with_poi"  ] 
#Accuracy: 0.763157894737 Precision: 0.166666666667 Recall: 0.2

#features_list = ['poi','salary', "bonus", "shared_receipt_with_poi"  ] 
#Accuracy: 0.763157894737 Precision: 0.25 Recall: 0.4

#features_list = ['poi','salary', 'total_stock_value', 
# "from_this_person_to_poi", "from_poi_to_this_person",
# "shared_receipt_with_poi"  ]
#Accuracy: 0.840909090909 Precision: 0.4 Recall: 0.333333333333

#features_list = ['poi','salary', 'total_stock_value', 
# 'shared_receipt_with_poi', 'bonus' ] 
#Accuracy: 0.840909090909 Precision: 0.428571428571 Recall: 0.5

#features_list = ['poi','salary', 'total_stock_value', 
# 'shared_receipt_with_poi', 'bonus', 'expenses', 'to_messages' ] 
#Accuracy: 0.934782608696 Precision: 0.5 Recall: 0.666666666667

# Finally settled on the following feature list as it has right balance
# with minimum number of features with higher performance
#
features_list = ['poi','salary', 'total_stock_value', 
                 'shared_receipt_with_poi', 'bonus', 'expenses' ]
#Accuracy: 0.913043478261 Precision: 0.4 Recall: 0.666666666667
        
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#
# Remove entry TOTAL which is just a total from spreadsheet
data_dict.pop( "TOTAL", 0 ) 

# Tried to remove all data items which di dnot have salary avaialble (NaN)
# but did NOT use it as it did not make much difference  
# 
#data_dict = 
# {k:data_dict[k] for k in data_dict if type(data_dict[k]['salary']) is int}


### Task 3: Create new feature(s)

# Created a new feature called 'high_worth"
# which is a proxy to high net worth of indvidual (Salary+bonus > 1M)
#
# Did not end up using it as it did not make much difference in performance.
# Uncomment following block of code to see it in action
#
#for key in data_dict:    
#  if (type(data_dict[key]['salary']) is int and \
#        type(data_dict[key]['bonus']) is int) and \
#       (data_dict[key]['salary'] + data_dict[key]['bonus'] >= 1000000):
#       data_dict[key]['high_worth'] = 1
# else:
#      data_dict[key]['high_worth'] = 0
#
#features_list.append('high_worth')  
#
# Performance with new feature 'high_worth" with GaussioanNB:
# Accuracy: 0.913043478261 Precision: 0.4 Recall: 0.666666666667
        
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Did try to scaling of feature but did not use later on as finally chosen
# Decision Tree classifer is not impacted by scaling.
#
# Scale features to 0.0 to 1.0, no impact on decision tree classifiers
#
#from sklearn import preprocessing
#scaler = preprocessing.MinMaxScaler() 
#features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#Accuracy: 0.913043478261 Precision: 0.4 Recall: 0.666666666667

# Following is peformance of above GaussioanNB reported by 'tester.py':
#Accuracy: 0.83571       Precision: 0.40578      Recall: 0.32300 F1: 0.35969

#from sklearn.svm import SVC
#clf = SVC()
#Accuracy: 0.934782608696 Precision: 0.0 Recall: 0.0

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
#Accuracy: 0.913043478261 Precision: 0.0 Recall: 0.0

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
#Accuracy: 0.934782608696 Precision: 0.0 Recall: 0.0

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier()
#Accuracy: 0.869565217391 Precision: 0.2 Recall: 0.333333333333

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier()
#Accuracy: 0.869565217391 Precision: 0.2 Recall: 0.333333333333

#clf = AdaBoostClassifier(n_estimators=20, learning_rate=2.0)
#Accuracy: 0.869565217391 Precision: 0.2 Recall: 0.333333333333

#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=None, min_samples_split=1,
#                            random_state=0) )
#Accuracy: 0.826086956522 Precision: 0.142857142857 Recall: 0.333333333333
            
#
from sklearn import tree  
#clf = tree.DecisionTreeClassifier()
#Accuracy: 0.847826086957 Precision: 0.25 Recall: 0.666666666667
 
#clf = tree.DecisionTreeClassifier(criterion="gini", max_features=None, 
#       max_depth=None, min_samples_split=1, random_state=0, splitter ="best")
#Accuracy: 0.847826086957 Precision: 0.166666666667 Recall: 0.333333333333

#clf = tree.DecisionTreeClassifier(random_state=0)
#Accuracy: 0.847826086957 Precision: 0.166666666667 Recall: 0.333333333333

clf = tree.DecisionTreeClassifier(random_state=0, min_samples_split=2)
#Accuracy: 0.847826086957 Precision: 0.166666666667 Recall: 0.333333333333

# Following is performance as reported by 'tester.py':
#Accuracy: 0.81871       Precision: 0.37970      Recall: 0.42450 F1: 0.40085

#
# Tried to find the best parameters for Decision Tree using GridSearchCV
# But did not end up using it as best parameters returned were not very
# different from hand tuned and performance as reported by 'tester.py' of
# hand tuned parameters was little better as well it rand fast as well!
#
#from sklearn import grid_search
#parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'),
#              'random_state':[0,10], 'min_samples_split':[1,8]}
#clf_grid = tree.DecisionTreeClassifier()
#clf = grid_search.GridSearchCV( clf_grid, parameters)

#
# Following performance was reported by 'tester.py':
#Accuracy: 0.82579       Precision: 0.36081      Recall: 0.28450 F1: 0.31814 


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.33, random_state=42)
    

#
### Accuracy ###
from sklearn.metrics import accuracy_score, precision_score, recall_score    
clf = clf.fit(features_train, labels_train)
pred = clf.predict( features_test ) 
print "Test Data: Predicted number of poi ", len(pred[pred == 1.0]), \
      "out of total persons", len(pred)
print "Accuracy:", accuracy_score(labels_test, pred), \
      "Precision:", precision_score(labels_test, pred), \
      "Recall:", recall_score(labels_test, pred)
# 
#Uncomment following for GridSearchCV based auto selection of best parameters
#print clf.best_params_
#print clf.best_score_

#Accuracy: 0.869565217391 Precision: 0.285714285714 Recall: 0.666666666667
#{'min_samples_split': 8, 'splitter': 'random', 'random_state': 10,
# 'criterion': 'entropy'}
#0.827956989247
#


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)