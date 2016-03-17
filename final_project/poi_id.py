#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary']
# Accuracy: 0.25560       Precision: 0.18481      Recall: 0.79800 F1: 0.30011
#features_list = ['poi','salary', 'total_payments', "bonus", "exercised_stock_options", "total_stock_value", "long_term_incentive", "expenses", "from_this_person_to_poi",  "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.20487       Precision: 0.11431      Recall: 0.73550 F1: 0.19786
#features_list = ['poi','salary', 'total_payments', "total_stock_value", "from_this_person_to_poi",  "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.62400       Precision: 0.15543      Recall: 0.41050 F1: 0.22549
#features_list = ['poi','salary', 'total_payments', "from_this_person_to_poi", "from_poi_to_this_person"  ] # You will need to use more features
# Accuracy: 0.63721       Precision: 0.11387      Recall: 0.22700 F1: 0.15166 
#features_list = ['poi','salary', 'total_stock_value', "from_this_person_to_poi", "from_poi_to_this_person"  ] # You will need to use more features
# Accuracy: 0.49807       Precision: 0.16321      Recall: 0.60900 F1: 0.25742
#features_list = ['poi','salary', 'total_stock_value', "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.60064       Precision: 0.15884      Recall: 0.41800 F1: 0.23021 
# features_list = ['poi','total_payments', "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.77529       Precision: 0.05304      Recall: 0.03400 F1: 0.04144
#features_list = ['poi','salary', "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
#  Accuracy: 0.65392       Precision: 0.14342      Recall: 0.21650 F1: 0.17254
#features_list = ['poi','salary', "bonus", "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.42050       Precision: 0.16050      Recall: 0.58550 F1: 0.25194 
#features_list = ['poi','salary', "bonus", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.21125       Precision: 0.14368      Recall: 0.75250 F1: 0.24128 

#features_list = ['poi','salary', 'total_stock_value', "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.60064       Precision: 0.15884      Recall: 0.41800 F1: 0.23021 
# Total predictions:14000 True positives:836 False positives:4427 False negatives:1164 True negatives: 7573

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'exercised_stock_options' ] # You will need to use more features
#Accuracy: 0.79414       Precision: 0.29393      Recall: 0.31450 F1: 0.30386     F2: 0.31016
#        Total predictions: 14000        True positives:  629    False positives: 1511   False negatives: 1371   True negatives: 10489

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus' ] # You will need to use more features
# Accuracy: 0.79857       Precision: 0.30269      Recall: 0.31450 F1: 0.30848     F2: 0.31207
# Total predictions: 14000        True positives:  629    False positives: 1449   False negatives: 1371   True negatives: 10551

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'expenses', 'from_poi_to_this_person' ] # You will need to use more features
#Accuracy: 0.80879       Precision: 0.35002      Recall: 0.39500 F1: 0.37115     F2: 0.38510
#Total predictions: 14000        True positives:  790    False positives: 1467   False negatives: 1210   True negatives: 10533

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'expenses', 'to_messages' ] # You will need to use more features
#Accuracy: 0.81114       Precision: 0.35417      Recall: 0.39100 F1: 0.37167     F2: 0.38303
#Total predictions: 14000        True positives:  782    False positives: 1426   False negatives: 1218   True negatives: 10574

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'expenses', 'loan_advances' ] # You will need to use more features
#Accuracy: 0.81493       Precision: 0.36501      Recall: 0.39950 F1: 0.38148     F2: 0.39209
#Total predictions: 14000        True positives:  799    False positives: 1390   False negatives: 1201   True negatives: 10610

#features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'expenses' ] # You will need to use more features
#Accuracy: 0.81457       Precision: 0.36601      Recall: 0.40700 F1: 0.38542     F2: 0.39808
#Total predictions: 14000        True positives:  814    False positives: 1410   False negatives: 1186   True negatives: 10590

features_list = ['poi','salary', 'total_stock_value', 'shared_receipt_with_poi', 'bonus', 'expenses' ] # You will need to use more features
        
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers


### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# GaussianNB: Accuracy: 0.60064       Precision: 0.15884      Recall: 0.41800 F1: 0.23021
#
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
#Accuracy: 0.84086       Precision: 0.05118      Recall: 0.00650 F1: 0.01154

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=20, learning_rate=2.0)
#Accuracy: 0.62343       Precision: 0.14998      Recall: 0.35050 F1: 0.21007 
#Accuracy: 0.75157       Precision: 0.26863      Recall: 0.42900 F1: 0.33038     F2: 0.38324
#Total predictions: 14000        True positives:  858    False positives: 2336   False negatives: 1142   True negatives: 9664

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0) )
#Accuracy: 0.81579       Precision: 0.36787      Recall: 0.40300 F1: 0.38463     F2: 0.39545
    
#
#from sklearn import tree  
#clf = tree.DecisionTreeClassifier()
# Accuracy: 0.78907       Precision: 0.22409      Recall: 0.19350 F1: 0.20767
#Accuracy: 0.81750       Precision: 0.37369      Recall: 0.41050 F1: 0.39123     F2: 0.40257
#Total predictions: 14000        True positives:  821    False positives: 1376   False negatives: 1179   True negatives: 10624

from sklearn import tree  
#clf = tree.DecisionTreeClassifier(criterion="gini", max_features=None, max_depth=None, min_samples_split=1, random_state=0, splitter ="best")

clf = tree.DecisionTreeClassifier(random_state=0)
#Accuracy: 0.81779       Precision: 0.37684      Recall: 0.42150 F1: 0.39792     F2: 0.41174
#Total predictions: 14000        True positives:  843    False positives: 1394   False negatives: 1157   True negatives: 10606

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10, max_depth=None,
#         min_samples_split=1, random_state=0)
#Accuracy: 0.83693       Precision: 0.32290      Recall: 0.12900 F1: 0.18435     F2: 0.14661

#from sklearn.svm import SVC
#clf = SVC()
# getting divide by zero error in SVC

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
 #      max_depth=1, random_state=0)
#Accuracy: 0.82607       Precision: 0.35662      Recall: 0.27050 F1: 0.30765     F2: 0.28423
#Total predictions: 14000        True positives:  541    False positives:  976   False negatives: 1459   True negatives: 11024


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
    
# PCA
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=2, whiten=True).fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)
#print pca
#print pca.explained_variance_ratio_,pca.components_, pca.mean_


#
### Accuracy ###
from sklearn.metrics import accuracy_score, precision_score, recall_score    
clf = clf.fit(features_train, labels_train)
pred = clf.predict( features_test ) 
print "Test Data: Predicted number of poi ", len(pred[pred == 1.0]), "out of total persons", len(pred)
print "Accuracy:", accuracy_score(labels_test, pred), "Precision:", precision_score(labels_test, pred), "Recall:", recall_score(labels_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)