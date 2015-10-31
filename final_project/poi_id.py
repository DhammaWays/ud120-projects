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

features_list = ['poi','salary', 'total_stock_value', "from_this_person_to_poi", "from_poi_to_this_person", "shared_receipt_with_poi"  ] # You will need to use more features
# Accuracy: 0.60064       Precision: 0.15884      Recall: 0.41800 F1: 0.23021 
# Total predictions:14000 True positives:836 False positives:4427 False negatives:1164 True negatives: 7573

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

from sklearn import tree  
clf = tree.DecisionTreeClassifier()
# Accuracy: 0.78907       Precision: 0.22409      Recall: 0.19350 F1: 0.20767

#from sklearn.svm import SVC
#clf = SVC()
# getting divide by zero error in SVC

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
# PCA
#from sklearn.decomposition import RandomizedPCA
#pca = RandomizedPCA(n_components=4, whiten=True).fit(features_train)
#features_train = pca.transform(features_train)
#features_test = pca.transform(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)