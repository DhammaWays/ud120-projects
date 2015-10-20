#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

def getClassificationAccuracy( clf ):
    #from sklearn import tree
    from sklearn.metrics import accuracy_score    
    #clf = tree.DecisionTreeClassifier( **kwargs )
    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print "Training time:", round(time()-t0, 3), "s", "Sample Size, Features:", len(features_train), len(features_train[0])
    
    t0 = time()
    pred = clf.predict( features_test ) 
    print "Prediction Time time:", round(time()-t0, 3), "s", "Sample Size, Features:", len(features_test), len(features_test[0])
    
    return accuracy_score( labels_test, pred )

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

clf = KNeighborsClassifier(n_neighbors=15, weights='distance') # Accuracy = 0.94
#clf = RandomForestClassifier(n_estimators=20, min_samples_split=15, max_features="log2") # accuracy = 0.92
#clf = AdaBoostClassifier(n_estimators=20, learning_rate=2.0) # accuracy = 0.936

accuracy = getClassificationAccuracy(clf)
print "Accuracy:", accuracy


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
