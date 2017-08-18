"""
This program is performs activity classification by applying a clustering algorithm (Fuzzy C-Means clustering)
to find similar patterns in an activity and group the activties for further prediction using an unseen data.

To run the program: a CSV file containing joint coordinates of activities and their corresponding labels is provided.
Data within the file = x,y,z coordinates of 15 joints, i.e. 3*15 = 45 columns. Plus a last column of activity label.

The number of clusters/classes/activities within the file is required
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
from resultMetrics import plot_confusion_matrix
from sklearn import preprocessing, model_selection, svm, neighbors, ensemble
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from Features import features
from sklearn.decomposition import PCA, KernelPCA
import time
import pickle
from sklearn.externals import joblib


start_time = time.time()
"""
Load features from CSV file
"""
filename = 'CAD-601.csv'
FEATURES, act_label = features(filename)

"""
Normalize features to zero mean and unit variance
"""
FEATURES_norm = (FEATURES - FEATURES.mean()) / FEATURES.std()
FEATURES_norm =FEATURES_norm.fillna(0)
#print(FEATURES_norm)
train_act, test_act, train_label, test_label = model_selection.train_test_split(FEATURES_norm, act_label, test_size=0.3)
#print(act_label)

#implement pca algorithm on data
#pca = PCA(n_components=99)
#pca.fit(FEATURES_norm)
#print(pca.explained_variance_ratio_)

"""
split features for prediction test
Actual training activity data = TRAIN_act
Training label = TRAIN_label
Test activity data = test_act
Test activity label = test_label

#predicting new activity
'pred_test' and 'pred_label' are used for predicting out the classifier
after training and testing are done
"""
#TRAIN_act, pred_test, TRAIN_label, pred_label = model_selection.train_test_split(train_act, train_label, test_size=0.1)
#discard rows in TRAIN_act with random activities
# training_data = pd.DataFrame(TRAIN_act)
# training_data['label'] = pd.DataFrame(TRAIN_label)
# training_data = training_data[training_data.label != 13]    #drop all rows with random activity data
# TRAIN_act = training_data.drop(['label'], axis=1)
# TRAIN_label = training_data.label




"""
SVM
"""
classifier = svm.SVC(decision_function_shape='ovo', kernel='rbf', probability=True)
classifier.fit(train_act, train_label)
_accuracy = classifier.score(test_act, test_label)
print(_accuracy)

joblib.dump(classifier, 'svm_model.pkl')    #save model to variable which will be used in classifying incoming kinect data


# #predict outcome using prediction data
prediction = classifier.predict(test_act)
pred_prob = classifier.predict_proba(test_act)
print(pred_prob)
# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, classification_report(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# #print(cnf_matrix.index)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

"""
K-nearest neighbour
"""
# classifier = neighbors.KNeighborsClassifier()
# classifier.fit(train_act, train_label)
# _accuracy = classifier.score(test_act, test_label)
# print(_accuracy)
#
# #predict outcome using prediction data
# prediction = classifier.predict(test_act)
# #print(test_label)
# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, precision_recall_fscore_support(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# #print(cnf_matrix.index)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

"""
Random Forest
"""

# classifier = ensemble.RandomForestClassifier()
# classifier.fit(train_act, train_label)
# _accuracy = classifier.score(test_act, test_label)
# print(_accuracy)
#
# #predict outcome using prediction data
# prediction = classifier.predict(test_act)
# #print(test_label)
# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, precision_recall_fscore_support(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
#
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

print("... %s seconds..." % (time.time() - start_time))