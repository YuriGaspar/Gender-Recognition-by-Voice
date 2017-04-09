# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:05:38 2017

@author: Yuri Gaspar

##########################################################
#	                                                      #
#              Gender Recognition by Voice               #
#   (https://www.kaggle.com/primaryobjects/voicegender)  #
#	                                                      #
##########################################################

"""

#------- Importing the Libraries 
import numpy as np
np.set_printoptions(threshold=np.inf) #Showing all the array in Console
import matplotlib.pyplot as plt
import pandas as pd

#------ Importing the DataSet 
dataset = pd.read_csv('voice.csv')
dataset.corr()
dataset.head()

#####################################################
#	                                                 #
#        Taking out some Plots of our Dataset       #
#	                                                 #
#####################################################

#------ Pearson Correlation Heatmap
import seaborn as sns
colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
plt.title('Pearson Correlation', y=1.05, size=15)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
sns.heatmap(dataset.iloc[:,:-1].astype(float).corr(), linewidths=0.3, vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True, annot_kws={"size": 7})

#------ Scatter plot of given features
# We can compare other features by simply change "meanfun" and "meanfreq"
sns.FacetGrid(dataset, hue="label", size=5)\
   .map(plt.scatter, "meanfun", "meanfreq")\
   .add_legend()
plt.show()

#------ Boxplot
# We can visualize other features by substituting "meanfun"
sns.boxplot(x="label",y="meanfun",data=dataset)
plt.show()

#------ Distribution of male and female(every feature)
# We can visualize other features by substituting "meanfun"
sns.FacetGrid(dataset, hue="label", size=6) \
   .map(sns.kdeplot, "meanfun") \
   .add_legend()
plt.show()

#------ Radviz circle 
# Good to compare every feature
from pandas.tools.plotting import radviz
radviz(dataset, "label")
plt.show()

#####################################################
#	                                                 #
#        Starting with Sets and Pre-Processing      #
#	                                                 #
#####################################################

#------ Separating the Independent and Dependent Variables
# Getting all Columns, except the last one with the genders
X = dataset.iloc[:, : -1].values
# Getting the last column
y = dataset.iloc[:, 20].values 

#------ Taking Care of Missing Data
# dataset.isnull().sum() 
# No Need of Taking Care of Missing Data :)

#------ Checking the Number of Male and Females
print("Number of Males {}".format(dataset[dataset.label == 'male'].shape[0])) # shape returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n. 
print("Number of Females {}".format(dataset[dataset.label == 'female'].shape[0]))
# If we don´t know the labels or they are too many, we can use 'dataset["label"].value_counts()'

#------ Encoding Categorical Data of the Dependent Variable
# male -> 1
# female -> 0
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#------ Splitting the Dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#------- Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating a Dictionaire
model_accuracy = {}

#####################################################
#	                                                 #
#                Logistic Regression                #
#	                                                 #
#####################################################

#------- Fitting Logistic Regression to the Training Set 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#------- Predicting the Test Set Results
y_pred = classifier.predict(X_test)

#------- Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Logistic Regression
from sklearn import metrics
print( "Accuracy of Logistic Regression: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.971608832808

model_accuracy['Logistic Regression'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#          K-Nearest Neighbors (K-NN)               #
#	                                                 #
#####################################################

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2 )
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

#------- Calculating the Performance of K-NN
from sklearn import metrics
print( "Accuracy of K-NN: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.965299684543

model_accuracy['K-NN'] = metrics.accuracy_score(y_test,y_pred)
	 
#####################################################
#	                                                 #
#                  Kernel SVM - RBF                 #
#	                                                 #
#####################################################

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Kernel SVM
from sklearn import metrics
print( "Accuracy of Kernel SVM - RBF: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.9747634069400631
	 
model_accuracy['Kernel SVM - RBF'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#                 Kernel SVM - Linear               #
#	                                                 #
#####################################################

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of SVM
from sklearn import metrics
print( "Accuracy of Kernel SVM - Linear: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.9747634069400631

model_accuracy['Kernel SVM - Linear'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#                    Naive Bayes                    #
#	                                                 #
#####################################################

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Naive Bayes
from sklearn import metrics
print( "Accuracy of Naive Bayes Model: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.8990536277602523

model_accuracy['Naive Bayes'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#             Decision Tree Classification          #
#	                                                 #
#####################################################

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Decision Tree Classification
from sklearn import metrics
print( "Accuracy of Decision Tree Classification: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.9463722397476341

model_accuracy['Decision Tree'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#            Random Forest Classification           #
#	                                                 #
#####################################################

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=250, criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Random Forest Classification
from sklearn import metrics
print( "Accuracy of Random Forest Classification: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.9779179810725552

model_accuracy['Random Forest'] = metrics.accuracy_score(y_test,y_pred)

#####################################################
#	                                                 #
#                 Feature Importances               #
#	                                                 #
#####################################################

header = list(dataset)

classifier.fit(X_train, y_train)
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1] # list in reverse order

# Print the feature ranking
print("Feature ranking:")
header1 = []
for f in range(X.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, header[indices[f]], importances[indices[f]]))
    header1.append(header[indices[f]])
    
# Plot the feature importances of the forest with Material Design

import random 
colors = ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#3F51B5', '#2196F3',
          '#03A9F4' ,'#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B',
          '#FFC107', '#FF9800', '#FF5722', '#795548', '#607D8B', '#B71C1C', '#880E4F',
          '#4A148C', '#311B92', '#1A237E', '#0D47A1', '#01579B', '#006064', '#004D40',
          '#1B5E20', '#33691E', '#827717', '#F57F17', '#FF6F00', '#E65100', '#BF360C',
          '#3E2723', '#212121', '#607D8B']

random_colors = random.sample(colors, 20)

plt.figure()
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color=random_colors, align="center")
plt.xticks(range(X.shape[1]),header1, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

#####################################################
#	                                                 #
#                  Models Accuracy                  #
#	                                                 #
#####################################################

from collections import OrderedDict
model_accuracy = OrderedDict(sorted(model_accuracy.items(), key=lambda t: t[1]))

plt.figure()
plt.gcf().subplots_adjust(left=0.22)
plt.title("Models Accuracy")
plt.barh(range(len(model_accuracy)), model_accuracy.values(), align='center', color='#009688')
plt.yticks(range(len(model_accuracy)), model_accuracy.keys(), rotation = 0)
axes = plt.gca()
axes.set_xlim([0.8,1.0])
plt.xlabel("Accuracy")
plt.ylabel('Classifier')
plt.show()
