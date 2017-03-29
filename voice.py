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

#------ Importing the DataSet and Separating the Independent and Dependent Variables
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

#------Scatter plot of given features
# We can compare other features by simply change "meanfun" and "meanfreq"
sns.FacetGrid(dataset, hue="label", size=5)\
   .map(plt.scatter, "meanfun", "meanfreq")\
   .add_legend()
plt.show()

#------Boxplot
#We can visualize other features by substituting "meanfun"
sns.boxplot(x="label",y="meanfun",data=dataset)
plt.show()

#Distribution of male and female(every feature)
sns.FacetGrid(dataset, hue="label", size=6) \
   .map(sns.kdeplot, "meanfun") \
   .add_legend()
plt.show()

#Radviz circle 
#Good to compare every feature
from pandas.tools.plotting import radviz
radviz(dataset, "label")
plt.show()

#####################################################
#	                                                 #
#        Starting with Sets and Pre-Processing      #
#	                                                 #
#####################################################

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
# If we don´t know the labels or are too many, we can use 'dataset["label"].value_counts()'


#------ Encoding Categorical Data of the Dependent Variable
# male -> 1
# female -> 0
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#------ Splitting the Dataset into the Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#------- Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

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

#####################################################
#	                                                 #
#          K-Nearest Neighbors (K-NN)               #
#	                                                 #
#####################################################

# Fitting classifier to the Training set
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

	 
#####################################################
#	                                                 #
#                Kernel SVM - Default               #
#	                                                 #
#####################################################

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#------- Calculating the Performance of Kernel SVM
from sklearn import metrics
print( "Accuracy of Kernel SVM - Default: {}".format(metrics.accuracy_score(y_test,y_pred)) ) # 0.9747634069400631
 
	 
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

#####################################################
#	                                                 #
#                    Naive Bayes                    #
#	                                                 #
#####################################################


#####################################################
#	                                                 #
#             Decision Tree Classification          #
#	                                                 #
#####################################################


#####################################################
#	                                                 #
#            Random Forest Classification           #
#	                                                 #
#####################################################



