#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries to retrieve data and read csv and to perform EDa tasks
import pandas as pd
import numpy as np
#importing for visualisation
import seaborn as sns


# In[2]:


#import dataset
data = pd.read_csv("C://Users//Lenovo//Downloads//Customer_Churn-1.csv")


# In[3]:


#Top 10 recordings
data.head(10)


# In[4]:


#descriptive stats for numeric columns
data.describe().T


# In[5]:


#$checking null Values in the data
data.isnull().sum()


# In[6]:


data.info()


# In[7]:


#changing total charges to numeric as it is continuos variable
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')

#errors = 'coerce' : It will ignore all non-numeric values.It will replace all non-numeric values with NaN.


# In[8]:


#drop nan values
data = data.dropna(how='any', axis = 0)

# axis=0  will be removing rows from dataset.
# axis=1 will be removing columns


# In[9]:


data = data.reset_index()
data

#Pandas reset_index() is a method to reset index of a Data Frame. 
#reset_index() method sets a list of integer ranging from 0 to length of data as index.


# In[10]:


#extratcing columns names with "object" Datatype
cols = data.select_dtypes(include=['object']).columns
cols
# 17 object data types are present


# In[11]:


#copying datset
data2 = data.copy()


# In[12]:


# now changing data via label encoding/....
# so thay we can apply for model
from sklearn.preprocessing import LabelEncoder

#making instance of labelnecoder
le = LabelEncoder()

#Label Encoding is a popular encoding technique for handling categorical variables.
#In this technique, each label is assigned a unique integer based on alphabetical ordering

#fitting and transforming
for col in cols:
    data2[col] = le.fit_transform(data2[col].astype(str))
# checking the datatypes chnges or not
print (data2.info())


# In[13]:


data2.head()


# In[14]:


import pandas_profiling


# In[15]:


pandas_profiling.ProfileReport(data2)


# In[16]:


def var_summary(x):
    uc = x.mean()+(2*x.std())
    lc = x.mean()-(2*x.std())
    
    for i in x:
        if i<lc or i>uc:
            count = 1
        else:
            count = 0
    outlier_flag = count
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max() , lc , uc,outlier_flag],
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX','LC','UC','outlier_flag'])




# In[17]:


data2.apply(lambda x: var_summary(x)).T


# # Linear Regression

# 
# > Linear regression is an approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression <br>
# > A simple linear regression model is given by Y=mX+b <br>
# > where m is the slope and b is the y-intercept. Y is the dependent variable and X is the explanatory variable. <br>
# > Very briefly and simplistically, Linear Regression is a class of techniques for fitting a straight line to a set of data points

# A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0).

# In[18]:


X = data2.drop('TotalCharges', axis = 1)


# In[19]:


y = data2['TotalCharges']


# In[20]:


data2.corr()


# In[21]:


# improting library for splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1111)


# In[22]:


#checking the shape of training and testing data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


# * Feature Scaling | Scaling the variables | standardizign the variable | Z - score | Mean = 0 and STD = 1
# To get all the variables on same scale [towards ZERO]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()



#The fit method is calculating the mean and variance of each of the features present in our train data.
#The transform method is transforming all the features using the respective mean and variance. 

X_train = sc.fit_transform(X_train)# you are finding the MEan and STD{with the fit()
#                                    }on training data and aslo transforming that

X_test= sc.transform(X_test)  # Only tranforming now


#transform method we can use the same mean and variance as it is calculated from our training data to transform our test data.


# In[24]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()  
lm.fit(X_train, y_train)


# In[25]:


print(lm.intercept_)


# In[26]:


# The coefficients
print('Coefficients', lm.coef_)


# In[27]:


#Testing
y_pred = lm.predict(X_test)


# In[28]:


y_pred = pd.DataFrame(y_pred, columns=['Predicted'])


# In[29]:


y_pred


# In[30]:


y_test


# ### Calculating mean square error ... RMSE
# > RMSE calculate the difference between the actual value and predicted value of the response(dependant) variable <br>
# > The square root of the mean/average of the square of all of the error. <br> 
# > Compared to the similar Mean Absolute Error, RMSE amplifies and severely punishes large errors. <br>
# > The lesser the RMSE value, the better is the model.

# In[31]:


from sklearn import metrics 
import numpy as np
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Evaluating Model Accuracy
# > R-squared is a statistical measure of how close the data are to the fitted regression line. <br>
# > R-square signifies percentage of variations in the reponse variable that can be explained by the model. <br>
# > R-squared = Explained variation / Total variation <br>
# > Total variation is variation of response variable around it's mean. <br>
# > R-squared value varies between 0 and 100%. 0% signifies that the model explains none of the variability, <br>
# > while 100% signifies that the model explains all the variability of the response. <br>
# > The closer the r-square to 100%, the better is the model. <br>

# In[32]:


#finding R-squared value
from sklearn.metrics import r2_score


# In[33]:


print(r2_score(y_test, y_pred))


# # Logistic Regression
# 

# Logistic Regression is used when the dependent variable(target) is categorical. logistic regression is a predictive analysis.
# Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more independent variables.

# In[34]:


#dividing independent and dependent variables
X1 = data2.iloc[:,:21]
y1 = data2.iloc[:,21:]


# In[35]:


X1.head()


# In[36]:


y1.head()


# In[37]:


# improting library for splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 1111)


# In[38]:


#importing lib for data to be align in standard scaling manner
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)


# In[39]:


#importing library for logistic regression
from sklearn.linear_model import LogisticRegression 


# In[40]:


#Creating intsnce of logistic regression and then fitting the data
logistic_reg= LogisticRegression()
logistic_reg.fit(X_train1,y_train1)


# In[41]:


#predicting the test data.....
y_pred1=logistic_reg.predict(X_test1)


# In[42]:


y_pred1


# In[43]:


#predcition on train data
y_pred_train = logistic_reg.predict(X_train)


# # Confusion matrix
# A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model. This gives us a holistic view of how well our classification model is performing and what kinds of errors it is making.

# In[44]:


#importing library for confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test1, y_pred1)
cnf_matrix


# In[45]:


#finding accuracy of model
log_acc = metrics.accuracy_score(y_test1, y_pred1)
print('Accuracy: ',log_acc)


# In[46]:


sns.heatmap(cnf_matrix, annot=True)


# In[47]:


#importing library for classification_report and finding report
from sklearn.metrics import classification_report
print(classification_report(y_test1,y_pred1))


# # Desicion Tree

# A decision tree is a tree-like graph with nodes representing the place where we pick an attribute and ask a question; edges represent the answers the to the question; and the leaves represent the actual output or class label. They are used in non-linear decision making with simple linear decision surface.

# **Advantages of decision trees:**
# 
# - Can be used for regression or classification
# - Can be displayed graphically
# - Highly interpretable
# - Can be specified as a series of rules, and more closely approximate human decision-making than other models
# - Prediction is fast
# - Features don't need scaling
# - Automatically learns feature interactions
# - Tends to ignore irrelevant features
# - Non-parametric (will outperform linear models if relationship between features and response is highly non-linear)
# - Robust to the outliers
# - Impact of Missing values is Minimal

# In[48]:


# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[49]:


# Spliting the overall data into X and Y [this is what required in ML]

X = data2.iloc[:, :21]

y = data2.iloc[:, 21:]


# In[50]:


X


# In[51]:


y


# In[52]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1111)


# In[53]:


from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[54]:


#importing library for Descision Tree and we have two criterion for DT 'gini' and 'Entropy'
# you can use gini when there is binary classification otherwise use entropy

from sklearn.tree import DecisionTreeClassifier

#max_depth -  the maximum height upto which the trees inside the forest can grow(to avoid overfitting)
#min_samples_split- minimum amount of samples an internal node must hold in order to split into further nodes(default value -2.)
#min_samples_leaf -  minimum amount of samples that a node must hold after getting split(default value -1.)

#making instance of DT
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0,
                                    max_depth = 2, min_samples_leaf = 10, min_samples_split = 20
                                   )
#fitting the training data
classifier.fit(X_train, y_train)


# In[55]:


#prediction using 'testing' data
y_pred = classifier.predict(X_test)


# In[56]:


y_pred


# In[57]:


#checking score via importing library
from sklearn.metrics import accuracy_score
DT_acc = accuracy_score(y_pred,y_test)
DT_acc


# In[58]:



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm


# In[59]:


#checking Auc_Roc_Score via importing library
from sklearn.metrics import roc_auc_score


# In[60]:


roc_auc_score(y_test,y_pred)


# # Hypertuning
#     
# 

# it is used to increases the performance of a model via providing best parameters

# In[61]:


#importing library for hypertuning
from sklearn.model_selection import GridSearchCV


# In[62]:


pGrid = {'max_depth': range(2, 10), # 8
        'min_samples_leaf': range(10, 51, 10),  # 5
        'min_samples_split': range(20, 81, 20)}   # 4
#intance of GScv
gscv = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = pGrid, cv = 5,
                       scoring = 'recall', n_jobs = -1, verbose = True)


# In[63]:


#fitiing the data
gscv.fit(X,y)


# In[64]:


#finding best params for model
gscv.best_params_


# **Disadvantages of decision trees:**
#     
# - Performance is (generally) not competitive with the best supervised learning methods
#     - Use Ensembles 
# - Can easily overfit the training data (tuning is required / PRUNING standard concept )
# 
# - Small variations in the data can result in a completely different tree (high variance)
#     - Use Ensembles to reduce the variance
#     
# - Recursive binary splitting makes "locally optimal" decisions that may not result in a globally optimal tree
# - Doesn't tend to work well if the classes are highly unbalanced
# - Doesn't tend to work well with very small datasets

# # RANDOM FOREST

# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean/average prediction of the individual trees. 

# In[65]:


from sklearn.ensemble import RandomForestClassifier

classifier2 = RandomForestClassifier(n_estimators = 70, criterion = 'gini', random_state = 0)
classifier2.fit(X_train, y_train)


# In[66]:


y_pred = classifier2.predict(X_test)


# In[67]:


from sklearn.metrics import accuracy_score
RF_acc = accuracy_score(y_test,y_pred)
RF_acc


# In[68]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)


# In[69]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[70]:


#n_estimators :- int, default=100
#The number of trees in the forest- the n_estimator parameter controls the number of trees inside the classifier.
#max_features helps to find the number of features to take into account in order to make the best split


pargrid_rf = {'n_estimators': [70, 80, 90, 100, 120],
                  'max_features': [5,10,15,20,25]}

gscv_rf = GridSearchCV(estimator = RandomForestClassifier(), 
                        param_grid = pargrid_rf, 
                        cv = 5,
                        verbose = True, 
                        n_jobs = -1)

gscv_rf.fit(X, y)


# In[71]:


gscv_rf.best_params_


# Importing libraries to draw tree structre

# In[72]:


from matplotlib import pyplot as plt
from sklearn import tree


# In[73]:


plt.figure(figsize = (15,10))
tree.plot_tree(classifier, filled = True)


# # KNN

# 1.Supervised Learning technique.
# 2.K-NN algorithm can be used for Regression & Classification -  mostly Classification problems.
# 3.K-NN is a non-parametric algoritham- which means it does not make any assumption on underlying data.
# 4.It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset.
# 5.At the time of classification, it performs an action on the dataset.
# 6.Example: Suppose, we have an image of a creature that looks similar to cat and dog, but we want to know either it is a cat or dog. So for this identification, we can use the KNN algorithm,

# Step-1: Select the number K of the neighbors
# Step-2: Calculate the Euclidean distance of K number of neighbors
# Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
# Step-4: Among these k neighbors, count the number of the data points in each category.
# Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
# Step-6: Our model is ready.

# In[74]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 11, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)




#p : integer, optional (default = 2)
#Power parameter for the Minkowski metric. 
#When p = 1, this is equivalent to using manhattan_distance (l1), 
# and euclidean_distance (l2) for p = 2.

#metric : string or callable, default ‘minkowski’

#The Minkowski distance is a metric in a normed vector space which can be considered as a 
# generalization of both the Euclidean distance and the Manhattan distance.

#the distance metric to use for the tree. The default metric is minkowski, 
#and with p=2 is equivalent to the standard Euclidean metric.


# In[75]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[76]:



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[77]:


from sklearn.metrics import accuracy_score
KNN_acc = accuracy_score(y_pred,y_test)
KNN_acc


# In[78]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_pred,y_test)


# In[79]:


import seaborn as sns
sns.heatmap(cm, annot = True)


# In[80]:


from sklearn.model_selection import GridSearchCV


# In[81]:


pGrid = {'n_neighbors': range(10,200),
        'leaf_size': range(10, 51, 10), }


gscv = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = pGrid, cv = 5,
                       scoring = 'recall', n_jobs = -1, verbose = True)


# In[ ]:


gscv.fit(X,y)


# # SVM

# 1supervised machine learning algorithm that can be used for both classification or regression challenges. 
# 2.However, it is mostly used in classification problems. ... 
# 3.The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes 
# 4.so that we can easily put the new data point in the correct category in the future

# In[ ]:


from sklearn.svm import SVC

#kernel parameters selects the type of hyperplane used to separate the data.
#Using ‘linear’ will use a linear hyperplane (a line in the case of 2D data). ‘rbf’ and ‘poly’ uses a non linear hyper-plane
#kernels = [‘linear’, ‘rbf’, ‘poly’]

#Linear Kernel - It is mostly used when there are a Large number of Features in a particular Data Set

classifier = SVC(kernel = 'linear', random_state = 0, C = 10, gamma = 0.01)
classifier.fit(X_train, y_train)

#C- is a hypermeter which is set before the training model and used to control error

#Gamma is also a hypermeter which is set before the training model and used to give curvature(the degree to which something is curved) weight of the decision boundary
#gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set.


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score
SVM_gscv_acc = accuracy_score(y_test,y_pred)
SVM_gscv_acc


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)


# In[ ]:




