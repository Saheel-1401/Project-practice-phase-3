#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
for dirname, _, filenames in os.walk(r"C:\Users\sahee\OneDrive\Desktop\Study\Data trained project\PSD_3\bank-full.csv\bank-full.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing the necessary libraries

# In[49]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# # Exploratory Data Analysis 

# In[3]:


df  = pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Study\Data trained project\PSD_3\bank-full.csv\bank-full.csv")
df.head()

Comments:
I have use read  read_csv() function of pandas for the Bank-Full dataset.
df is a dataframe. 
I have used head() funtion to display first 5 records of the dataset.
# In[4]:


rows_count, columns_count = df.shape
print('Total Number of rows :', rows_count)
print('Total Number of columns :', columns_count)

The Shape of the dataframe is (45211, 17). 
In Which 45211 rows and 17 columns in the dataset. 
Out of 17 columns 16 is our independent variables and 1 is our dependent variable which target.
# # Data type of each attribute

# In[5]:


df.dtypes

Dispalying Data types of each columns
As we can see the datatypes several of variables are string object. 
So, we will be changing it to appropriate datatype in data cleaning part.
# In[6]:


df.info()


# In[7]:


for feature in df.columns: 
    if df[feature].dtype == 'object': 
        df[feature] = pd.Categorical(df[feature])


# In[8]:


df.info()


# In[9]:


print(df.job.value_counts())
print('\n',df.marital.value_counts())
print('\n',df.education.value_counts())
print('\n',df.default.value_counts())
print('\n',df.housing.value_counts())
print('\n',df.loan.value_counts())
print('\n',df.contact.value_counts())
print('\n',df.month.value_counts())
print('\n',df.poutcome.value_counts())


# In[10]:


replaceStruct = {
                "job" :      {"unknown": -1, "blue-collar": 1, "management":2 , "technician": 3, "admin.": 4,"services": 5, 
                         "retired": 6, "self-employed": 7, "entrepreneur": 8, "unemployed": 9, "housemaid": 10,
                         "student": 11},
                "marital":   {"single": 1, "married": 2 ,"divorced": 3},
                "education": {"unknown":-1, "primary": 1, "secondary": 2 ,"tertiary": 3},
                "default":   {"no": 0, "yes": 1},
                "housing":   {"no": 0, "yes": 1},
                "loan":      {"no": 0, "yes": 1},
                "contact":   {"unknown": -1 , "cellular": 1, "telephone": 2},
                "month":     {"jan": 1, "feb":2 , "mar": 3, "apr": 4,"may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12},
                "poutcome":  {"unknown": -1, "failure": 0, "success": 1, "other": 2},
                "Target":    {"no": 0, "yes": 1} 
                    }

df=df.replace(replaceStruct)
df.head(10)


# In[11]:


df.tail(10)


# In[12]:


df.info()


# In[13]:


df.isnull().sum() 

Here total missing values count from each column is 0 and we can see there is no missing value in the dataframe.
But there are some 'unknown' values in the dataset with which we will deal after visualization.
# In[14]:


df.isnull().values.any()


# In[15]:


df.isna().any()


# In[16]:


for value in df.columns:
     print(value,":", sum(df[value] == '?'))


# # Descriptive Statistics

# In[17]:


df.describe()


# In[18]:


df_transpose = df.describe().T
df_transpose


# In[19]:


df_transpose[['min', '25%', '50%', '75%', 'max']]


# In[20]:


# Checking outliers
sns.boxplot(data=df, orient="h", palette="Set2", dodge=False)


# In[21]:


df.boxplot(return_type='axes', figsize=(20,5))


# In[22]:


Q1 =  df['balance'].quantile(0.25) 
Q3 =  df['balance'].quantile(0.75) 
IQR = Q3 - Q1                     
print('Interquartile range = ', IQR)
print('Numerber of outliers in balance column below the lower whisker :', df[df['balance'] < (Q1-(1.5*IQR))]['balance'].count())
print('Numerber of outliers above balance column the upper whisker  :', df[df['balance'] > (Q3+(1.5*IQR))]['balance'].count())


# In[23]:


for i in df.describe().columns:
    Q1 = df.describe().at['25%', i]
    Q3 = df.describe().at['75%', i]
    IQR = Q3-Q1
    LTV = Q1 - 1.5 * IQR 
    UTV = Q3 + 1.5 * IQR 
    print('Column Name                                         :', i)
    print('Interquartile range IQR                             :', IQR)
    print('Numerber of outliers below the lower whisker        :', df[df[i] < LTV][i].count())
    print('Numerber of outliers above the upper whisker         :', df[df[i] > UTV][i].count())
    print('Total Numbers of Outliers                           :', (df[df[i] < LTV][i].count()) + (df[df[i] > UTV][i].count()))
    
    print('\n')


# # Data Visualization 

# In[24]:


sns.pairplot(df.iloc[:,1:])


# In[25]:


subscriber_counts = pd.DataFrame(df["Target"].value_counts()).reset_index()
subscriber_counts.columns =["Labels","Target"]
subscriber_counts


# In[26]:


fig1, ax1 = plt.subplots()
explode = (0, 0.15)
ax1.pie(subscriber_counts["Target"], explode=explode, labels=subscriber_counts["Labels"], autopct='%1.1f%%',
        shadow=True, startangle=70)
ax1.axis('equal')
plt.title("Subscriber Percentage")
plt.show()


# In[27]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

sns.distplot(df['age'], ax = ax1)
sns.despine(ax = ax1)
ax1.set_xlabel('Age', fontsize=15)
ax1.set_ylabel('Occurence', fontsize=15)
ax1.set_title('Age x Ocucurence', fontsize=15)
ax1.tick_params(labelsize=15)

#Age group
bins = range(0, 100, 10)
ax2 = sns.distplot(df.age[df.Target==1], color='red', kde=False, bins=bins, label='Have Subscribed')
sns.distplot(df.age[df.Target==0], ax=ax2,  
         color='blue', kde=False, bins=bins, label="Haven't Subscribed")
plt.legend()

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 


# In[28]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
fig.set_size_inches(20, 8)

sns.countplot(x = 'job', data = df, ax = ax1, )
ax1.set_xlabel('Client Job', fontsize=15)
ax1.set_ylabel('Job', fontsize=15)
ax1.set_title('Job Distribution', fontsize=15)
ax1.tick_params(labelsize=15)
ax1.set_xticklabels(replaceStruct['job'], rotation=90)

sns.countplot(x = 'job', data = df, hue = 'Target', ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Job', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Job x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xticklabels(replaceStruct['job'], rotation=90)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 
plt.legend(title='Subscribers', labels=["Haven't Subscribed", 'Have Subscribed'])


# In[29]:


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 4))
fig.set_size_inches(20, 8)

sns.countplot(x = 'marital', data = df, ax = ax1)
ax1.set_xlabel('Client Marital Status', fontsize=15)
ax1.set_ylabel('Marital Status', fontsize=15)
ax1.set_title('Marital Distribution', fontsize=15)
ax1.set_xticklabels(replaceStruct['marital'])
ax1.tick_params(labelsize=15)

sns.countplot(x='marital', data = df, hue = 'Target', ax = ax2)
ax2.set_xlabel('Client Marital Status', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Marital Status with Target', fontsize=15)
ax2.set_xticklabels(replaceStruct['marital'])
ax2.tick_params(labelsize=15)

sns.boxplot(x='marital', y='age', hue="Target", data=df, ax=ax3)
ax3.set_xlabel('Client Marital Status', fontsize=15)
ax3.set_ylabel('Age', fontsize=15)
ax3.set_title('Marital Status with Age', fontsize=15)
ax3.set_xticklabels(replaceStruct['marital'])
ax3.tick_params(labelsize=15)

ax3.set_xticklabels(replaceStruct['marital'])
plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 


# In[30]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
fig.set_size_inches(20, 8)

sns.countplot(x = 'education', data = df, ax = ax1)
ax1.set_xlabel('Client Education', fontsize=15)
ax1.set_ylabel('Education', fontsize=15)
ax1.set_title('Education Distribution', fontsize=15)
ax1.set_xticklabels(replaceStruct['education'])
ax1.tick_params(labelsize=15)


sns.countplot(x = 'education', data = df, hue = 'Target', ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Education', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Education with Target', fontsize=15)
ax2.set_xticklabels(replaceStruct['education'])
ax2.tick_params(labelsize=15)


plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 
plt.legend(title='Subscribers', labels=["Haven't Subscribed", 'Have Subscribed'])


# In[ ]:


fig, (ax1, ax2, x3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13, 5))
sns.boxplot(x ='balance', data = df, orient = 'v', ax = ax1)
ax1.set_xlabel('Client balance', fontsize=15)
ax1.set_ylabel('balance', fontsize=15)
ax1.set_title('balance Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(x ='balance',data = df,ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('balance', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('balance x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

bins = range(0, 100, 10)
ax3 = sns.distplot(df.balance[df.Target==1], color='red', kde=False, bins=bins, label='Have Subscribed')
sns.distplot(df.balance[df.Target==0], ax=ax3, color='blue', kde=False, bins=bins, label="Haven't Subscribed")
plt.legend()

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 

plt.figure(figsize=(10,4))
sns.distplot(df[df["Target"] == 0]['balance'], color = 'r',label="Target= Haven't Subscribed")
sns.distplot(df[df["Target"] == 1]['balance'], color = 'b',label='Target= Have Subscribed')
plt.legend()
plt.title("balance Distribution")


# In[ ]:


# Client loan
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,5))
sns.countplot(x = 'default', data = df, hue='Target', ax = ax1)
ax1.set_title('Credit Crad in Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)
ax1.set_xticklabels(replaceStruct['default'])
ax1.legend(["Haven't Subscribed", "Have Subscribed"])
# Housing, has housing loan ?
sns.countplot(x = 'housing', data = df,  hue='Target', ax = ax2)
ax2.set_title('Housing Loan', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xticklabels(replaceStruct['housing'])
ax2.legend(["Haven't Subscribed", "Have Subscribed"])

# Loan, has personal loan ?
sns.countplot(x = 'loan', data = df,  hue='Target', ax = ax3)
ax3.set_title('Personal Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)
ax3.set_xticklabels(replaceStruct['loan'])
ax3.legend(["Haven't Subscribed", "Have Subscribed"])

plt.subplots_adjust(wspace=0.25)


# In[ ]:


# Contact Distribution

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))

sns.countplot(x = 'contact', data = df,  ax = ax1)
ax1.set_title('Contact', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.set_xticklabels(replaceStruct['contact'])
ax1.tick_params(labelsize=15)

sns.countplot(x = 'contact', data = df, hue='Target', ax = ax2)
ax2.set_title('Contact wiht target column', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xticklabels(replaceStruct['contact'])
ax2.legend(["Haven't Subscribed", "Have Subscribed"])
plt.subplots_adjust(wspace=0.25)


# In[ ]:


# Months Distribution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,4))

sns.countplot(x='month',data=df, ax = ax1)
ax1.set_xticklabels(replaceStruct['month'], rotation=90)
ax1.set_title('Month Counts')


sns.countplot(x = 'month', data=df, hue='Target',ax = ax2)
ax2.set_xticklabels(replaceStruct['month'], rotation=90)
ax2.set_title('Month Counts with subscription rate')

plt.subplots_adjust(wspace=0.25)
plt.legend(title='Subscribers', labels=["Haven't Subscribed", 'Have Subscribed'])


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,4))

sns.countplot(x = 'day', data=df,ax = ax1)
ax1.set_title('Last Contact Day of month')


sns.countplot(x = 'day', data=df, hue='Target',ax = ax2)
ax2.set_title('Last Contact Day of month with subscription rate')

plt.subplots_adjust(wspace=0.25)
plt.legend(title='Subscribers', labels=["Haven't Subscribed", 'Have Subscribed'])


# In[ ]:


# Duration Distribution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.distplot(df['duration'], ax = ax1)
sns.despine(ax = ax1)
ax1.set_title('duration x Ocucurence', fontsize=15)
ax1.tick_params(labelsize=15)

sns.barplot(x="Target", y="duration", data=df, ax = ax2)
sns.despine(ax = ax2)
ax2.set_xticklabels(replaceStruct['Target'])


# In[ ]:


# Campaign Distrbution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.distplot(df['campaign'], ax = ax1)
sns.despine(ax = ax1)
ax1.set_title('campaign x Ocucurence', fontsize=15)
ax1.tick_params(labelsize=15)

sns.barplot(x="Target", y="campaign", data=df, ax = ax2)
sns.despine(ax = ax2)
ax2.set_xticklabels(replaceStruct['Target'])


# In[ ]:


fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (13, 5))
sns.countplot(x = 'poutcome', data=df, hue='Target')
ax.set_xticklabels(replaceStruct['poutcome'])
plt.legend(title='Subscribers', labels=["Haven't Subscribed", 'Have Subscribed'])


# In[ ]:


# Correlation using Heatmap
plt.figure(figsize = (15,7))
plt.title('Correlation of Attributes', y=1.05, size=19)
sns.heatmap(df.corr(), cmap='plasma',annot=True, fmt='.2f')


# In[ ]:


df['poutcome'].head(4)


# In[ ]:


oneHotCols=["job", "marital", "poutcome"]
df=pd.get_dummies(df, columns=oneHotCols)


# In[ ]:


df.head(1)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[33]:


X = df.drop('Target', axis=1)
y = df[['Target']]
df.head(1)


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
print('x train data {}'.format(x_train.shape))
print('y train data {}'.format(y_train.shape))
print('x test data  {}'.format(X_test.shape))
print('y test data  {}'.format(y_test.shape))


# # Comparing Base Models

# In[42]:


from sklearn.model_selection import KFold
seed = 7
kfold = model_selection.KFold(n_splits=10, shuffle= True, random_state=seed)


# In[43]:


# LRR
LogReg = LogisticRegression(solver = 'lbfgs')
LogReg.fit(X_train, y_train.values.ravel())

# Predicting for test set
LogReg_y_pred               = LogReg.predict(X_test)
LogReg_Score                = LogReg.score(X_test, y_test)

LogReg_ScoreAccuracy        = accuracy_score(y_test, LogReg_y_pred)

LogReg_PrecisonScore        = precision_score(y_test, LogReg_y_pred)
LogReg_RecollScore          = recall_score(y_test, LogReg_y_pred)
LogReg_F1                   = f1_score(y_test, LogReg_y_pred)

cross_validation_result = model_selection.cross_val_score(LogReg, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

base_model_results = pd.DataFrame([['Logistic Regression', LogReg_ScoreAccuracy, LogReg_PrecisonScore,
                                LogReg_RecollScore, LogReg_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])

print('\nLogistic Regression classification Report : \n',metrics.classification_report(y_test, LogReg_y_pred))


# # KNN

# In[44]:


Knn = KNeighborsClassifier(n_neighbors=9, weights = 'uniform', metric='euclidean')
Knn.fit(X_train, y_train.values.ravel())

# Predicting for test set
Knn_y_pred               = Knn.predict(X_test)
Knn_Score                = Knn.score(X_test, y_test)

Knn_ScoreAccuracy        = accuracy_score(y_test, Knn_y_pred)
Knn_PrecisonScore        = precision_score(y_test, Knn_y_pred)
Knn_RecollScore          = recall_score(y_test, Knn_y_pred)
Knn_F1                   = f1_score(y_test, Knn_y_pred)

cross_validation_result = model_selection.cross_val_score(Knn, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')
knn_models_results = pd.DataFrame([['K-Nearest Neighbors', Knn_ScoreAccuracy, Knn_PrecisonScore,
                                Knn_RecollScore, Knn_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
base_model_results = base_model_results.append(knn_models_results, ignore_index = True)

print('\nK-Nearest Neighbors (K-NN) classification Report : \n',metrics.classification_report(y_test, Knn_y_pred))


# In[45]:


base_model_results


# # Gaussian

# In[ ]:


GNB = GaussianNB()
GNB.fit(X_train, y_train.values.ravel())

# Predicting for test set
GNB_y_pred               = GNB.predict(X_test)
GNB_Score                = GNB.score(X_test, y_test)

GNB_ScoreAccuracy        = accuracy_score(y_test, GNB_y_pred)
GNB_PrecisonScore        = precision_score(y_test, GNB_y_pred)
GNB_RecollScore          = recall_score(y_test, GNB_y_pred)
GNB_F1                   = f1_score(y_test, GNB_y_pred)

cross_validation_result = model_selection.cross_val_score(GNB, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

GNB_models_results = pd.DataFrame([['Naive Bayes (Gaussian)', GNB_ScoreAccuracy, GNB_PrecisonScore,
                                GNB_RecollScore, GNB_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
base_model_results = base_model_results.append(GNB_models_results, ignore_index = True)
print('\nGNB classification Report : \n',metrics.classification_report(y_test, GNB_y_pred))


# In[46]:


base_model_results


# # Decision Tree

# In[65]:


dTree = DecisionTreeClassifier(criterion = 'entropy', random_state=1)
dTree.fit(X_train, y_train)

# Predicting for test set
dTree_y_pred               = dTree.predict(X_test)
dTree_Score                = dTree.score(X_test, y_test)

dTree_ScoreAccuracy        = accuracy_score(y_test, dTree_y_pred)
dTree_PrecisonScore        = precision_score(y_test, dTree_y_pred)
dTree_RecollScore          = recall_score(y_test, dTree_y_pred)
dTree_F1                   = f1_score(y_test, dTree_y_pred)

cross_validation_result = model_selection.cross_val_score(dTree, X_train, y_train, cv=kfold, scoring='accuracy')
dTree_models_results = pd.DataFrame([['Decision Tree ', dTree_ScoreAccuracy, dTree_PrecisonScore,
                                dTree_RecollScore, dTree_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
base_model_results = base_model_results.append(dTree_models_results, ignore_index = True)
print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))

print('\nDTree classification Report : \n',metrics.classification_report(y_test, dTree_y_pred))


# In[67]:


dTreePR = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
dTreePR.fit(X_train, y_train)
print(dTreePR.score(X_train, y_train))
print(dTreePR.score(X_test, y_test))


# In[ ]:


# Predicting for test set
dTreePR_y_pred               = dTreePR.predict(X_test)
dTreePR_Score                = dTreePR.score(X_test, y_test)

dTreePR_ScoreAccuracy        = accuracy_score(y_test, dTreePR_y_pred)
dTreePR_PrecisonScore        = precision_score(y_test, dTreePR_y_pred)
dTreePR_RecollScore          = recall_score(y_test, dTreePR_y_pred)
dTreePR_F1                   = f1_score(y_test, dTreePR_y_pred)

cross_validation_result = model_selection.cross_val_score(dTreePR, X_train, y_train, cv=kfold, scoring='accuracy')
dTreePR_models_results = pd.DataFrame([['Decision Tree (Prune)', dTreePR_ScoreAccuracy, dTreePR_PrecisonScore,
                                dTreePR_RecollScore, dTreePR_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
base_model_results = base_model_results.append(dTreePR_models_results, ignore_index = True)

print('\nDTree with Prune classification Report : \n',metrics.classification_report(y_test, dTreePR_y_pred))


# In[ ]:


Svm = SVC(random_state = 0, kernel = 'rbf', probability= True)
Svm.fit(X_train, y_train.values.ravel())

# Predicting for test set
Svm_y_pred               = Svm.predict(X_test)
Svm_Score                = Svm.score(X_test, y_test)

Svm_ScoreAccuracy        = accuracy_score(y_test, Svm_y_pred)
Svm_PrecisonScore        = precision_score(y_test, Svm_y_pred)
Svm_RecollScore          = recall_score(y_test, Svm_y_pred)
Svm_F1                   = f1_score(y_test, Svm_y_pred)

cross_validation_result = model_selection.cross_val_score(Svm, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')
svm_models_results = pd.DataFrame([['SVM (RBF)', Svm_ScoreAccuracy, Svm_PrecisonScore,
                                Svm_RecollScore, Svm_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
base_model_results = base_model_results.append(svm_models_results, ignore_index = True)
print('\nSVM classification Report : \n',metrics.classification_report(y_test, Svm_y_pred))


# In[ ]:


#CM
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (20,10))

cm=metrics.confusion_matrix(y_test, LogReg_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,0])
axs[0,0].set_xlabel('Predicted Labels');
axs[0,0].set_ylabel('Actual Labels'); 
axs[0,0].set_title('Logistic Regression'); 

cm=metrics.confusion_matrix(y_test, Knn_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,1])
axs[0,1].set_xlabel('Predicted Labels');
axs[0,1].set_ylabel('Actual Labels'); 
axs[0,1].set_title('K-NN'); 

cm=metrics.confusion_matrix(y_test, Svm_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,2])
axs[0,2].set_xlabel('Predicted Labels');
axs[0,2].set_ylabel('Actual Labels'); 
axs[0,2].set_title('SVM(RBF)');

cm=metrics.confusion_matrix(y_test, GNB_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,0])
axs[1,0].set_xlabel('Predicted Labels');
axs[1,0].set_ylabel('Actual Labels'); 
axs[1,0].set_title('Naive Bayes (Gaussian)');


cm=metrics.confusion_matrix(y_test, dTree_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,1])
axs[1,1].set_xlabel('Predicted Labels');
axs[1,1].set_ylabel('Actual Labels'); 
axs[1,1].set_title('Decision Tree');

cm=metrics.confusion_matrix(y_test, dTreePR_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,2])
axs[1,2].set_xlabel('Predicted Labels');
axs[1,2].set_ylabel('Actual Labels'); 
axs[1,2].set_title('Decision Tree(Prune)');


# In[ ]:


ensemble_results =[]  # empty array to store the model result


# In[58]:


bagging_model = BaggingClassifier(base_estimator=dTreePR, n_estimators=60,random_state=1)
bagging_model.fit(X_train, y_train.values.ravel())

# Predicting for test set
bagging_y_pred               = bagging_model.predict(X_test)
bagging_Score                = bagging_model.score(X_test, y_test)

bagging_ScoreAccuracy        = accuracy_score(y_test, bagging_y_pred)
bagging_PrecisonScore        = precision_score(y_test, bagging_y_pred)
bagging_RecollScore          = recall_score(y_test, bagging_y_pred)
bagging_F1                   = f1_score(y_test, bagging_y_pred)

bagging_cross_validation_result = model_selection.cross_val_score(bagging_model, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

bagging_ensemble_results = pd.DataFrame([['Bagging with DTree', bagging_ScoreAccuracy, bagging_PrecisonScore,
                                bagging_RecollScore, bagging_F1, bagging_cross_validation_result.mean(), bagging_cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])

ensemble_results = bagging_ensemble_results
ensemble_results

print('\nclassification Report : \n',metrics.classification_report(y_test, bagging_y_pred))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
adaBoosting = AdaBoostClassifier(n_estimators=50, random_state=1)
adaBoosting.fit(X_train, y_train.values.ravel())

# Predicting for test set
adaBoosting_y_pred               = adaBoosting.predict(X_test)
adaBoosting_Score                = adaBoosting.score(X_test, y_test)
adaBoosting_ScoreAccuracy        = accuracy_score(y_test, adaBoosting_y_pred)

adaboosting_ScoreAccuracy        = accuracy_score(y_test, adaBoosting_y_pred)
adaboosting_PrecisonScore        = precision_score(y_test, adaBoosting_y_pred)
adaboosting_RecollScore          = recall_score(y_test, adaBoosting_y_pred)
adaboosting_F1                   = f1_score(y_test, adaBoosting_y_pred)

ada_cross_validation_result = model_selection.cross_val_score(adaBoosting, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

adaboosting_results = pd.DataFrame([['Ada Boosting with DTree', adaboosting_ScoreAccuracy, adaboosting_PrecisonScore,
                                adaboosting_RecollScore, adaboosting_F1, ada_cross_validation_result.mean(), ada_cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
ensemble_results=ensemble_results.append(adaboosting_results, ignore_index = True)
ensemble_results

print('\nclassification Report : \n',metrics.classification_report(y_test, adaBoosting_y_pred))


# In[ ]:


# ROC graph

logistic_fpr, logistic_tpr, logistic_threshold = metrics.roc_curve(y_test, LogReg_y_pred)
logistic_roc_auc = metrics.roc_auc_score(y_test, LogReg_y_pred)
fig1_graph = plt.figure(figsize=(15,10))
fig1_graph.add_subplot(2,3,1)
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Model (area = %0.2f)' % logistic_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")


knn_fpr, knn_tpr, knn_threshold = metrics.roc_curve(y_test, Knn_y_pred)
knn_roc_auc = metrics.roc_auc_score(y_test, Knn_y_pred)
fig1_graph.add_subplot(2,3,2)
plt.plot(knn_fpr, knn_tpr, label='K-NN Model (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

svm_fpr, svm_tpr, svm_threshold = metrics.roc_curve(y_test, Svm_y_pred)
svm_roc_auc = metrics.roc_auc_score(y_test,  Svm_y_pred)
fig1_graph.add_subplot(2,3,3)
plt.plot(svm_fpr, svm_tpr, label='SVM Model (area = %0.2f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

gnb_fpr, gnb_tpr, gnb_threshold = metrics.roc_curve(y_test,  GNB_y_pred)
gnb_roc_auc = metrics.roc_auc_score(y_test,  GNB_y_pred)
fig1_graph.add_subplot(2,3,4)
plt.plot(gnb_fpr, gnb_tpr, label='GNB Model (area = %0.2f)' % gnb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

dTree_fpr, dTree_tpr, dTree_threshold = metrics.roc_curve(y_test, dTreePR_y_pred)
dTree_roc_auc = metrics.roc_auc_score(y_test,  dTreePR_y_pred)
fig1_graph.add_subplot(2,3,5)
plt.plot(dTree_fpr, dTree_tpr, label='Decision Tree Model (area = %0.2f)' % dTree_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()

base_model_results


# In[ ]:


Dtree_Base_Ensemble = bagging_ensemble_results

Dtree_Base = dTreePR_models_results
Dtree_Base_Ensemble = Dtree_Base_Ensemble.append(Dtree_Base, ignore_index = True)
Dtree_Base_Ensemble


# In[106]:


from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier(base_estimator=LogReg, n_estimators=60,random_state=1)
bgcl = bgcl.fit(X_train_scaled, y_train.values.ravel())
y_pred = bgcl.predict(X_test)

bagging_ScoreAccuracy        = accuracy_score(y_test, y_pred)
bagging_PrecisonScore        = precision_score(y_test, y_pred)
bagging_RecollScore          = recall_score(y_test, y_pred)
bagging_F1                   = f1_score(y_test, y_pred)

LogReg_cross_validation_result = model_selection.cross_val_score(bgcl, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

log_bagging_ensemble_results = pd.DataFrame([['Bagging with Logistic Regession', bagging_ScoreAccuracy, bagging_PrecisonScore,
                                bagging_RecollScore, bagging_F1, LogReg_cross_validation_result.mean(), LogReg_cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
ensemble_results = ensemble_results.append(log_bagging_ensemble_results, ignore_index = True)

print('\nclassification Report : \n',metrics.classification_report(y_test, y_pred))


# In[ ]:


Log_Base_Ensemble = log_bagging_ensemble_results
Log_base =  pd.DataFrame([['Logistic Regression', LogReg_ScoreAccuracy, LogReg_PrecisonScore,
                                LogReg_RecollScore, LogReg_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
Log_Base_Ensemble = Log_Base_Ensemble.append(Log_base, ignore_index = True)
Log_Base_Ensemble


# # RandomForest Classifier 

# In[85]:


from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators = 50, random_state=1, max_features=12)
randomForest.fit(X_train, y_train.values.ravel())

# Predicting for test set
randomForest_y_pred               = randomForest.predict(X_test)
randomForest_Score                = randomForest.score(X_test, y_test)
randomForest_ScoreAccuracy        = accuracy_score(y_test, randomForest_y_pred)

randomForest_ScoreAccuracy        = accuracy_score(y_test, randomForest_y_pred)
randomForest_PrecisonScore        = precision_score(y_test, randomForest_y_pred)
randomForest_RecollScore          = recall_score(y_test, randomForest_y_pred)
randomForest_F1                   = f1_score(y_test, randomForest_y_pred)

rdf_cross_validation_result = model_selection.cross_val_score(randomForest, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

randomForest_results = pd.DataFrame([['Random Forest', randomForest_ScoreAccuracy, randomForest_PrecisonScore,
                                randomForest_RecollScore, randomForest_F1, rdf_cross_validation_result.mean(), rdf_cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
ensemble_results = ensemble_results.append(randomForest_results, ignore_index = True)

print('\nclassification Report : \n',metrics.classification_report(y_test, randomForest_y_pred))


# # Comparison with base model :-

# In[ ]:


rdf_Base_Ensemble = randomForest_results
Dtree_Base = pd.DataFrame([['Base Model with DTree', dTreePR_ScoreAccuracy, dTreePR_PrecisonScore,
                                dTreePR_RecollScore, dTreePR_F1, cross_validation_result.mean(), cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
rdf_Base_Ensemble = rdf_Base_Ensemble.append(Dtree_Base, ignore_index = True)
rdf_Base_Ensemble


# # Voting Classifier

# In[103]:


from sklearn.ensemble import VotingClassifier

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
# create the ensemble model
voting_model = VotingClassifier(estimators)
voting_model.fit(X_train, y_train.values.ravel())

# Predicting for test set
voting_model_y_pred               = voting_model.predict(X_test)

voting_model_ScoreAccuracy        = accuracy_score(y_test, voting_model_y_pred)
voting_model_PrecisonScore        = precision_score(y_test, voting_model_y_pred)
voting_model_RecollScore          = recall_score(y_test, voting_model_y_pred)
voting_model_F1                   = f1_score(y_test, voting_model_y_pred)

voting_cross_validation_result = model_selection.cross_val_score(voting_model, X_train, y_train.values.ravel(), cv=kfold, scoring='accuracy')

voting_model_results = pd.DataFrame([['Voting', voting_model_ScoreAccuracy,
                                        voting_model_PrecisonScore, voting_model_RecollScore, voting_model_F1,
                                      voting_cross_validation_result.mean(), voting_cross_validation_result.std()]], 
                              columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean', 'Std Deviation'])
ensemble_results = ensemble_results.append(voting_model_results, ignore_index = True)
ensemble_results

print('\nclassification Report : \n',metrics.classification_report(y_test, voting_model_y_pred))


# In[75]:


# Compare performances of all the models
base_model_results


# In[76]:


ensemble_results


# In[77]:


Final_Score_DF = [base_model_results, ensemble_results]


# In[78]:


Final_Model_Accuracy_DF =pd.concat(Final_Score_DF)
Final_Model_Accuracy_DF.reset_index(drop=True)


# # Confusion Matrix of Above Ensembed Models

# In[ ]:


fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (20,10))

cm=metrics.confusion_matrix(y_test, bagging_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,0])
axs[0,0].set_xlabel('Predicted Labels');
axs[0,0].set_ylabel('Actual Labels'); 
axs[0,0].set_title('Bagging With DTree'); 

cm=metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,1])
axs[0,1].set_xlabel('Predicted Labels');
axs[0,1].set_ylabel('Actual Labels'); 
axs[0,1].set_title('Bagging with Logistic Regression'); 

cm=metrics.confusion_matrix(y_test, randomForest_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[0,2])
axs[0,2].set_xlabel('Predicted Labels');
axs[0,2].set_ylabel('Actual Labels'); 
axs[0,2].set_title('Random Forest');

cm=metrics.confusion_matrix(y_test, adaBoosting_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,0])
axs[1,0].set_xlabel('Predicted Labels');
axs[1,0].set_ylabel('Actual Labels'); 
axs[1,0].set_title('Ada Boosting With Decision tree');


cm=metrics.confusion_matrix(y_test, adaBoostingLog_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,1])
axs[1,1].set_xlabel('Predicted Labels');
axs[1,1].set_ylabel('Actual Labels'); 
axs[1,1].set_title('Ada Boosting With Logistic Regression');

cm=metrics.confusion_matrix(y_test, gradientBoost_y_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm, annot=True ,fmt='g', ax = axs[1,2])
axs[1,2].set_xlabel('Predicted Labels');
axs[1,2].set_ylabel('Actual Labels'); 
axs[1,2].set_title('Gradient Boosting');


# In[ ]:


# ROC
bagging_fpr, bagging_tpr, bagging_threshold = metrics.roc_curve(y_test, bagging_y_pred)
bagging_roc_auc = metrics.roc_auc_score(y_test, bagging_y_pred)
fig1_graph = plt.figure(figsize=(15,10))
fig1_graph.add_subplot(2,3,1)
plt.plot(bagging_fpr, bagging_tpr, label='bagging with DTree (area = %0.2f)' % bagging_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")


bagging_log_fpr, bagging_log_tpr, bagging_log_threshold = metrics.roc_curve(y_test,y_pred)
bagging_log_roc_auc = metrics.roc_auc_score(y_test, y_pred)
fig1_graph.add_subplot(2,3,2)
plt.plot(bagging_log_fpr, bagging_log_tpr, label='bagging with LR (area = %0.2f)' % bagging_log_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

rdf_fpr, rdf_tpr, rdf_threshold = metrics.roc_curve(y_test,  randomForest_y_pred)
rdf_roc_auc = metrics.roc_auc_score(y_test,  randomForest_y_pred)
fig1_graph.add_subplot(2,3,3)
plt.plot(rdf_fpr, rdf_tpr, label='Random Forest (area = %0.2f)' % rdf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

ada_fpr, ada_tpr, ada_threshold = metrics.roc_curve(y_test,  adaBoosting_y_pred)
ada_roc_auc = metrics.roc_auc_score(y_test,  adaBoosting_y_pred)
fig1_graph.add_subplot(2,3,4)
plt.plot(ada_fpr, ada_tpr, label='Ada Boosting with DT (area = %0.2f)' % ada_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

ada_log_fpr, ada_log_tpr, ada_log_threshold = metrics.roc_curve(y_test,  adaBoostingLog_y_pred)
ada_log_roc_auc = metrics.roc_auc_score(y_test,  adaBoostingLog_y_pred)
fig1_graph.add_subplot(2,3,5)
plt.plot(ada_log_fpr, ada_log_tpr, label='Ada Boosting With LR (area = %0.2f)' % ada_log_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

grd_fpr, grd_tpr, grd_threshold = metrics.roc_curve(y_test,gradientBoost_y_pred)
grd_roc_auc = metrics.roc_auc_score(y_test,gradientBoost_y_pred)
fig1_graph.add_subplot(2,3,6)
plt.plot(grd_fpr, grd_tpr, label='Gradient Boosting (area = %0.2f)' % grd_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()


# In[ ]:


Final_Result1[Final_Result1['Accuracy'] == Final_Result1['Accuracy'].max()]['Model']

1) Gradiant Boosting and Bagging with Decission Tree model with scaled data gives us best accuracy of  91% approx.
2) Also the Type I(False Posssitive) and Type II(False Negative) errors are least in Gradiant Boosting and Bagging with Decission Tree model.
3) The area in ROC curve for Bagging with Decission Tree and Gradiant Boosting is around 0.66 and the ROC curve is highest for random forest.
4) Hence among the above three algorithm applied on the underline dataset, Bagging with Decission Tree or Gradiant Boosting would be the best choice to predict the clients who will subscribe for the term deposite.