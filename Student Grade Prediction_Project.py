#!/usr/bin/env python
# coding: utf-8

# # Student Grade Prediction

# # 1) Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.go_offline()


# In[2]:


# Reading the Student Grade Prediction_Project.csv file into a pandas dataframe.
students=pd.read_csv(r"C:\Users\sahee\OneDrive\Desktop\Study\Data trained project\PSD_3\Student_Grade_Prediction.csv")


# In[3]:


print('Total number of students:',len(students))


# In[4]:


students['G3'].describe()


# In[5]:


# Information on dataset
students.info()


# In[6]:


# Columns Dataset 
students.columns 


# In[7]:


# description of dataset 
students.describe()


# In[8]:


# 1st five values of dataset
students.head()   


# In[9]:


# last five values of dataset
students.tail()  


# In[10]:


# To check any null values present in dataset of file
students.isnull().any()    


# In[13]:


# Plot for the all attributes of data set
students.iplot()    


# In[14]:


# Plot for age vs G3
students.iplot(kind='scatter',x='age',y='G3',mode='markers',size=8)


# In[15]:


students.iplot(kind='box')


# In[16]:


students['G3'].iplot(kind='hist',bins=100,color='blue')


# # Data Visualiztion

# In[17]:


# To check any null values present in dataset pictorially
sns.heatmap(students.isnull(),cmap="rainbow",yticklabels=False)   


# There are no null values in the given dataset

# In[18]:


# Student's Sex


# In[19]:


# Number of female students
f_students = len(students[students['sex'] == 'F'])    
print('Number of female students:',f_students)
# Number of male students
m_students = len(students[students['sex'] == 'M'])    
print('Number of male students:',m_students)


# In[20]:


# male & female student representaion on countplot
sns.set_style('darkgrid')    
sns.countplot(x='sex',data=students,palette='plasma')


# # Age of Students

# In[21]:


# Kernel Density Estimations
b = sns.kdeplot(students['age'])    
b.axes.set_title('Ages of students')
b.set_xlabel('Age')
b.set_ylabel('Count')
plt.show()


# In[22]:


b = sns.countplot(x='age',hue='sex', data=students, palette='inferno')
b.axes.set_title('Number of Male & Female students in different age groups')
b.set_xlabel("Age")
b.set_ylabel("Count")
plt.show()


# In[23]:


# The student age seems to be ranging from 15-19, where gender distribution is pretty even in each age group.
# The age group above 19 may be outliers, year droupout students.


# #  Urban & Rural Areas Students

# In[24]:


# Number of urban areas students

u_students = len(students[students['address'] == 'U'])    
print('Number of Urban students:',u_students)

# Number of rural areas students

r_students = len(students[students['address'] == 'R'])    
print('Number of Rural students:',r_students)


# In[25]:


# urban & rural representaion on countplot
sns.set_style('darkgrid')
sns.countplot(x='address',data=students,palette='magma')  


# In[26]:


# Approximately 77.72% students come from urban region and 22.28% from rural region.


# In[27]:


sns.countplot(x='address',hue='G1',data=students,palette='Blues')


# # EDA - Exploratory Data Analysis

# In[28]:


b= sns.boxplot(x='age', y='G3',data=students,palette='gist_heat')
b.axes.set_title('Age vs Final Grade')


# In[29]:


b = sns.swarmplot(x='age', y='G3',hue='sex', data=students,palette='PiYG')
b.axes.set_title('Does age affect final grade?')


# In[30]:


# Grade distribution by address
sns.kdeplot(students.loc[students['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(students.loc[students['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('Do urban students score higher than rural students?')
plt.xlabel('Grade');
plt.ylabel('Density')
plt.show()


# In[31]:


#The above graph clearly shows there is not much difference between the grades based on location.


# In[32]:


students.corr()['G3'].sort_values()


# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
students.iloc[:,0]=le.fit_transform(students.iloc[:,0])
students.iloc[:,1]=le.fit_transform(students.iloc[:,1])
students.iloc[:,3]=le.fit_transform(students.iloc[:,3])
students.iloc[:,4]=le.fit_transform(students.iloc[:,4])
students.iloc[:,5]=le.fit_transform(students.iloc[:,5])
students.iloc[:,8]=le.fit_transform(students.iloc[:,8])
students.iloc[:,9]=le.fit_transform(students.iloc[:,9])
students.iloc[:,10]=le.fit_transform(students.iloc[:,10])
students.iloc[:,11]=le.fit_transform(students.iloc[:,11])
students.iloc[:,15]=le.fit_transform(students.iloc[:,15])
students.iloc[:,16]=le.fit_transform(students.iloc[:,16])
students.iloc[:,17]=le.fit_transform(students.iloc[:,17])
students.iloc[:,18]=le.fit_transform(students.iloc[:,18])
students.iloc[:,19]=le.fit_transform(students.iloc[:,19])
students.iloc[:,20]=le.fit_transform(students.iloc[:,20])
students.iloc[:,21]=le.fit_transform(students.iloc[:,21])
students.iloc[:,22]=le.fit_transform(students.iloc[:,22])


# In[34]:


students.head()


# In[35]:


students.tail()


# In[36]:


# Correlation with respect to G3
students.corr()['G3'].sort_values()    


# In[37]:


# drop the school and grade columns
students = students.drop(['school', 'G1', 'G2'], axis='columns')


# In[38]:


# Find correlations with the Grade
most_correlated = students.corr().abs()['G3'].sort_values(ascending=False)

# Maintain the top 8 most correlation features with Grade
most_correlated = most_correlated[:9]
most_correlated


# In[39]:


students = students.loc[:, most_correlated.index]
students.head()


# # Failure Attribute

# In[40]:


b = sns.swarmplot(x=students['failures'],y=students['G3'],palette='Blues')
b.axes.set_title('Previous Failures vs Final Grade(G3)')


# #  Fedu + Medu

# In[41]:


fa_edu = students['Fedu'] + students['Medu']
b = sns.swarmplot(x=fa_edu,y=students['G3'],palette='summer')
b.axes.set_title('Family Education vs Final Grade(G3)')


# # Planning to go for Higher Education Attribute

# In[42]:


b = sns.boxplot(x=students['higher'],y=students['G3'],palette='binary')
b.axes.set_title('Higher Education vs Final Grade(G3)')


# # Going Out with Friends Attribute

# In[43]:


b = sns.countplot(x=students['goout'],palette='OrRd')
b.axes.set_title('Go Out vs Final Grade(G3)')


# # Romantic relationship Attribute

# In[44]:


b = sns.swarmplot(x=students['romantic'],y=students['G3'],palette='YlOrBr')
b.axes.set_title('Romantic Relationship vs Final Grade(G3)')


# # Machine Learning Algorithms

# In[45]:


# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Splitting data into training/testing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Distributions
import scipy


# In[46]:


# splitting the data into training and testing data (75% and 25%)
X_train, X_test, y_train, y_test = train_test_split(students, students['G3'], test_size = 0.25, random_state=42)


# In[47]:


X_train.head()


# # MAE  and RMSE

# In[48]:


# Calculate mae and rmse
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    return mae, rmse


# In[49]:


# find the median
median_pred = X_train['G3'].median()

# create a list with all values as median
median_preds = [median_pred for _ in range(len(X_test))]

# store the true G3 values for passing into the function
true = X_test['G3']


# In[50]:


# Display the naive baseline metrics
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))


# In[67]:


# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')
    
    # Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=50)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results


# In[68]:


results = evaluate(X_train, X_test, y_train, y_test)
results


# In[75]:


plt.figure(figsize=(12, 7))
# Root mean squared error
ax =  plt.subplot(1, 2, 1)
results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'red', ax = ax)
plt.title('Model Mean Absolute Error') 
plt.ylabel('MAE')

# Median absolute percentage error
ax = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'Blue', ax = ax)
plt.title('Model Root Mean Squared Error') 
plt.ylabel('RMSE')

plt.show()


# In[ ]:


# Conclusion


# In[ ]:


#As we see both Model Mean Absolute Error & Model Root Mean Squared Error that the linear regression is performing the best in both cases

