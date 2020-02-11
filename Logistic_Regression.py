#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression (For Predicting yes/no , true/false , 0/1)

# ## Data Collection

# In[153]:


import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[154]:


df = pd.read_csv('C:/Datascience/Datasets/titanic_train.csv')
df.head()


# ## Data Analysis

# In[155]:


sb.countplot(x = 'Survived',data=df)


# In[156]:


sb.countplot(x = 'Survived',hue='Sex',data=df)


# In[157]:


df['Age'].plot.hist()


# In[158]:


sb.boxplot(x = 'Pclass', y = 'Age',data = df)


# ## Data Manipulation 

# In[159]:


df.describe().transpose()


# In[160]:


df.isnull().sum()


# In[161]:


sb.heatmap(df.isnull())


# In[162]:


# Dropping the unwanted columns 

df.drop(['Cabin','Ticket','Fare'],axis =1,inplace = True)


# In[163]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
df[['Age']] = imputer.fit_transform(df[['Age']])


# In[164]:


sb.heatmap(df.isnull())


# In[165]:


df.head()


# In[166]:


df.drop('Name',axis = 1, inplace=True)


# In[167]:


df.head()


# In[168]:


sex = pd.get_dummies(df['Sex'],drop_first = True)


# In[169]:


embarked = pd.get_dummies(df['Embarked'],drop_first = True)


# In[170]:


pclass = pd.get_dummies(df['Pclass'],drop_first = True)


# In[171]:


df = pd.concat([df,sex,embarked,pclass],axis=1)


# In[172]:


df.head()


# In[173]:


df.drop(['Embarked','Pclass','Sex'],axis = 1, inplace=True)


# In[174]:


df.head()


# In[175]:


X = df.drop('Survived',axis =1)
y = df['Survived']


# In[176]:


from sklearn.model_selection import train_test_split
X_train , X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20)


# In[179]:


from sklearn.linear_model import LogisticRegression


# In[180]:


lgr = LogisticRegression()
lgr.fit(X_train,y_train)


# In[181]:


predictions = lgr.predict(X_test)


# In[183]:


from sklearn.metrics import classification_report


# In[184]:


classification_report(y_test,predictions)


# In[185]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[187]:


predictions[0:5]


# In[189]:


y_test[0:5]


# In[ ]:




