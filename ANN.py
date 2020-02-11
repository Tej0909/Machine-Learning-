#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('C:/Datascience/Datasets/P16-Artificial-Neural-Networks/Artificial_Neural_Networks/Churn_Modelling.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.describe()


# In[5]:


dataset.isnull().sum()


# In[6]:


import seaborn as sb


# In[7]:


sb.boxplot('Gender','Age',data=dataset)


# In[8]:


sb.countplot(dataset['Exited'],hue=dataset['Gender'])


# In[9]:


sb.countplot(dataset['Exited'],hue=dataset['Geography'])


# In[10]:


x = dataset.iloc[:,3:13].values


# In[11]:


y = dataset.iloc[:,13].values


# In[12]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[13]:


label_encoder_x = LabelEncoder()
x[:,1] = label_encoder_x.fit_transform(x[:,1])


# In[14]:


label_encoder_x1 = LabelEncoder()
x[:,2] = label_encoder_x1.fit_transform(x[:,2])


# In[15]:


x


# In[16]:


onehotencoder = OneHotEncoder(categorical_features = [1] )
x = onehotencoder.fit_transform(x).toarray()


# In[17]:


x


# In[18]:


x = x[:,1:]


# In[19]:


x


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[21]:


from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_train = x_scale.fit_transform(x_train)
x_test = x_scale.transform(x_test)


# In[22]:


import keras


# In[23]:


# To intialize neural network 

from keras.models import Sequential 


# In[24]:


# To build the Layers of neural networks

from keras.layers import Dense


# In[25]:


# Initializing ANN
classifier = Sequential()


# In[26]:


# Adding input layer and first hiddenlayer
classifier.add(Dense(output_dim = 6,init='uniform',activation = 'relu',input_dim = 11))


# In[27]:


classifier.add(Dense(output_dim = 6,init='uniform',activation='relu'))


# In[28]:


classifier.add(Dense(output_dim = 1,init = 'uniform',activation ='sigmoid'))


# In[29]:


classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])


# In[30]:


classifier.fit(x_train,y_train,batch_size = 10,epochs = 100)


# In[ ]:





# In[ ]:




