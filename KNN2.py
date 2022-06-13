#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data=pd.read_csv('knn2data.csv')
data


# In[4]:


x=data.iloc[:,1:3].values
y=data.iloc[:,0].values


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier


# In[8]:


knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train,y_train)
y_predict=knn1.predict(x_test)
y_predict


# In[11]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_predict)
cm


# In[15]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[ ]:




