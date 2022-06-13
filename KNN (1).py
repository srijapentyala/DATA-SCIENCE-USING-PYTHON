#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


dataset=load_iris()
dataset


# In[4]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df.head()


# In[5]:


df.shape


# In[6]:


dataset.target


# In[7]:


df['iris']=dataset.target


# In[8]:


df['iris'].value_counts()


# In[9]:


x=df.drop(['iris'],axis=1)
y=df['iris']


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[14]:



knn.fit(x_train, y_train)


# In[15]:


y_pred=knn.predict(x_test)


# In[16]:


y_pred


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


accuracy_score(y_test,y_pred)


# In[19]:


x_pred=np.array([[5,3,1,0],[4.6,3.1,2,0.1],[7.2,3.1,5.1,1],[8,4,7,2]])
x_predict=knn.predict(x_pred)
x_predict


# In[21]:


x_pred1=dataset['target_names'][x_predict]


# In[22]:


x_pred1


# In[23]:


y_test1=np.array(["setosa","setosa","virginica","virginica"])


# In[24]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test1,x_pred1)


# In[ ]:




