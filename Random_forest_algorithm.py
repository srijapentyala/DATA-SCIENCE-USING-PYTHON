#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import load_digits


# In[3]:


digits=load_digits()


# In[4]:


digits


# In[6]:


df=pd.DataFrame(digits.data,columns=digits.feature_names)
df.head()


# In[7]:


#get no of rows and columns 
df.shape


# In[8]:


#get the target data 
digits.target


# In[9]:


#create a digits columns
df['digits']=digits.target


# In[10]:


#print the first two rows
df.head(2)


# In[11]:


df['digits'].value_counts()


# In[12]:


len(df['digits'].value_counts())


# In[13]:


df.isnull().sum()


# In[14]:


x=df.drop(['digits'],axis=1)
y=df['digits']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


rfc1=RandomForestClassifier(n_estimators=200,max_depth=2)


# In[20]:


rfc1.fit(x_train,y_train)


# In[21]:


y_pred=rfc1.predict(x_test)


# In[22]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[23]:


confusion_matrix(y_test,y_pred)


# In[26]:


accuracy_score(y_test,y_pred)


# In[27]:


rfc1.estimators_[0]


# In[28]:


from sklearn import tree
import matplotlib.pyplot as plt


# In[30]:


plt.figure(figsize=(25,5))
tree.plot_tree(rfc1.estimators_[0])
plt.show()


# In[ ]:




