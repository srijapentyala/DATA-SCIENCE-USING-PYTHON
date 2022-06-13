#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


from matplotlib import pyplot as plt


# In[10]:


img=cv2.imread('stop.jpg')


# In[11]:


print(img)


# In[12]:


img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[13]:


img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[14]:


data=cv2.CascadeClassifier('stop_data.xml')


# In[15]:


found=data.detectMultiScale(img_gray,minSize=(20,20))


# In[16]:


amount_found=len(found)
if amount_found!=0:
    for(x,y,width,height) in found:
        cv2.rectangle(img_rgb,(x,y),(x+height,y+width),(0,255,0),5)


# In[17]:


plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()


# In[ ]:




