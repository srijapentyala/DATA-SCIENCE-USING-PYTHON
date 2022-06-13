#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
from matplotlib import pyplot as plt


# In[3]:


img=cv2.imread('people.jpg')
print(img)


# In[4]:


img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[5]:


data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[6]:


found=data.detectMultiScale(img_gray,minSize=(20,20))
amount_found=len(found)
if amount_found!=0:
    for(x,y,width,height) in found:
        cv2.rectangle(img_rgb,(x,y),(x+height,y+width),(0,255,0),5)


# In[7]:


plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()


# In[ ]:




