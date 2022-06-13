#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
from matplotlib import pyplot as plt 
import numpy as np
from keras.datasets import mnist


# In[ ]:





# In[5]:


objects=mnist 
(train_img,train_lab),(test_img,test_lab)=objects.load_data()

for i in range(20): 
 plt.subplot(4,5,i+1) 
 plt.imshow(train_img[i],cmap='gray_r') 
 plt.title("Digit : {}".format(train_lab[i])) 
 plt.subplots_adjust(hspace=0.5) 
 plt.axis('off')


# In[6]:


print('Training images shape : ',train_img.shape) 
print('Testing images shape : ',test_img.shape)

print('How image looks like : ') 
print(train_img[0]) 


# In[7]:


plt.hist(train_img[0].reshape(784),facecolor='orange') 
plt.title('Pixel vs its intensity',fontsize=16) 
plt.ylabel('PIXEL') 
plt.xlabel('Intensity') 


# In[8]:


train_img=train_img/255.0
test_img=test_img/255.0
print('How image looks like after normalising: ') 
print(train_img[0]) 


# In[9]:


from keras.models import Sequential 
from keras.layers import Flatten,Dense 
model=Sequential()
input_layer= Flatten(input_shape=(28,28)) 
model.add(input_layer) 
hidden_layer1=Dense(512,activation='relu') 
model.add(hidden_layer1) 
hidden_layer2=Dense(512,activation='relu') 
model.add(hidden_layer2) 
output_layer=Dense(10,activation='softmax') 
model.add(output_layer)


# In[10]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[11]:


model.fit(train_img,train_lab,epochs=3)


# In[12]:


model.save('project.h5')


# In[ ]:





# In[13]:


model.predict(test_img)


# In[3]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# In[10]:


import sys
print(sys.getrecursionlimit())


# In[11]:


sys.setrecursionlimit(5000)


# In[12]:


def load_img(file):
    img=load_img(file)
    img=img_to_array(img)
    img=img.reshape(28,28,1)
    img=img.astype('float32')
    img=img/255.0
    return img
    


# In[ ]:





# In[13]:


from IPython.display import Image 
Image('5img.png') 


# In[ ]:


img =load_img('5img.png') 


# In[ ]:


digit=model.predict(img) 
print('Predicted value : ',np.argmax(digit)) 

