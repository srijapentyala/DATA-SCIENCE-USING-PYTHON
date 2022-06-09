#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dict1={"player":['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20'],
       "runs":[1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,90,100,110,120,130,140,150,160,170,180],
       "wickets":[9,10,11,12,13,14,15,16,17,18,110,120,130,140,150,160,170,180,190,200],
      "type_of_player":["BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN","BATSMEN",
                        "BOWLER","BOWLER","BOWLER","BOWLER","BOWLER","BOWLER","BOWLER","BOWLER","BOWLER","BOWLER"]}


# In[3]:


d2=pd.DataFrame(dict1)


# In[4]:


d2


# In[5]:


tfile = open('data.txt', 'a')


# In[6]:


tfile.write(d2.to_string())


# In[7]:


tfile.close()


# In[11]:


import matplotlib.pyplot as plt


# In[22]:


x=d2["player"]
y=d2["runs"]
plt.scatter(x,y)
plt.xlabel("players")
plt.ylabel("runs")
plt.show()


# In[23]:


x=d2["player"]
y=d2["wickets"]
plt.scatter(x,y)
plt.xlabel("players")
plt.ylabel("wickets")
plt.show()


# In[31]:


x=np.array(['batsman','bowler'])
y=d2['type_of_player'].value_counts()
plt.bar(x,y,color="blue")


# In[ ]:




