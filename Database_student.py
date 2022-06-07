#!/usr/bin/env python
# coding: utf-8

# In[1]:


student_details={'stud1':{'Name':'Srija','Rollno':'19881A0540','ssc':10,'inter':10,'cgpa':9.56,'language':'python'},
                 'stud2':{'Name':'Sarita','Rollno':'19881A05M6','ssc':10,'inter':9,'cgpa':9.45,'language':'JAVA'}}
for key,value in student_details[input()].items():
    print(key,":",value)
'''for key,value in student_details.items():
    print(key,"Details")
    for key1,value1 in student_details[key].items():
        print(key1,":",value1)
    print("****")'''


# In[ ]:




