#!/usr/bin/env python
# coding: utf-8

# In[14]:


def evenorodd(n):
    if(n%2==0):
        print(str(n)+"is Even")
    else:
        print(str(n)+"is Odd")
def div(n):
    if(n%5==0):
        print(str(n)+"is divisible by 5")
    else:
        print(str(n)+"is not divisible by 5")
def prime(n):
    flag=True
    for i in range(2,n):
        if(n%i==0):
            flag=False
    if(flag):
        print(str(n)+" is Prime")
    else:
        print(str(n)+"is not prime")
def sum(n):
    return int((n*(n+1))/2)
           
print("Enter the number:")
n=int(input())
prime(n)
evenorodd(n)
div(n)
print("The sum of all natural numbers from 1 to",n,"is",sum(n))


# In[ ]:




