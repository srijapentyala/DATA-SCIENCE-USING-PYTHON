#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Enter the income:")
Income=int(input())
print("Enter the House rent allowance:")
hra=int(input())
print("Enter the deductions:")
deduction=int(input())
tax_income=Income-hra*12-deduction
tax1=tax_income-300000
if(tax1<300000):
    print("No tax")
elif(tax1>300000 and tax1<600000):
    tax=0.1*tax_income
    print(tax)
elif(tax1>600000 and tax1<1000000):
    tax=0.15*tax_income
    print(tax)
elif(tax1>1000000):
    tax=0.2*tax_income
    print(int(tax))


# In[ ]:




