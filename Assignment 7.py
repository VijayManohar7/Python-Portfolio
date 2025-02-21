#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# 1
array_1 = np.arange(1, 11).reshape(2, 5)
print("Exercise 1:\n", array_1, "\n")


# In[3]:


# 2
array_2 = np.arange(1, 21)
extracted_elements = array_2[5:16]
print("Exercise 2:\n", extracted_elements, "\n")


# In[4]:


# 3
fruits = pd.Series({'apples': 3, 'bananas': 2, 'oranges': 1})
fruits['pears'] = 4
print("Exercise 3:\n", fruits, "\n")


# In[5]:


# 4
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack'],
    'age': [25, 30, 22, 40, 35, 28, 32, 27, 45, 38],
    'gender': ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
}
df = pd.DataFrame(data)
print("Exercise 4:\n", df, "\n")


# In[6]:


# 5
df['occupation'] = ['Programmer', 'Manager', 'Analyst', 'Programmer', 'Manager', 'Analyst', 'Programmer', 'Manager', 'Analyst', 'Programmer']
print("Exercise 5:\n", df, "\n")


# In[7]:


# 6
df_filtered = df[df['age'] >= 30]
print("Exercise 6:\n", df_filtered, "\n")


# In[8]:


# 7
csv_filename = "dataframe.csv"
df.to_csv(csv_filename, index=False)
df_read = pd.read_csv(csv_filename)
print("Exercise 7:\n", df_read)


# In[ ]:




