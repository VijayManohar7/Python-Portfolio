#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

file_path = "Employee.csv"
df_emp = pd.read_csv(file_path)

print(df_emp.info())
print(df_emp.describe())


# In[2]:


# Standardizing company names
df_emp['Company'].replace({
    'Tata Consultancy Services': 'TCS',
    'CTS': 'Cognizant',
    'Congnizant': 'Cognizant',
    'Infosys Pvt Lmt': 'Infosys'
}, inplace=True)


# In[3]:


# Replacing incorrect age values
df_emp['Age'].replace(0, np.nan, inplace=True)


# In[4]:


# Standardizing city names
df_emp['Place'].replace({
    'Podicherry': 'Pondicherry',
    'Cochin': 'Kochi'
}, inplace=True)


# In[5]:


# Dropping redundant column (Country)
df_emp.drop(columns=['Country'], inplace=True, errors='ignore')


# In[6]:


# Handling missing values (filling with median for numerical, mode for categorical)
df_emp['Age'].fillna(df_emp['Age'].median(), inplace=True)
df_emp['Salary'].fillna(df_emp['Salary'].median(), inplace=True)
df_emp['Company'].fillna(df_emp['Company'].mode()[0], inplace=True)
df_emp['Place'].fillna(df_emp['Place'].mode()[0], inplace=True)


# In[7]:


# Removing duplicate rows
df_emp.drop_duplicates(inplace=True)


# In[8]:


# Outlier Detection using IQR
Q1 = df_emp['Salary'].quantile(0.25)
Q3 = df_emp['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[9]:


# Removing outliers using trimming
df_emp = df_emp[(df_emp['Salary'] >= lower_bound) & (df_emp['Salary'] <= upper_bound)]


# In[10]:


# Filter data (Age > 40 and Salary < 5000)
filtered_df = df_emp[(df_emp['Age'] > 40) & (df_emp['Salary'] < 5000)]


# In[11]:


# Plot Age vs Salary
plt.figure(figsize=(8,5))
sns.scatterplot(x=filtered_df['Age'], y=filtered_df['Salary'])
plt.title("Age vs Salary (Filtered Data)")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()


# In[12]:


# Count people per city and visualize
city_counts = df_emp['Place'].value_counts()

plt.figure(figsize=(10,5))
sns.barplot(x=city_counts.index, y=city_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Number of Employees by City")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()


# In[13]:


# Encoding categorical variables using Label Encoding
label_enc = LabelEncoder()
df_emp['Company'] = label_enc.fit_transform(df_emp['Company'])
df_emp['Place'] = label_enc.fit_transform(df_emp['Place'])


# In[14]:


# Feature Scaling
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df_emp[['Age_scaled', 'Salary_scaled']] = scaler_standard.fit_transform(df_emp[['Age', 'Salary']])
df_emp[['Age_minmax', 'Salary_minmax']] = scaler_minmax.fit_transform(df_emp[['Age', 'Salary']])


# In[15]:


# Display the processed dataset
print(df_emp.head())


# In[ ]:




