#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# In[2]:


file_path = "house_price.csv"
df = pd.read_csv(file_path)


# In[3]:


print(df.info())
print(df.describe())


# In[4]:


# Outlier Detection
mean_price_per_sqft = df['price_per_sqft'].mean()
std_price_per_sqft = df['price_per_sqft'].std()


# In[5]:


# Mean & Standard Deviation Method
lower_bound_std = mean_price_per_sqft - 3 * std_price_per_sqft
upper_bound_std = mean_price_per_sqft + 3 * std_price_per_sqft
outliers_std = df[(df['price_per_sqft'] < lower_bound_std) | (df['price_per_sqft'] > upper_bound_std)]


# In[6]:


# Percentile Method
lower_bound_pct = np.percentile(df['price_per_sqft'], 1)
upper_bound_pct = np.percentile(df['price_per_sqft'], 99)
outliers_pct = df[(df['price_per_sqft'] < lower_bound_pct) | (df['price_per_sqft'] > upper_bound_pct)]


# In[7]:


# IQR Method
Q1 = np.percentile(df['price_per_sqft'], 25)
Q3 = np.percentile(df['price_per_sqft'], 75)
IQR = Q3 - Q1
lower_bound_iqr = Q1 - 1.5 * IQR
upper_bound_iqr = Q3 + 1.5 * IQR
outliers_iqr = df[(df['price_per_sqft'] < lower_bound_iqr) | (df['price_per_sqft'] > upper_bound_iqr)]


# In[8]:


# Z-Score Method
df['z_score'] = (df['price_per_sqft'] - mean_price_per_sqft) / std_price_per_sqft
outliers_z = df[np.abs(df['z_score']) > 3]


# In[9]:


# Outlier Removal
# Trimming using IQR
df_trimmed = df[(df['price_per_sqft'] >= lower_bound_iqr) & (df['price_per_sqft'] <= upper_bound_iqr)]


# In[10]:


# Capping using Percentile Method
df_capped = df.copy()
df_capped['price_per_sqft'] = np.where(df_capped['price_per_sqft'] > upper_bound_pct, upper_bound_pct, df_capped['price_per_sqft'])
df_capped['price_per_sqft'] = np.where(df_capped['price_per_sqft'] < lower_bound_pct, lower_bound_pct, df_capped['price_per_sqft'])


# In[11]:


# Box Plot Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.boxplot(y=df['price_per_sqft'])
plt.title("Original Data")

plt.subplot(1, 3, 2)
sns.boxplot(y=df_trimmed['price_per_sqft'])
plt.title("IQR Trimmed")

plt.subplot(1, 3, 3)
sns.boxplot(y=df_capped['price_per_sqft'])
plt.title("Percentile Capped")

plt.tight_layout()
plt.show()


# In[12]:


# Normality Check
plt.figure(figsize=(12, 5))
sns.histplot(df_capped['price_per_sqft'], bins=50, kde=True)
plt.title("Histogram of Price per Sqft (Capped)")
plt.show()


# In[13]:


# Skewness and Kurtosis Before Transformation
skewness_before = skew(df_capped['price_per_sqft'])
kurtosis_before = kurtosis(df_capped['price_per_sqft'])


# In[14]:


# Log Transformation
df_capped['price_per_sqft_log'] = np.log1p(df_capped['price_per_sqft'])


# In[15]:


# Histogram After Transformation
plt.figure(figsize=(12, 5))
sns.histplot(df_capped['price_per_sqft_log'], bins=50, kde=True)
plt.title("Histogram of Log Transformed Price per Sqft")
plt.show()


# In[16]:


# Skewness and Kurtosis After Transformation
skewness_after = skew(df_capped['price_per_sqft_log'])
kurtosis_after = kurtosis(df_capped['price_per_sqft_log'])
print(f"Skewness Before: {skewness_before}, Kurtosis Before: {kurtosis_before}")
print(f"Skewness After: {skewness_after}, Kurtosis After: {kurtosis_after}")


# In[17]:


# Correlation Heatmap
correlation_matrix = df_capped.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[18]:


# Scatter Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_capped['total_sqft'], y=df_capped['price'], alpha=0.5)
plt.title("Price vs Total Sqft")

plt.subplot(1, 2, 2)
sns.scatterplot(x=df_capped['total_sqft'], y=df_capped['price_per_sqft'], alpha=0.5)
plt.title("Price Per Sqft vs Total Sqft")

plt.tight_layout()
plt.show()


# In[ ]:




