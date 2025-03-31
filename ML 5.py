#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# In[2]:


# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)


# In[3]:


# Feature scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


# In[4]:


# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

df['KMeans_Cluster'] = kmeans_labels


# In[5]:


# Visualizing KMeans Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=kmeans_labels, palette='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering of Iris Dataset')
plt.legend()
plt.show()


# In[6]:


# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(df_scaled)
df['Hierarchical_Cluster'] = hierarchical_labels


# In[7]:


# Dendrogram
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(df_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# In[8]:


# Visualizing Hierarchical Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=hierarchical_labels, palette='coolwarm', s=50)
plt.title('Hierarchical Clustering of Iris Dataset')
plt.show()


# In[ ]:




