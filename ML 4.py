#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target


# In[3]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())


# In[4]:


# Splitting data into features and target
X = df.drop(columns=['Target'])
y = df['Target']


# In[5]:


# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(),
    "k-Nearest Neighbors": KNeighborsClassifier()
}


# In[8]:


# Model evaluation results
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    results[name] = {"Accuracy": accuracy, "Report": report}


# In[9]:


# Convert results to DataFrame
accuracy_df = pd.DataFrame({name: [result["Accuracy"]] for name, result in results.items()})
print(accuracy_df)


# In[10]:


# Identify best and worst models
best_model = accuracy_df.idxmax(axis=1).values[0]
worst_model = accuracy_df.idxmin(axis=1).values[0]
print(f"Best performing model: {best_model}")
print(f"Worst performing model: {worst_model}")


# In[ ]:




