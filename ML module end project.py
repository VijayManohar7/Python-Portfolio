#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('CarPrice_Assignment.csv')


# In[3]:


print(df.head())


# In[4]:


print(df.info())


# In[5]:


df.drop(columns=['car_ID', 'CarName'], inplace=True)


# In[6]:


df = pd.get_dummies(df, drop_first=True)


# In[7]:


print(df.isnull().sum())


# In[8]:


from sklearn.preprocessing import StandardScaler

X = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[11]:


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR()
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model


# In[12]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'R2 Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values(by='R2 Score', ascending=False))


# In[13]:


importances = trained_models['Random Forest'].feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


# In[14]:


# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()


# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)


# In[17]:


best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("R2 after tuning:", r2_score(y_test, y_pred_tuned))


# In[ ]:




