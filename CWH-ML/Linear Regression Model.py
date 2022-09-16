#!/usr/bin/env python
# coding: utf-8

# <FONT COLOR = 'BLUE'>**Importing modules needed for linear regresssion model**</FONT>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


# *Using daibetes dataset to determine how much diabetic a person is and then calculating the squared mean error*

# In[2]:


diabetes = datasets.load_diabetes()


# <font color = 'red'> **Linear Regression Model using SINGLE FEATURE or parameter** </font>
# <br>The data array's index 2 is considered as the only feature for this regression model

# In[3]:


diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[4]:


diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]


# In[5]:


model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)


# In[6]:


diabetes_Y_predicted = model.predict(diabetes_X_test)


# In[7]:


print("Mean squared error is : ", mean_squared_error(diabetes_Y_predicted, diabetes_Y_test))


# In[8]:


print("Weight: ", model.coef_)
print("Intercept: ", model.intercept_)


# In[9]:


plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)


# In[10]:


plt.show()


# <font color = 'red'> **Linear Regression Model using MULTIPLE FEATURE or parameter** </font>

# In[11]:


diabetes_X = diabetes.data
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[12]:


diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]


# In[13]:


model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)


# In[14]:


diabetes_Y_predicted = model.predict(diabetes_X_test)


# In[15]:


print("Mean squared error is : ", mean_squared_error(diabetes_Y_predicted, diabetes_Y_test))

