#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


df=pd.read_csv("Advertising.csv") 


# In[3]:


df.head() 


# In[4]:


df.tail() 


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df=df.drop(columns=["Unnamed: 0"])


# In[9]:


df 


# In[10]:


x=df.iloc[:, 0:-1]


# In[11]:


x


# In[12]:


y=df.iloc[:,-1]


# In[13]:


y


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[15]:


x_train


# In[16]:


x_test


# In[17]:


y_train


# In[18]:


y_test


# In[19]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[20]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lr=LinearRegression()


# In[23]:


lr.fit(x_train_scaled,y_train)


# In[24]:


y_pred=lr.predict(x_test_scaled)


# In[25]:


from sklearn.metrics import r2_score


# In[26]:


r2_score(y_test,y_pred)


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


plt.scatter(y_test,y_pred,c='g')

