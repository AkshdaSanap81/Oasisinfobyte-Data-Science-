#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv("Unemployment in India.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.fillna(0, inplace=True)
df.fillna(df.mean(), inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


df = df.rename(columns={df.columns[0]:'Region',df.columns[3]:'Unemployment_rate',df.columns[4]:'Employed', df.columns[5]:'labour_participation_rate', df.columns[6]:'area'})
df.head()


# In[7]:


df["Region"].unique()


# In[8]:


df2 = pd.read_csv("Unemployment in India.csv")

df2


# In[9]:


df2 = df2.rename(columns={df2.columns[0]:'Region',df2.columns[3]:'Unemployment_rate',df2.columns[4]:'Employed', df2.columns[5]:'labour_participation_rate', df2.columns[6]:'area'})
df2.head()


# In[10]:


heat_maps = df[['Unemployment_rate','Employed','labour_participation_rate']]

heat_maps = heat_maps.corr()

plt.figure(figsize=(12,7))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps, annot=True,cmap='winter');


# In[23]:


plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(12,10))
sns.heatmap(df2.corr(),cmap="Greens")
plt.show()


# In[ ]:




