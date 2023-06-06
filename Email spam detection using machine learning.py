#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import LogisticRegression


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


from sklearn.metrics import accuracy_score


# In[9]:


df = pd.read_csv("spam.csv",encoding='latin-1')


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.ndim


# In[16]:


df.isna().sum()


# In[17]:


df.isnull


# In[18]:


df.drop(columns=df[["Unnamed: 2","Unnamed: 3","Unnamed: 4"]],axis=1,inplace=True)
print(df.head(7))


# In[19]:


df.columns=['spam/ham','sms']
df


# In[21]:


df.dtypes


# In[22]:


df.shape


# In[24]:


df.nunique()


# In[28]:


df.min()


# In[27]:


df.max()


# In[29]:


x=df['sms']
x


# In[30]:


y=df['spam/ham']
y


# In[31]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=3)


# In[32]:


print("Training set=(",x_train.shape,y_train.shape,")")
print("Training set=(",x_test.shape,y_test.shape,")")


# In[33]:


feature=TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)
feature


# In[38]:


import seaborn as sns


# In[43]:


sns.countplot(df['spam/ham'])
plt.show()


# In[45]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


df = pd.read_csv("spam.csv",encoding='latin-1')
df.head()


# In[ ]:




