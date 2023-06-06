#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib as plt


# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')


# In[5]:


col_names=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']


# In[6]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names=col_names)
df


# In[7]:


irisSet = (df['Species']=='Iris-setosa')
irisSet 


# In[8]:


irisVir=(df['Species']=='Iris-virginica')
print('Iris-virginica')
print(df[irisVir].describe())


# In[9]:


df.mean()


# In[10]:


df.median()


# In[11]:


df.mode()


# In[12]:


df.min()


# In[13]:


df.max()


# In[14]:


df.std()


# In[15]:


df.info()


# In[16]:


df.loc[:,'Sepal_Length'].mean()


# In[17]:


df.mean(axis=0)[0:4]


# In[18]:


df.loc[:,'Sepal_Length'].median()


# In[19]:


df.mean(axis=0)[0:4]
df.mean(axis=1)[0:4]


# In[20]:


df.mode()


# In[21]:


df.mode()
df.loc[:,'Sepal_Length'].mode()


# In[22]:


df.loc[:,'Sepal_Length'].min(skipna = False) # to find min of specific column


# In[23]:


df.loc[:,'Sepal_Length'].max(skipna = False) # to find max of specific column


# In[24]:


#std of specific column
df.loc[:,'Sepal_Length'].std()


# In[25]:


# find  std row wise 
df.std(axis=0)[0:4]


# In[26]:


# find  std col wise 
df.std(axis=1)[0:4]


# In[27]:


df.describe()


# In[37]:


x=df.iloc[:,:4]
y=df.iloc[:,4]
print(x)


# In[38]:


print(y)


# In[39]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=0)


# In[40]:


x_train.shape


# In[41]:


x_test.shape


# In[42]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[45]:


model.fit(x_train,y_train)
LogisticRegression()


# In[48]:


y_pred=model.predict(x_test)


# In[49]:


y_pred


# In[50]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[51]:


confusion_matrix(y_test,y_pred)


# In[52]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[53]:


df.tail(n=5)


# In[54]:


df.index


# In[58]:


df.shape


# In[60]:


df.dtypes


# In[63]:


print(df.columns)


# In[66]:


df.sort_index(axis=1,ascending=False)


# In[69]:


df.sort_values(by='Petal_Width', inplace=True)


# In[70]:


df.iloc[5]


# In[71]:


df[0:3]


# In[72]:


df.iloc[:,:6]


# In[73]:


df.iloc[:5,:7]


# In[74]:


df.iloc[:,1:3]


# In[75]:


df.iloc[1:3,:]


# In[76]:


df.iloc[[1,2,4],[0,2]]


# In[77]:


df.describe(include="all")


# In[78]:


df.head()


# In[79]:


df.corr()


# In[81]:


import seaborn as sns 


# In[82]:


import pandas as pd


# In[83]:


import numpy as np


# In[84]:


import matplotlib.pyplot as plt


# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


df.head()


# In[87]:


corr = df.corr()


# In[88]:


sns.heatmap(corr)


# In[89]:


corr = df.corr()


# In[90]:


sns.heatmap(corr,cmap='Greens')


# In[97]:


sns.histplot(df['Sepal_Length'], kde=False, bins=4)


# In[112]:


sns.rugplot(df['Sepal_Width'])


# In[118]:


sns.swarmplot(x='Species',data=df, hue='Petal_Width')


# In[120]:


sns.stripplot(x='Species',data=df,jitter=True,hue='Petal_Width')


# In[121]:


sns.violinplot(x='Species',y='Petal_Width',data=df)


# In[122]:


sns.boxplot(x='Species',y='Petal_Length', data=df)


# In[123]:


sns.rugplot(df['Sepal_Width'])


# In[124]:


sns.jointplot(x = df['Species'], y=df['Sepal_Length'])


# In[ ]:




