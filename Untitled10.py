#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[4]:


from sklearn import preprocessing


# In[10]:


data = pd.read_csv('C:/Users/Mustafa/Desktop/datasets/Student-Project-20230815T154511Z-001/Student-Project/StudentsPerformance.csv')


# In[11]:


from sklearn.impute import SimpleImputer


# In[12]:


df= pd.DataFrame(data)


# In[13]:


df = df.drop(['test preparation course'], axis = 1)


# In[14]:


df.head()


# In[15]:


df = df.drop(['lunch'], axis = 1)


# In[16]:


df.head()


# In[19]:


le = preprocessing.LabelEncoder()


# In[20]:


object_data = df.select_dtypes(include = [ 'object'])


# In[21]:


object_data.head()


# In[23]:


for i in range( object_data.shape[1]):
    object_data.iloc[:,i]=le.fit_transform(object_data.iloc[:,i])
    
    


# In[24]:


object_data.head()


# In[25]:


num_data = df.select_dtypes(include = 'int')


# In[31]:


new_data = pd.concat([object_data,num_data], axis = 1)


# In[29]:


new_data.head()


# In[32]:


print(new_data)


# In[33]:


import seaborn as sns


# In[34]:


c = new_data.corr()


# In[35]:


print(c)


# In[37]:


sns.heatmap(c, annot= True)


# In[ ]:




