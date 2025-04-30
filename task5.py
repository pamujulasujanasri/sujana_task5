#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


train=pd.read_csv("train5.csv")
train


# In[4]:


test=pd.read_csv("test5.csv")
test


# In[5]:


train.describe()


# In[6]:


train.info()


# In[7]:


#train=train.drop('Name',axis=1)
train


# In[8]:


train.info()


# In[9]:


train["Sex"]=train['Sex'].astype('string')
#train['Age']=pd.to_numeric(train['Age'],errors='coerce')


# In[10]:


train.info()


# In[11]:


test["Sex"]=test['Sex'].astype('string')


# In[12]:


test.info()


# In[13]:


train.duplicated()


# In[14]:


train.isnull().sum()


# In[15]:


train.dropna()


# In[30]:


train.isnull().sum()


# In[16]:


test.dropna()


# In[17]:


gs=pd.read_csv("gender_submission.csv")


# In[18]:


gs


# In[19]:


gs.describe()


# In[20]:


gs.info()


# In[21]:


train.value_counts()


# In[22]:


test.value_counts()


# In[23]:


gs.value_counts()


# In[24]:


#train.duplicated()
#test.duplicated()
gs.duplicated()


# In[25]:


train['Age'].hist()
plt.show()


# In[26]:


train['Survived'].hist()
plt.show()


# In[27]:


train.boxplot(column=['Survived'])


# In[28]:


train['Sex'].value_counts().plot.bar()


# In[29]:


train.plot.scatter(x='SibSp',y='Parch')
plt.show()


# In[31]:


train[train.isnull().any(axis=1)].head() 


# In[32]:


train1=train.drop('Cabin', axis=1)


# In[33]:


train1


# In[34]:


train1=train1.drop('Name',axis=1)
train1


# In[35]:


train1.isnull().sum()


# In[70]:


import seaborn as sns
cols=train1.columns
cols
colours=['#000099', '#ffff00'] 
sns.heatmap(train1[cols].isnull(),cmap=sns.color_palette(colours),cbar=False)


# In[71]:


median=train1["Age"].median()
median


# In[72]:


train1["Age"]=train1["Age"].fillna(median)


# In[73]:


train1.isnull().sum()


# In[74]:


train1['Embarked']=train1['Embarked'].astype('category')  
train1.info()


# In[75]:


mode=train1["Embarked"].mode()


# In[76]:


mode


# In[77]:


train1["Embarked"]=train1["Embarked"].fillna(mode)
train1


# In[78]:


train1.isnull().sum()


# In[79]:


train1["Age"].hist()


# In[80]:


sns.pairplot(train1)


# In[ ]:





# In[ ]:




