#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("C:/Users/hp/Downloads/insurance (1).csv")
data


# In[2]:


data.columns


# In[3]:


#technical Summary
data.info()


# In[4]:


#Statistical Summary
data.describe()


# In[9]:


#EDA:-Exp;oratory data analysis
#Exploring Data Techniques
#plt.scatter(data['sex'],data['region'])
plt.scatter(data['sex'],data['smoker'])


# In[ ]:


#1. Column information
#2.top rows
#3.Bottom rows
#4.technical summary: datatypes of all the columns and check is their 
#any null present
#5.Statical summary: mean, max,min, count,avg
#6.correleation between numerical parameters
#7.experiment how each input affect for predicting the profit
#8.if data distribution is not normalize apply appropriate method to analyze data
#9.if any categorical parameters were present lets convert it into numerics
#by appropriate methods


# In[3]:


data.head()
#Top 5 columns


# In[4]:


data.tail()
#Last 5 columns


# In[5]:


#checking for null values
data.isna().sum()


# In[6]:


#Correlation graph between numerical elements
sns.heatmap(data.corr(),annot=True)


# In[7]:


#Correlation on data
data.corr()


# In[13]:


#Univariate and Bivariate analysis
#Univariate Analysis:-WHEN A SINGLE COLUMN IS USED:HISTOGRAM AND BOXPLOT
#BIVARIATE ANALYSIS:-WHEN WE COMPARE TWO OR MORE COLUMNS SIMULTANEOUSLY:SCATTER AND LINE
#UNIVARIATE:-
sns.distplot(data['age'])
sns.distplot(data['children'],kde=True)


# In[14]:


sns.boxplot(data['children'])


# In[15]:


#Bivariate Analysis
plt.scatter(data['age'],data['children'])


# In[19]:


#line plot
plt.plot(data['age'],data['children'])

