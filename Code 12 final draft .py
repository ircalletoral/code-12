#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[16]:


current_directory = os.getcwd
print(current_directory)


# In[17]:


new_directory_path = r"C:\Users\ircal\Desktop\Coding\Health Python"
os.chdir(new_directory_path)


# In[18]:


updated_dir = os.getcwd()
print(updated_dir)


# In[19]:


filepath = 'Week14Assignment.txt'
df = pd.read_csv(filepath)


# In[20]:


print(df.columns)


# In[21]:


#Calculating Statistics 
num_readmitted = np.sum(df[' Readmission'])
satisfaction_staff = np.mean(df[' StaffSatisfaction'])
satisfaction_staff = np.mean(df[' CleanlinessSatisfaction'])
satisfaction_staff = np.mean(df[' FoodSatisfaction'])
satisfaction_staff = np.mean(df[' ComfortSatisfaction'])
satisfaction_staff = np.mean(df[' CommunicationSatisfaction'])


# In[22]:


#Printing out Descriptive Statistics
print(f"Number of patients readmited : {num_readmitted}.")
print(f"Average Staff Satisfaction: {satisfaction_staff}.")
print(f"Average Cleanliness  Satisfaction: {satisfaction_staff}.")
print(f"Average Food Satisfaction: {satisfaction_staff}.")
print(f"Average Confort Satisfaction: {satisfaction_staff}.")
print(f"Average Communications Satisfaction: {satisfaction_staff}.")


# In[23]:


#Calculate Overall Satisfaction
df = df.convert_dtypes()
df[' OverallSatisfaction'] = df[[' StaffSatisfaction', ' CleanlinessSatisfaction', ' FoodSatisfaction', ' ComfortSatisfaction', ' CommunicationSatisfaction']].mean(axis = 1)
plt.boxplot(df[' OverallSatisfaction'], showfliers = True)


# In[24]:


#Logistic Regression 
x = df[' OverallSatisfaction'].values.reshape(-1, 1)
y = df[" Readmission"]

log_reg = LogisticRegression().fit(x, y)


# In[25]:


#Correlation Results 

correlation_coefficient = log_reg.coef_[0][0]

if correlation_coefficient > 0:
    print("Logistic regression results indicated a: ")
    if correlation_coefficient > 0.5 :
        print("Moderate Correlation")
    elif correlation_coefficient > 0.7:
        print("Strong Correlation")
    else:
        print("Weak Correlation")
else:
    print("Logistic Regression Results Indicated:")
    print("No Correlation")

print(f"Correlation Coefficient was: {correlation_coefficient}.")


# In[26]:


#Plotting the data 
plt.scatter(x, y)
plt.xlabel("Overall Satisfaction Scores")
plt.ylabel("Logistic Regression - Overall Satisfaction vs Readmission")
plt.plot(x, log_reg.predict(x), color = 'blue')
plt.xlim(2, 5)


# In[ ]:




