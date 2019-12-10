#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing libraries to be used
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
dataSet=pd.read_csv("../input_data/train.csv") # reading dataset for training
trainingData=dataSet.sample(frac=0.8,random_state=200) # splitting the data set into 80% for training
graph=sb.scatterplot(x="satisfaction_level",y="average_montly_hours",hue="left", style="left", data=trainingData)
plt.show()


# In[ ]:




