#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from q_1_1 import LassoRegression
from q_1_2 import RidgeRegression


# In[2]:


dataSet=pd.read_csv('../input_data/data.csv').drop(['Serial No.'],axis=1)


# In[3]:


trainingData=dataSet.sample(frac=0.8,random_state=100)
validationData=dataSet.drop(trainingData.index)
trainingLabel=trainingData.pop('Chance of Admit ')
validationLabel=validationData.pop('Chance of Admit ')


# In[4]:


X_Train=np.array(trainingData)
X_Train=(X_Train-X_Train.mean(axis=0))/X_Train.std(axis=0)
Y_Train=np.array(trainingLabel).reshape((len(trainingLabel),1))

X_Test=np.array(validationData)
X_Test=(X_Test-X_Test.mean(axis=0))/X_Test.std(axis=0)
Y_Test=np.array(validationLabel).reshape((len(validationLabel),1))


# In[5]:


RidgeModel=RidgeRegression(alpha=0.1,lamda=5,epochs=1000)
RidgeModel.fit(X_Train,Y_Train)

LassoModel=LassoRegression(alpha=0.1,lamda=5,epochs=1000)
LassoModel.fit(X_Train,Y_Train)


# In[6]:


thetaR=RidgeModel.theta
thetaL=LassoModel.theta
diffTheta=thetaR-thetaL


# In[7]:


print("Weights of Ridge Regression:\n",thetaR)
print("Weights of Lasso Regression:\n",thetaL)
print("Weight difference b/w Ridge & Lasso:\n",diffTheta)


# In[8]:


print("Error on Ridge Regression:",RidgeModel.MSE(X_Test,Y_Test))


# In[9]:


print("Error on Lasso Regression:",LassoModel.MSE(X_Test,Y_Test))


# In[ ]:




