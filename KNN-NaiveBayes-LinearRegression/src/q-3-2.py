#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import random
import sys

dataSet = pd.read_csv("../input_data/data.csv")

trainingData=dataSet.sample(frac=0.8)
validationData=dataSet.drop(trainingData.index)

testingData=pd.read_csv(sys.argv[1])
# In[6]:


def MSE(y,y1):
    ans=0
    for i in range(len(y)):
        A=(y[i]-y1[i][0])**2
        ans+=A
    return ans/len(y)

def MAE(y,y1):
    ans=0
    for i in range(len(y)):
        ans+=abs(y[i]-y1[i])
    return ans/len(y);

def MAP(y,y1):
    ans=0
    for i in range(len(y)):
        A=y[i]-y1[i]
        A/=y[i]
        ans+=A
    return ans*100/len(y)


# In[7]:


temp=[1 for x in range(len(trainingData))]
trainingData.insert(0,'temp',temp)

temp=[1 for x in range(len(testingData))]
testingData.insert(0,'temp',temp)

y=trainingData.pop('Chance of Admit ')

trainingData.pop('Serial No.')
x=trainingData

y = [y]
y=np.array(y)
x=x.values
y=y.transpose()

xt=x.transpose()
x_xt=np.matmul(xt,x)
x_xt_inv=np.linalg.inv(x_xt)
temp=np.matmul(x_xt_inv,xt)
B=np.matmul(temp,y)

test_y=testingData.pop('Chance of Admit ')
test_x=testingData

testingData.pop('Serial No.')
test_x=testingData

test_x=test_x.values

predict = np.matmul(test_x,B)

actual_y=test_y.tolist()
predicted_y=predict

print ("Mean Square Error: ", MSE(actual_y,predicted_y))

print ("Mean Absolute Error: ", MAE(actual_y,predicted_y))

print ("Mean Absolute % Error : ", MAP(actual_y,predicted_y))


# In[ ]:




