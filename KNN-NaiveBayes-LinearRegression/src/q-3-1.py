#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
import sys

dataSet = pd.read_csv("../input_data/data.csv")

trainingData=dataSet.sample(frac=0.8,random_state=200)
validationData=dataSet.drop(trainingData.index)

testingData=pd.read_csv(sys.argv[1])
# In[3]:


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

print(predict)


# In[ ]:




