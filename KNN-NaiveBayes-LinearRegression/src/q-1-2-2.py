#!/usr/bin/env python
# coding: utf-8

# In[18]:


#importing libraries to be used
import numpy as np
import pandas as pd
import math
import sklearn
import matplotlib.pyplot as plt
import sys

# In[19]:


# reading the dataset for training the model
dataSet=pd.read_csv("../input_data/Iris.csv",delimiter=",",header=None)


# In[20]:


cols=[x for x in range(5)]
inputData=dataSet[cols]


# In[21]:


inputData.columns=["a1","a2","a3","a4","label"]


# In[22]:


trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)

testingDataFrame=pd.read_csv(sys.argv[1],delimiter=",",header=None)
testingData=testingDataFrame[cols]
testingData.columns=["a1","a2","a3","a4","label"]


# In[23]:


def euclidDistance(a,b):
    temp=0
    for key in b:
        if key is not 'label':
            temp+=((b[key]-a[key])**2)
    return math.sqrt(temp)

distances=[]

for vrow in testingData.to_dict('records'):
    temp=[]
    for trow in trainingData.to_dict('records'):
        temp.append([euclidDistance(vrow,trow),trow['label']])
    distances.append(sorted(temp))


# In[24]:


def KNNPredict(k,testingData):
    predictions=[]
    for i in range(len(testingData)):
        kLabels=[item[1] for item in distances[i][:k]]
        uVals,counts=np.unique(kLabels,return_counts=True)
        uValCount=dict(zip(uVals,counts))
        predLabel = max(uValCount.keys(), key=(lambda k: uValCount[k]))
        predictions.append(predLabel)
    labels=list(testingData['label'])
    truths,falses=0,0
    for i in range(len(labels)):
        a=labels[i]
        b=predictions[i]
        if(a==b):
            truths+=1
        else:
            falses+=1
    accuracy=(truths)/(len(testingData))
    return accuracy*100


# In[25]:


# for k in range(1,int(math.sqrt(len(trainingData))),2):
#     print("Accuracy on k: ",k,": ", KNNPredict(k),"%")

accuracies=[KNNPredict(k,testingData) for k in range(1,int(math.sqrt(len(trainingData))))]
plt.plot(range(1,int(math.sqrt(len(trainingData)))), accuracies)
plt.title('Acc vs K Value')
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')
plt.show()

