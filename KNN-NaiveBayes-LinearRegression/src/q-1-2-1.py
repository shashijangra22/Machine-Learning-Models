#!/usr/bin/env python
# coding: utf-8

# In[35]:


#importing libraries to be used
import numpy as np
import pandas as pd
import math
import sklearn
import sys

# In[36]:


# reading the dataset for training the model
dataSet=pd.read_csv("../input_data/Iris.csv",delimiter=",",header=None)


# In[37]:


cols=[x for x in range(5)]
inputData=dataSet[cols]


# In[38]:


inputData.columns=["a1","a2","a3","a4","label"]


# In[39]:


trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)

testingDataFrame=pd.read_csv(sys.argv[1],delimiter=",",header=None)
testingData=testingDataFrame[cols]
testingData.columns=["a1","a2","a3","a4","label"]

# In[40]:


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


# In[41]:


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


# In[42]:


for k in range(1,int(math.sqrt(len(trainingData))),2):
    print("Accuracy on k: ",k,": ", KNNPredict(k,testingData),"%")

# SK learn results
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=7)  
classifier.fit(trainingData.drop(columns=['label']), trainingData['label']) 
y_pred = classifier.predict(testingData.drop(columns=['label']))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(testingData['label'], y_pred))  
print(classification_report(testingData['label'], y_pred)) 
print("Accuracy is : ",accuracy_score(testingData['label'], y_pred))