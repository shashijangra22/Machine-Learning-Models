#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries to be used
import numpy as np
import pandas as pd
import math
import sklearn
import sys

# In[2]:


# reading the dataset for training the model
dataSet=pd.read_csv("../input_data/Robot1",delimiter=" ",header=None)


# In[3]:


cols=[x for x in range(2,8)]+[1]
inputData=dataSet[cols]


# In[4]:


inputData.columns=["a1","a2","a3","a4","a5","a6","label"]


# In[5]:


trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)

testingDataFrame=pd.read_csv(sys.argv[1],delimiter=" ",header=None)
testingData=testingDataFrame[cols]
testingData.columns=["a1","a2","a3","a4","a5","a6","label"]

# In[6]:


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


# In[7]:


def KNNPredict(k,testingData):
    predictions=[]
    for i in range(len(testingData)):
        ones=0
        for item in distances[i][:k]:
            ones+=item[1]
        zeros=k-ones
        if(ones>zeros):
            ans=1
        else:
            ans=0
        predictions.append(ans)

    truePositive,trueNegative,falsePositive,falseNegative=0,0,0,0
    labels=list(testingData['label'])
    for i in range(len(labels)):
        a=labels[i]
        b=predictions[i]
        if(a==b):
            if(a):
                truePositive+=1
            else:
                trueNegative+=1
        else:
            if(b):
                falsePositive+=1
            else:
                falseNegative+=1
    accuracy=(truePositive+trueNegative)/(len(testingData))
    return accuracy*100


# In[15]:


for k in range(1,int(math.sqrt(len(trainingData))),2):
    print("Accuracy on k: ",k,": ", KNNPredict(k,testingData),"%")


# In[12]:


# SK learn results
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=7)  
classifier.fit(trainingData.drop(columns=['label']), trainingData['label']) 
y_pred = classifier.predict(testingData.drop(columns=['label']))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(testingData['label'], y_pred))  
print(classification_report(testingData['label'], y_pred)) 
print("Accuracy is : ",accuracy_score(testingData['label'], y_pred))


# In[ ]:





# In[ ]:




