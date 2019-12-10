#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from pprint import pprint


# In[2]:


dataSet=pd.read_csv("../input_data/train.csv")


# In[3]:


def makeDict(temp):
    a=np.array(temp)
    unique, counts=np.unique(a,return_counts=True)
    return dict(zip(unique,counts))


# In[4]:


def calcEntropy(q):
    if q==0 or q==1:
        return 0
    return -(q*math.log2(q) + (1-q)*math.log2(1-q))


# In[5]:


#calculating gini index
def calcGini(q):
    return 2*q*(1-q)


# In[6]:


# calculating misclassification
def calcMisClass(q):
    return min(q,1-q)


# In[7]:


def calcImpurity(q,iMeasure):
    if(iMeasure==0):
        return calcEntropy(q)
    if(iMeasure==1):
        return calcGini(q)
    return calcMisClass(q)


# In[8]:


def classifyData(data):
    for item in data:
        temp=makeDict(data[item])
        if(len(temp)>10):
            x=len(temp)//10 + 1
            listOfKeys=list(temp.keys())
            newCol=[]
            for row in data[item]:
                ind=listOfKeys.index(row)
                upperBorder=(ind//x + 1)*x
                if(upperBorder>=len(temp)):
                    upperBorder=-1
                newCol.append(listOfKeys[upperBorder])
            data[item]=newCol
    return data


# In[9]:


def calcBestAttr(impurity,data,iMeasure):
    iGains={}
    for col in data:
        if(col=="left"):
            continue
        uValCount=makeDict(data[col])
        wAvg=0
        for category in uValCount:
            a=data[(data[col]==category) & (data['left']==1)]
            q=len(a)/uValCount[category]
            wAvg+=((uValCount[category]/len(data))*calcImpurity(q,iMeasure))
        iGains[col]=impurity-wAvg
    try:
        maxCategory = max(iGains.keys(), key=(lambda k: iGains[k]))
    except:
        return None
    return maxCategory


# In[10]:


def predict(model,root,sample):
    key=list(root.keys())[0]
    if(key=='value'):
        return root[key]
    try:
        root=root[key]
        if sample[key] in root.keys():
            return predict(model,root[sample[key]],sample)
        else:
            for k in root.keys():
                if(sample[key]<=k):
                    return predict(model,root[k],sample)
            return 0
    except:
        return 0


# In[11]:


def predictChars(data):
    truePositive,falsePositive,trueNegative,falseNegative=0,0,0,0
    for sample in data:
        pred_label=predict(DTree,DTree,sample)
        if(pred_label==sample['left']):
            if(pred_label):
                truePositive+=1
            else:
                trueNegative+=1
        else:
            if(pred_label):
                falsePositive+=1
            else:
                falseNegative+=1
    accuracy=(truePositive+trueNegative)*100/len(data)
    recall=truePositive/(truePositive+falseNegative)
    precision=truePositive/(truePositive+falsePositive)
    f1score=2/((1/precision)+(1/recall))
    print("Accuracy is : {}".format(accuracy))
    print("Recall is : {}".format(recall))
    print("precision is : {}".format(precision))
    print("f1score is : {}".format(f1score))


# In[12]:


inputData=dataSet


# In[13]:


trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)
trainingData=classifyData(trainingData)


# In[14]:


def makeTree(root,data,iMeasure):
    a=len(data[data.left==1])
    if(a==0 or a==len(data)):
        root.update({'value':1 if a else 0})
        return
    q=a/len(data)
    newNode=calcBestAttr(calcImpurity(q,iMeasure),data,iMeasure)
    if(newNode is None):
        b=len(data)-a
        if(a>=b):
            root.update({'value':1})
        else:
            root.update({'value':0})
        return
    root.update({newNode:{}})
    root=root[newNode]
    for category in np.unique(data[newNode]):
        root.update({category:{}})
        temp=data[data[newNode]==category]
        makeTree(root[category],temp.drop(columns=[newNode]),iMeasure)


# In[15]:


DTree={}
print("Training with Entropy as impurity measure...")
makeTree(DTree,trainingData,0)
print("Training Finished!")
print("Predicting Labels...")
predictChars(validationData.to_dict('records'))


# In[16]:


DTree={}
print("Training with GiniIndex as impurity measure...")
makeTree(DTree,trainingData,1)
print("Training Finished!")
print("Predicting Labels...")
predictChars(validationData.to_dict('records'))


# In[17]:


DTree={}
print("Training with Misclassification as impurity measure...")
makeTree(DTree,trainingData,2)
print("Training Finished!")
print("Predicting Labels...")
predictChars(validationData.to_dict('records'))

