#!/usr/bin/env python
# coding: utf-8

# In[11]:


#importing libraries to be used
import numpy as np
import pandas as pd
import math


# In[12]:


# reading the dataset for training the model
dataSet=pd.read_csv("../input_data/train.csv")


# In[13]:


inputData=dataSet


# In[14]:


# helper function to get counts of unique keys in a column
def makeDict(temp):
    a=np.array(temp)
    unique, counts=np.unique(a,return_counts=True)
    return dict(zip(unique,counts))


# In[15]:


# entropy calculation
def calcEntropy(q):
    if q==0 or q==1:
        return 0
    return -(q*math.log2(q) + (1-q)*math.log2(1-q))


# In[16]:


# classifying the numerical data to categorical data
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


# In[17]:


# calculating best attribute on the basis of max information Gain
def calcBestAttr(entropy,data):
    iGains={}
    for col in data:
        if(col=="left"):
            continue
        uValCount=makeDict(data[col])
        wAvg=0
        for category in uValCount:
            a=data[(data[col]==category) & (data['left']==1)]
            q=len(a)/uValCount[category]
            wAvg+=((uValCount[category]/len(data))*calcEntropy(q))
        iGains[col]=entropy-wAvg
    try:
        maxCategory = max(iGains.keys(), key=(lambda k: iGains[k]))
    except:
        return None
    return maxCategory


# In[18]:


# traversing the Decision tree model on a testing sample 
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


# In[19]:


# function to predict accuracy, precision, recall and f1score
def predictChars(data):
    truePositive,falsePositive,trueNegative,falseNegative=0,0,0,0
    for sample in data: # calling predict function on every data sample
        pred_label=predict(DTree,DTree,sample)
        if(pred_label==sample['left']):
            if(pred_label): # if prediction is correct
                truePositive+=1
            else:
                trueNegative+=1
        else: # if prediction is wrong
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


# In[20]:


# making the decision tree using id3 algorithm
def makeTree(root,data):
    a=len(data[data.left==1])
    if(a==0 or a==len(data)): # if a pure data stage reaches
        root.update({'value':1 if a else 0})
        return
    q=a/len(data)
    newNode=calcBestAttr(calcEntropy(q),data) # calculate best attribute
    if(newNode is None): # if not enought data left, mark the label based on purity
        b=len(data)-a
        if(a>=b):
            root.update({'value':1})
        else:
            root.update({'value':0})
        return
    root.update({newNode:{}}) # creating new attribute node
    root=root[newNode]
    for category in np.unique(data[newNode]): # branching the tree further on unique values
        root.update({category:{}})
        temp=data[data[newNode]==category]
        makeTree(root[category],temp.drop(columns=[newNode])) # recursively making the decision tree further


# In[21]:


# splitting data into training(80%) and validation(20%) parts
trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)
trainingData=classifyData(trainingData)


# In[22]:


DTree={} # initializing empty Decision Tree
makeTree(DTree,trainingData) # calling the makeTree function


# In[23]:


predictChars(validationData.to_dict('records')) # calling the predict chars function


# In[ ]:




