#!/usr/bin/env python
# coding: utf-8

# In[45]:


#importing libraries to be used
import numpy as np
import pandas as pd
import math
import random
import sys

# In[46]:


# reading the dataset for training the model
attrList = ['id','age','exp','income','zip','familySize','avgMonExp','education','mortgage','label','securities','certDeposit','IT_Banking','creditCard']
dataSet=pd.read_csv("../input_data/loan.csv",names=attrList)
inputData=dataSet.drop("id",axis=1)

numCols = ['age','exp','income','avgMonExp','education','mortgage']
catCols = ['familySize','securities','certDeposit','IT_Banking','creditCard']


# In[47]:


trainingData=inputData.sample(frac=0.8)
validationData=inputData.drop(trainingData.index)

testingDataFrame=pd.read_csv(sys.argv[1],names=attrList)
testingData=testingDataFrame.drop("id",axis=1)

# In[48]:


def calcProb(x,mean,std):
    exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exp


# In[49]:


catInfo={}
for col in catCols:
    catInfo[col]={}
    uVals=trainingData[col].unique()
    for x in uVals:
        catInfo[col][x]={}
        
        temp=trainingData[trainingData[col]==x]
        
        positive=temp[temp["label"]==1][col].count()/trainingData[trainingData["label"]==1][col].count()
        catInfo[col][x]['yes']=positive
        
        negative=temp[temp["label"]==0][col].count()/trainingData[trainingData["label"]==0][col].count()
        catInfo[col][x]["no"]=negative


# In[50]:


numericMean={}
numericDev={}
for i in numCols:
    numericMean[i]={}
    numericDev[i]={}
    
    positive=trainingData[trainingData['label']==1][i].mean()
    numericMean[i]['yes']=positive
    
    negative=trainingData[trainingData['label']==0][i].mean()
    numericMean[i]['no']=negative
    
    devPositive=trainingData[trainingData['label']==1][i].std()
    numericDev[i]['yes']=devPositive
    
    devNegative=trainingData[trainingData['label']==0][i].std()
    numericDev[i]['no']=devNegative;


# In[51]:


POS=len(trainingData[trainingData['label']==1])/len(trainingData)
NEG=1-POS
predictions=[]
for index,data in testingData.iterrows():
    pos=1
    nos=1
    for cat in catCols:
        pos=pos*catInfo[cat][data[cat]]['yes']
        nos=nos*catInfo[cat][data[cat]]['no']
    for num in numCols:
        
        posProb=calcProb(data[num],numericMean[num]['yes'],numericDev[num]['yes'])
        pos=pos*posProb
        
        negProb=calcProb(data[num],numericMean[num]['no'],numericDev[num]['no'])
        nos=nos*negProb
    pos=pos*POS
    nos=nos*NEG
    if pos>nos:
        predictions.append(1)
    else:
        predictions.append(0)

given=testingData['label'].tolist()
c=0
for i in range(len(given)):
    if(given[i]==predictions[i]):
        c+=1
print("accuracy: ",c/len(given))


# In[ ]:




