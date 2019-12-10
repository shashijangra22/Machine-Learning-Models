#!/usr/bin/env python
# coding: utf-8

# In[208]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# In[209]:


dataSet=pd.read_csv('../input_data/data.csv').drop(['Serial No.'],axis=1)
Y = np.array(dataSet.pop('Chance of Admit '))
X = np.array(dataSet)
X = (X-X.mean(axis=0))/X.std(axis=0)


# In[210]:


def divisors(X):
    div=[]
    for d in range(2,X+1):
        if X%d==0:
            div.append(d)
    return div


# In[211]:


class RidgeRegression():
    
    def __init__(self,alpha,lamda,epochs):
        self.alpha=alpha
        self.lamda=lamda
        self.epochs=epochs
        
    def Cost(self,X,Y):
        X_theta=np.dot(X,self.theta)+self.bias
        return (0.5/len(X))*(np.sum((X_theta-Y)**2)+self.lamda*np.sum(self.theta**2))
    
    def MSE(self,Y_predict,Y):
        return (0.5/len(X))*(np.sum((Y_predict-Y)**2))
    
    def GD(self,X,Y):
        theta=np.zeros((X.shape[1],1))
        bias=1
        for e in range(self.epochs):
            X_theta=np.dot(X,theta)+bias
            dBias=np.sum(X_theta-Y)
            regularizedTerm=np.dot(X.T,(X_theta-Y))+self.lamda*theta
            bias=bias-(self.alpha/len(X)*dBias)
            theta=theta-(self.alpha/len(X)*regularizedTerm)
        return theta,bias
    
    def fit(self,X,Y):
        self.theta,self.bias=self.GD(X,Y)
    
    def predict(self,X):
        return np.dot(X,self.theta)+self.bias


# In[212]:


errors=[]
for k in divisors(len(X)):
    kf=KFold(n_splits=k, random_state=100, shuffle=False)
    error=0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        Y_train=Y_train.reshape((len(Y_train),1))
        Y_test=Y_test.reshape((len(Y_test),1))

        model=RidgeRegression(alpha=0.1,lamda=5,epochs=1000)
        
        model.fit(X_train,Y_train)
        Y_predict=model.predict(X_test)

        error+=model.MSE(Y_predict,Y_test)
    errors.append(error/k)
    if k==len(X):
        print("Error for K: {} (Leave One Out Cross-Validation) is: {}".format(k,error/k))
    else:
        print("Error for K: {} is: {}".format(k,error/k))


# In[207]:


plt.plot(divisors(len(X)),errors)
plt.title('K vs Error')
plt.xlabel('K')
plt.grid(True)
plt.ylabel('Error')
plt.show()


# In[ ]:




