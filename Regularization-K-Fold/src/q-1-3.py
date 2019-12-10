#!/usr/bin/env python
# coding: utf-8

# ## Analyse how the hyper-parameter λ plays a role in deciding between bias and variance.

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


testIMG=Image.open("../output_data/lambVSTestError.png")
trainIMG=Image.open("../output_data/lambVSTrainError.png")


# In[5]:


plt.imshow(testIMG)


# In[6]:


plt.figure()
plt.imshow(trainIMG)


#     lambda (regularization factor) helps in reducing overfitting of model on the training data.
#     As it is clear from the graphs above:
#         
#         1. At Lambda=0 Training Error is minimum (Overfitting case: LOW BIAS and HIGH VARIANCE)
#         
#             As we increase the value of lambda we will penalize the weights of some parameters to make our function LESS COMPLEX which makes the increment in TRAINING ERROR and decrement in TESTING ERROR until a suitable lambda is found. Thus increase in BIAS and decrease in VARIANCE is clearly visible.
