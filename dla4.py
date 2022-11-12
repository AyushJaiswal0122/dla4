#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers,losses


# In[3]:


(x_train,__),(x_test,__)=fashion_mnist.load_data()


# In[4]:


len(x_train)


# In[30]:


plt.matshow(x_train[0])


# In[32]:


plt.imshow(x_test[0])


# In[9]:


x_train=x_train/255
x_test=x_test/255


# In[10]:


x_train.shape


# In[12]:


x_test.shape


# In[25]:


latent_dim=64
class Autoencoder(Model):
    def __init__(self,latent_dim):
        super(Autoencoder,self).__init__()
        self.latent_dim=latent_dim
        self.encoder=tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim,activation='relu')
        ])
        self.decoder=tf.keras.Sequential([
            layers.Dense(784,activation='sigmoid'),
            layers.Reshape((28,28))
        ])
    def call(self,x):    
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
autoencoder=Autoencoder(latent_dim)
        
        


# In[26]:


autoencoder.compile(optimizer='adam',loss=losses.MeanSquaredError())
autoencoder.fit(x_train,x_train,epochs=10,shuffle=True,validation_data=(x_test,x_test))


# In[27]:


encoded_imgs=autoencoder.encoder(x_test).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs).numpy()


# In[28]:


n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




