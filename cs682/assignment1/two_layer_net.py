#!/usr/bin/env python
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

# In[1]:


# A bit of setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
from cs682.classifiers.neural_net import TwoLayerNet


from cs682.data_utils import load_CIFAR10

def get_raw_CIFAR10_data(num_training=4900, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs682/datasets/cifar-10-batches-py'

    return load_CIFAR10(cifar10_dir)

def get_CIFAR10_data(num_training=4900, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """

    X_train, y_train, X_test, y_test = get_raw_CIFAR10_data()

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    print(X_train.shape)
    y=X_train[0]
    #print(y.shape,y.reshape(-1).shape)
    # Reshape data to rows
    #X_train = X_train.reshape(num_training, -1)
    print(X_train.shape)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test



# # Train a network
# To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

# In[23]:


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10

def tonumpy(x):
    return x.numpy()

def transpose(x):
    return x.transpose()

from torchvision import transforms

pipeline=[
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    tonumpy,
    transpose,
]


# In[24]:


from torchvision import transforms

def trans_id(op):
    if hasattr(op,'__name__'):
        return op.__name__
    return repr(op)

def getKey(ops,objName):
    key=""
    for op in ops:
        key+=trans_id(op)+"_"
    key+=objName
    return key

cache={}

def transforms(image_id,image):
    global pipeline, cache
    ops=pipeline
    for i,op in enumerate(ops):
        key = getKey(ops[:i+1],str(image_id))
        if cache!=None:
            if key in cache:
                image=cache[key]
                continue
        image=op(image)
        if cache!=None:
            cache[key] = image
    return image


# In[ ]:

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(49000,1000,1000)
# Invoke the above function to get our data.
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)



import time
start_time=time.time()
# Train the network
batch_size = 200
cache=None
net1 = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net1.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=batch_size,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True,transforms=transforms)
end_time = time.time()
# Predict on the validation set
val_acc = (net1.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
timeWithoutCache=end_time-start_time
print("total time taken without cache: ",end_time-start_time,"seconds")
print("---------------------------------------------------------------------------")
start_time=time.time()
cache={}
net2 = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net2.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=batch_size,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True,transforms=transforms)
end_time = time.time()

# Predict on the validation set
val_acc = (net2.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
timeWithCache=end_time-start_time
print("total time taken with cache: ",end_time-start_time,"seconds")
print("---------------------------------------------------------------------------")

print("speedup seen: ",(timeWithoutCache-timeWithCache)*100/timeWithoutCache,"%")
