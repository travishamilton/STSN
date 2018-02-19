
# coding: utf-8

# In[1]:


# import autograd functionality
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad   

# import various other libraries
import copy
import matplotlib.pyplot as plt

import pandas as pd

# this is needed to compensate for %matplotl+ib notebook's tendancy to blow up images when plotted inline
from matplotlib import rcParams
rcParams['figure.autolayout'] = True

import pickle


# get_ipython().magic(u'matplotlib notebook')
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[2]:


# gradient descent function
def gradient_descent(g,w,alpha,max_its,beta,version):    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = compute_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))

    # start gradient descent loop
    z = np.zeros((np.shape(w)))      # momentum term
    
    # over the line
    for k in range(max_its): 
        print("Epoch: " +str(k+1) + "\t Training Loss: " + str(g_flat(w)) )
        # plug in value into func and derivative
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        ### normalized or unnormalized descent step? ###
        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
            grad_eval /= grad_norm
            
        # take descent step with momentum
        z = beta*z + grad_eval
        w = w - alpha*z

        # record weight update
        w_hist.append(unflatten(w))


        # saves weight_history to pickle file (so you can use it later)
        if k%10 == 0:
            print w_hist[-1]

    return w_hist


# In[9]:


# create initial weights for arbitrary feedforward network
def scatter_layer_weights(scale):
    
    # loop over desired layer sizes and create appropriately sized initial 
    # weight matrix for each layer (two sets of matrices per layer)
    # get layer sizes for current weight matrix

    w = scale*np.ones(100)
    
    return w


# In[10]:


# create propogation layer
def prop_layer(x):
    out = []
    for j in range(x.shape[0]):
        currout = []
        for i in range(x.shape[1]):
            if i == 0 or i == x.shape[1]-1:
                currout.append(0)
            elif i % 2:
                currout.append(x[j,i+1])
            else:
                currout.append(x[j,i-1])
    
        out.append(currout)
    out = np.asarray(out)
    return out

# create scatter layer
def scatter_layer(x,w):
    out = []
    for j in range(x.shape[0]):
        currout = []
        for i in range(w.shape[0]):
            top = x[j,2*i+1]*w[i] + x[j,2*i]*(1-w[i])
            bottom = x[j,2*i]*w[i] + x[j,2*i+1]*(1 - w[i])
            currout.append(top)
            currout.append(bottom)
        out.append(currout)
    out = np.asarray(out)
    return out


# In[11]:


w_init = scatter_layer_weights(1)
w_init


# In[12]:


X = np.genfromtxt('data/scatter02_T10_all_in.csv', delimiter = ',').T
y = np.genfromtxt('data/scatter02_T10_all_out.csv', delimiter = ',').T
print(X.shape)
print(y.shape)


# In[25]:


# our "predict" function
def propagate(x,w):
    out = scatter_layer(x,w)
    #print("\nScatter: 1")
    for i in range(9):
        out = prop_layer(out)
        out = scatter_layer(out,w)
        #print("Scatter: " + str(i+2))
    out = prop_layer(out)
    return out  #scatter_layer(out,w)

cost = lambda w: np.sum( np.sum( (propagate(X,w) - y)**2, axis = 0) )


# In[27]:


# learning rate
lr = 0.01

# number of epochs
max_its = 1000

# momentum rate
beta = 0.9

# perform gradient descent using cost function
weight_history_1 = gradient_descent(cost,w_init,lr,max_its,beta,version = 'normalized')


