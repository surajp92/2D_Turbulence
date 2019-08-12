#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:35:43 2019

@author: Suraj Pawar
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
n_snapshots  = 50
n_snapshots_test = 10
n_snapshots_train = n_snapshots - n_snapshots_test
for m in range(1,n_snapshots_train):
    file_input = "spectral/Re_4000/uc/uc_"+str(m)+".csv"
    data_input = np.genfromtxt(file_input, delimiter=',')
    
    nx,ny = data_input.shape
    nt = int((nx-2)*(ny-2))
    
    x_t = np.zeros((nt,9))
    y_t = np.zeros((nt,1))
    
    n = 0
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            x_t[n,0] = data_input[i,j]
            x_t[n,1] = data_input[i,j-1]
            x_t[n,2] = data_input[i,j+1]
            x_t[n,3] = data_input[i-1,j]
            x_t[n,4] = data_input[i+1,j]
            x_t[n,5] = data_input[i-1,j-1]
            x_t[n,6] = data_input[i-1,j+1]
            x_t[n,7] = data_input[i+1,j-1]
            x_t[n,8] = data_input[i+1,j+1]
            y_t[n,0] = data_input[i,j]
            n = n+1
    
    if m == 1:
        x_train = x_t
        y_train = y_t
    else:
        x_train = np.vstack((x_train,x_t))
        y_train = np.vstack((y_train,y_t))

for m in range(n_snapshots_train,n_snapshots):
    file_input = "spectral/Re_4000/uc/uc_"+str(m)+".csv"
    data_input = np.genfromtxt(file_input, delimiter=',')
    
    nx,ny = data_input.shape
    nt = int((nx-2)*(ny-2))
    
    x_t = np.zeros((nt,9))
    y_t = np.zeros((nt,1))
    
    n = 0
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            x_t[n,0] = data_input[i,j]
            x_t[n,1] = data_input[i,j-1]
            x_t[n,2] = data_input[i,j+1]
            x_t[n,3] = data_input[i-1,j]
            x_t[n,4] = data_input[i+1,j]
            x_t[n,5] = data_input[i-1,j-1]
            x_t[n,6] = data_input[i-1,j+1]
            x_t[n,7] = data_input[i+1,j-1]
            x_t[n,8] = data_input[i+1,j+1]
            y_t[n,0] = data_input[i,j]
            n = n+1
    
    if m == n_snapshots_train:
        x_test = x_t
        y_test = y_t
    else:
        np.vstack((x_test,x_t))
        np.vstack((y_test,y_t))