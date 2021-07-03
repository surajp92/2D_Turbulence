#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:16:29 2021

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt

nx =  1024
nxc = 128
re = 8000

data= np.load(f'./KT_DNS/solution_{nx}_{nxc}_{re:0.2e}_10/save/ws_0.npz')
w10 = data['w']

data= np.load(f'./KT_DNS/solution_{nx}_{nxc}_{re:0.2e}_20/save/ws_0.npz')
w20 = data['w']


#%%
fig, ax = plt.subplots(1,2, figsize=(12,5))
cs = ax[0].contourf(w10, 120, cmap='jet')
fig.colorbar(cs, ax=ax[0])
cs = ax[1].contourf(w20, 120, cmap='jet')
fig.colorbar(cs, ax=ax[1])
plt.show()


#%%
data= np.load(f'./KT_DNS/solution_{nx}_{nxc}_{re:0.2e}_10/apriori/ws_0.npz')
w10c = data['wc']

data= np.load(f'./KT_DNS/solution_{nx}_{nxc}_{re:0.2e}_20/apriori/ws_0.npz')
w20c = data['wc']

fig, ax = plt.subplots(1,2, figsize=(12,5))
cs = ax[0].contourf(w10c, 120, cmap='jet')
fig.colorbar(cs, ax=ax[0])
cs = ax[1].contourf(w20c, 120, cmap='jet')
fig.colorbar(cs, ax=ax[1])
plt.show()

#%%
ns = 2000
pi_avg = np.zeros(ns+1)
for i in range(ns+1):
    data= np.load(f'./KT_DNS/solution_{nx}_{nxc}_{re:0.2e}/apriori/ws_{i}.npz')
    pi_avg[i] = np.average(data['pi'])
    
#%%
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.plot(pi_avg)
ax.set_ylim([-0.05, 0.05])
plt.show()    