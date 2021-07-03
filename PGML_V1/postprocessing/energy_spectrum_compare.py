#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:36:51 2021

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy.integrate import simps
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%% 
# assign parameters
nx = 1024
ny = 1024

nxc = 128
nyc = 128

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

re = 16000
ns = 800
k0 = 10.0

dt = 1e-3
nt = 4000

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')

xc = np.linspace(0.0,2.0*np.pi,nxc+1)
yc = np.linspace(0.0,2.0*np.pi,nyc+1)
xc, yc = np.meshgrid(xc, yc, indexing='ij')

t = np.linspace(0,nt*dt,ns+1)

#%%
data = np.load('./data_16000_1024_1024_Old/results_800.npz')
w = data['w']
en_dns, n_dns = energy_spectrumd(nx,ny,dx,dy,w[2:nx+3,2:ny+3])
k_dns = np.linspace(1,n_dns,n_dns)

data = np.load('./data_16000_fdns_1024_128_S/results_800.npz')
wc = data['wc']
en_fdns, n_fdns = energy_spectrumd(nxc,nyc,dxc,dyc,wc)
k_fdns = np.linspace(1,n_fdns,n_fdns)

data = np.load('./data_16000_fdns_1024_128_G/results_800.npz')
wc_G = data['wc']
en_fdns_G, n_fdns_G = energy_spectrumd(nxc,nyc,dxc,dyc,wc_G)
k_fdns_G = np.linspace(1,n_fdns_G,n_fdns_G)

#data = np.load('./results/data_16000_cnn2_128_128_S2/results_800.npz')
#wles_S = data['w']
#en_les_S, n_les_S = energy_spectrumd(nxc,nyc,dxc,dyc,wles_S[2:nxc+3,2:nyc+3])
#k_les_S = np.linspace(1,n_les_S,n_les_S)
#
#data = np.load('./results/data_16000_cnn2_128_128_G2/results_800.npz')
#wles_G = data['w']
#en_les_G, n_les_G = energy_spectrumd(nxc,nyc,dxc,dyc,wles_G[2:nxc+3,2:nyc+3])
#k_les_G = np.linspace(1,n_les_G,n_les_G)

data = np.load('./data_16000_ann2_128_128/ws_800.npz')
wles_Gd = data['w']
en_les_Gd, n_les_Gd = energy_spectrumd(nxc,nyc,dxc,dyc,wles_Gd[2:nxc+3,2:nyc+3])
k_les_Gd = np.linspace(1,n_les_Gd,n_les_Gd)

data = np.load('./data_16000_ann3_128_128/ws_800.npz')
wles_Gd3 = data['w']
en_les_Gd3, n_les_Gd3 = energy_spectrumd(nxc,nyc,dxc,dyc,wles_Gd3[2:nxc+3,2:nyc+3])
k_les_Gd3 = np.linspace(1,n_les_Gd3,n_les_Gd3)

data = np.load('./data_16000_ann3_128_128_v/ws_800.npz')
wles_Gd3_v = data['w']
en_les_Gd3_v, n_les_Gd3_v = energy_spectrumd(nxc,nyc,dxc,dyc,wles_Gd3_v[2:nxc+3,2:nyc+3])
k_les_Gd3_v = np.linspace(1,n_les_Gd3_v,n_les_Gd3_v)

fig, ax = plt.subplots(1,1,figsize=(6,6))

ax.loglog(k_dns,en_dns[1:], 'k',  lw = 2, label = 'DNS - '+str(nx))
# ax.loglog(k_fdns,en_fdns[1:], 'r',  lw = 2, label = 'fDNS-S '+str(nxc))
ax.loglog(k_fdns_G,en_fdns_G[1:], 'b',  lw = 2, label = 'FDNS-G '+str(nxc))
# ax.loglog(k_les_S,en_les_S[1:], 'g',  lw = 2, label = 'LES-CNN-S '+str(nxc))
#ax.loglog(k_les_G,en_les_G[1:], 'm',  lw = 2, label = 'LES-CNN2-G '+str(nxc))
ax.loglog(k_les_Gd,en_les_Gd[1:], 'y',  lw = 2, label = 'LES-ANN2-G '+str(nxc))
ax.loglog(k_les_Gd3,en_les_Gd3[1:], 'g',  lw = 2, label = 'LES-ANN3-G '+str(nxc))
ax.loglog(k_les_Gd3_v,en_les_Gd3_v[1:], 'm',  lw = 2, label = 'LES-ANN3-G-V '+str(nxc))

ax.set_ylim([1e-12,1e0])   
ax.set_xlim([1e0,1e3])   

ax.set_xlabel('$k$')
ax.set_ylabel('$E(k)$')
ax.legend()
fig.tight_layout()    
plt.show()
#filename = folder_out + '/energy_spectrum_' + str(nx)+'_'+str(nxc) + '.png'
fig.savefig('clipping_average.png', dpi=200)


#%%
plot_tf = 0
if plot_tf:
    nx = 2048
    ny = 2048
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    k = np.linspace(1,n,n)
    
    df = 2*(2.0*np.pi/256)
    en_gaussian = np.exp(-(k**2)*(df**2)/24.0)
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    ax.loglog(k,en_gaussian, 'k--',  lw = 2, label = 'Gaussian ')
    
    ax.set_ylim([1e-12,1e2])   
    ax.legend()    
    plt.show()