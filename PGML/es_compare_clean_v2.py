#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:43:00 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K

from scipy.ndimage import gaussian_filter

#from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
    epsilon = 1.0e-6
    
    dx = 2.0*np.pi/np.float64(nx)
    dy = 2.0*np.pi/np.float64(ny)

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[2:nx+2,2:ny+2]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

#%%
# compute the energy spectrum numerically
def energy_spectrum_t(nx,ny,w):
    epsilon = 1.0e-6
    
    dx = 2.0*np.pi/np.float64(nx)
    dy = 2.0*np.pi/np.float64(ny)

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])                    
        en[k] = en[k]/ic
        
    return en, n

#%%
re = 16000.0
ns_dns = 400
ns_fdns = 6000
nt_dns = 6000
dt = 1.0e-3

nxf = 1024
nyf = 1024

nxc = 128
nyc = 128

data = np.load('../w_dns_16000.0_1_1024_0.npz')
wdns = data['wa']
oib_dns = [k for k in range(401)]
wdns= wdns[:,:][:,200:]

data = np.load('../we_les_16000.0_128_1_v2.npz')
wles_dsm = data['ws']

data = np.load('we_les_16000.0_128_2_bs3.npz')
wles_cnn = data['ws']

#%%
data = np.load('we_les_16000.0_128_2_bs3convolvemirror.npz')
wles_cnn_bs = data['ws']


#%%
npx = 64
npy = 64

ne = (nxc+1)*(nyc+1)
me = npx*npy

mean = 0.0
sd2 = 0.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

si2 = 1.0e0
si1 = np.sqrt(si2)

shift = 0.5

xp = np.arange(0,npx)*int(nxc/npx) + int(shift*nxc/npx)
yp = np.arange(0,npy)*int(nyc/npy) + int(shift*nyc/npy)
xprobe, yprobe = np.meshgrid(xp,yp)

oin = []
for i in xp:
    for j in yp:
        n = i*(nxc+1) + j
        oin.append(n)
#        print(n)

roin = np.int32(np.linspace(0,npx*npy-1,npx*npy))
dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

nf = 5         # frequency of observation

ns_start = 2000
ns_end = 6000
nb = int((ns_end-ns_start)/nf) # number of observation time
oib_obs = [ns_start + nf*k for k in range(nb+1)]

z = np.zeros((me,nb+1))
data = np.load('../w_fdns_16000.0_1024_128.npz')
wfdns = data['wac_obs']

for k in range(nb+1):
    wo = np.reshape(wfdns[:,oib_obs[k]],[nxc+5,nyc+5])
    wobs_tr = np.reshape(wo[2:nxc+3,2:nyc+3], [ne])
    z[:,k] = wobs_tr[oin] + np.random.normal(mean,sd1,[me])

ns_fdns = 4000
wfdns = data['wac_obs'][:,ns_start:]
    
#%%
w0 = np.reshape(wdns[:,0], [nxf+5,nyf+5])
wh2 = np.reshape(wdns[:,int(ns_dns/2)], [nxf+5,nyf+5])
wh4 = np.reshape(wdns[:,-1], [nxf+5,nyf+5])
en0_dns, n = energy_spectrum(nxf,nyf,w0)    
enh2_dns, n = energy_spectrum(nxf,nyf,wh2)
enh4_dns, n = energy_spectrum(nxf,nyf,wh4)
kdns = np.linspace(1,n,n)    

#%%
w0 = np.reshape(wfdns[:,0], [nxc+5,nyc+5])
wh2 = np.reshape(wfdns[:,int(ns_fdns/2)], [nxc+5,nyc+5])
wh4 = np.reshape(wfdns[:,-1], [nxc+5,nyc+5])
en0_fdns, n = energy_spectrum(nxc,nyc,w0)    
enh2_fdns, n = energy_spectrum(nxc,nyc,wh2)
enh4_fdns, n = energy_spectrum(nxc,nyc,wh4) 
kfdns = np.linspace(1,n,n)

#%%
ns_les = 4000
w0 = np.reshape(wles_dsm[:,0,0], [nxc+5,nyc+5])
wh2 = np.reshape(wles_dsm[:,0,int(ns_les/2)], [nxc+5,nyc+5])
wh4 = np.reshape(wles_dsm[:,0,-1], [nxc+5,nyc+5])
en0_les_dsm, n = energy_spectrum(nxc,nyc,w0)    
enh2_les_dsm, n = energy_spectrum(nxc,nyc,wh2)
enh4_les_dsm, n = energy_spectrum(nxc,nyc,wh4) 
kles_dsm = np.linspace(1,n,n)

#%%
ns_les = 4000
w0 = np.reshape(wles_cnn[:,0,0], [nxc+5,nyc+5])
wh2 = np.reshape(wles_cnn[:,0,int(ns_les/2)], [nxc+5,nyc+5])
wh4 = np.reshape(wles_cnn[:,0,-1], [nxc+5,nyc+5])
en0_les_cnn, n = energy_spectrum(nxc,nyc,w0)    
enh2_les_cnn, n = energy_spectrum(nxc,nyc,wh2)
enh4_les_cnn, n_les = energy_spectrum(nxc,nyc,wh4) 
kles_dsm = np.linspace(1,n,n)

#%%
ns_les = 4000
w0 = np.reshape(wles_cnn_bs[:,0,0], [nxc+5,nyc+5])
wh2 = np.reshape(wles_cnn_bs[:,0,int(ns_les/2)], [nxc+5,nyc+5])
wh4 = np.reshape(wles_cnn_bs[:,0,-1], [nxc+5,nyc+5])
en0_les_cnn_bs, n = energy_spectrum(nxc,nyc,w0)    
enh2_les_cnn_bs, n = energy_spectrum(nxc,nyc,wh2)
enh4_les_cnn_bs, n_les = energy_spectrum(nxc,nyc,wh4) 
kles_dsm = np.linspace(1,n,n)

#%%
wh2 = np.reshape(z[:,0], [npx,npy])
wh4 = np.reshape(z[:,int(nb/2)], [npx,npy])
wh6 = np.reshape(z[:,-1], [npx,npy])
enh2_obs, n = energy_spectrum_t(npx,npy,wh2)    
enh4_obs, n = energy_spectrum_t(npx,npy,wh4) 
enh6_obs, n = energy_spectrum_t(npx,npy,wh6) 
kobs = np.linspace(1,n,n)



#%%
fig, ax = plt.subplots(1,2,figsize=(12,5))

axs = ax.flat

axs[0].loglog(kdns,enh2_dns[1:], 'b', ls = '-', lw = 2, alpha = 1.0, label = 'DNS')
axs[1].loglog(kdns,enh4_dns[1:], 'b', ls = '-', lw = 2, alpha = 1.0, label = 'DNS')

axs[0].loglog(kles_dsm,enh2_les_dsm[1:], 'y', ls = '-', lw = 2, alpha = 1.0, label = 'DSM')
axs[1].loglog(kles_dsm,enh4_les_dsm[1:], 'y', ls = '-', lw = 2, alpha = 1.0, label = 'DSM')

#axs[0].loglog(kles_dsm,enh2_les_cnn[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'CNN')
#axs[1].loglog(kles_dsm,enh4_les_cnn[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'CNN')

axs[0].loglog(kles_dsm,enh2_les_cnn_bs[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'CNN')
axs[1].loglog(kles_dsm,enh4_les_cnn_bs[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'CNN')


#axs[0].loglog(kobs,enh2_obs[1:],'g', ls = '-', lw = 2, alpha = 1.0, label='Observations')
#axs[1].loglog(kobs,enh4_obs[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'Observations')
#axs[2].loglog(kobs,enh6_obs[1:], 'g', ls = '-', lw = 2, alpha = 1.0, label = 'Observations')

#axs[0].loglog(kenkf,enh_avg_enkf[1:],'r', ls = '-', lw = 2, alpha = 0.9,label='CNN-EnKF')    
#axs[1].loglog(kenkf,ent_avg_enkf[1:], 'r', lw = 2, alpha = 0.9, label='CNN-EnKF')

line = 100*kdns**(-3.0)

for k in range(2):
    axs[k].loglog(kdns,line, 'k--', lw = 2, )
    axs[k].set_xlabel('$K$')
    axs[k].set_ylabel('$E(K)$')
    axs[k].legend(loc=3)
    axs[k].set_ylim(1e-8,1e0)
    axs[k].set_xlim(1.0,350.0)
    axs[k].set_title('$t$ = ' + str(2.0+k*2.0))

fig.tight_layout()
plt.show()
fig.savefig('es_compare3' + '_' + str(npx) + '.png',bbox_inches = 'tight', pad_inches = 0, dpi = 300)
