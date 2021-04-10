#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:16:04 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from scipy import ndimage

from scipy.ndimage import gaussian_filter
import yaml
import sys
import argparse
from scipy.integrate import simps

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs='?', default="config/input.yaml", help="Config yaml file")
parser.add_argument("tf_version", nargs='?', default=1, type=int, help="Tensorflow version")
parser.add_argument("log", nargs='?', default=0, type=int, help="Write to a log file")
args = parser.parse_args()
config_file = args.config
tf_version = args.tf_version
print_log = args.log

if tf_version == 1:
    from keras.models import Sequential, Model, load_model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from keras.layers import concatenate
    from keras.optimizers import adam
    from keras.utils import plot_model
    from keras import backend as K
    
elif tf_version == 2:
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.optimizers import Adam as adam
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras import backend as K
    
#from utils import *

#font = {'family' : 'Times New Roman',
#        'size'   : 14}    
#plt.rc('font', **font)
#
#import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# fast poisson solver using second-order central difference scheme
# def fps(nx, ny, dx, dy, f):
#     epsilon = 1.0e-6
#     aa = -2.0/(dx*dx) - 2.0/(dy*dy)
#     bb = 2.0/(dx*dx)
#     cc = 2.0/(dy*dy)
#     hx = 2.0*np.pi/np.float64(nx)
#     hy = 2.0*np.pi/np.float64(ny)
    
#     kx = np.empty(nx)
#     ky = np.empty(ny)
    
#     kx[:] = hx*np.float64(np.arange(0, nx))

#     ky[:] = hy*np.float64(np.arange(0, ny))
    
#     kx[0] = epsilon
#     ky[0] = epsilon

#     kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
#     data = np.empty((nx,ny), dtype='complex128')
#     data1 = np.empty((nx,ny), dtype='complex128')
    
#     data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

#     a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
#     b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
#     fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
#     fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
#     e = fft_object(data)
#     e[0,0] = 0.0
#     data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

#     ut = np.real(fft_object_inv(data1))
    
#     #periodicity
#     u = np.empty((nx+5,ny+5)) 
#     u[2:nx+2,2:ny+2] = ut
#     u[:,ny+2] = u[:,2]
#     u[nx+2,:] = u[2,:]
#     u[nx+2,ny+2] = u[2,2]
    
#     return u

def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    
    kx = np.fft.fftfreq(nx, d=dx)*(2.0*np.pi)
    ky = np.fft.fftfreq(ny, d=dx)*(2.0*np.pi)
    
    kx[0] = epsilon
    ky[0] = epsilon
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
        
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    # compute the fourier transform
    e = fft_object(data)
    
    e[0,0] = 0.0
    
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    data1 = e/(-kx**2 - ky**2)
    
    # compute the inverse fourier transform
    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,1] = u[:,ny+1]
    u[:,ny+3] = u[:,3]
    u[:,ny+4] = u[:,4]
    
    u[0,:] = u[nx,:]
    u[1,:] = u[nx+1,:]
    u[nx+3,:] = u[3,:]
    u[nx+4,:] = u[4,:]
    
    return u

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx (size = [nx+1,ny+1])
    uy : du/dy (size = [nx+1,ny+1])
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc

#%%
def gaussian_filter_f(nx,ny,dx,dy,dxc,dyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    epsilon = 1.0e-6
    
    kx = np.fft.fftfreq(nx, d=dx)*(2.0*np.pi)
    ky = np.fft.fftfreq(ny, d=dx)*(2.0*np.pi)
    
    kx[0] = epsilon
    ky[0] = epsilon
    
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    uf = np.fft.fft2(u[0:nx,0:ny])
    
    df = 2.0*dxc
    ufg = np.exp(-(kx**2 + ky**2)*df**2/24) * uf
    
    utc = np.real(np.fft.ifft2(ufg))
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc    

def coarsen(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field along with the size of the data 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : solution field on coarse grid [nxc , nyc]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
    
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc [0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]     
    ufc [int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)] 
    ufc [0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):] 
    ufc [int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    utc = np.real(np.fft.ifft2(ufc))
    
    uc = np.zeros((nxc+1,nyc+1))
    uc[0:nxc,0:nyc] = utc
    uc[:,nyc] = uc[:,0]
    uc[nxc,:] = uc[0,:]
    uc[nxc,nyc] = uc[0,0]
        
    return uc

def gaussian_coarsen(nx,ny,nxc,nyc,dx,dy,dxc,dyc,u):
    ug = gaussian_filter_f(nx,ny,dx,dy,dxc,dyc,u)
    ugc = coarsen(nx,ny,nxc,nyc,ug) # change to [2:nx+3,2:ny+3] afterwards 
    return ugc

#%%  
def dyn_smag(nx,ny,kappa,sc,wc):
    '''
    compute the eddy viscosity using Germanos dynamics procedure with Lilys 
    least square approximation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kapppa : sub-filter grid filter ratio
    wc : vorticity on LES grid
    sc : streamfunction on LES grid
    
    Output
    ------
    ev : (cs*delta)**2*|S| (size = [nx+1,ny+1])
    '''
    
    nxc = int(nx/kappa) 
    nyc = int(ny/kappa)
    
    scc = les_filter(nx,ny,nxc,nyc,sc[2:nx+3,2:ny+3])
    wcc = les_filter(nx,ny,nxc,nyc,wc[2:nx+3,2:ny+3])
    
    scx,scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
    wcx,wcy = grad_spectral(nx,ny,wc[2:nx+3,2:ny+3])
    
    wcxx,wcxy = grad_spectral(nx,ny,wcx)
    wcyx,wcyy = grad_spectral(nx,ny,wcy)
    
    scxx,scxy = grad_spectral(nx,ny,scx)
    scyx,scyy = grad_spectral(nx,ny,scy)
    
    dac = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2) # |\bar(s)|
    dacc = les_filter(nx,ny,nxc,nyc,dac)        # |\tilde{\bar{s}}| = \tilde{|\bar(s)|}
    
    sccx,sccy = grad_spectral(nx,ny,scc)
    wccx,wccy = grad_spectral(nx,ny,wcc)
    
    wccxx,wccxy = grad_spectral(nx,ny,wccx)
    wccyx,wccyy = grad_spectral(nx,ny,wccy)
    
    scy_wcx = scy*wcx
    scx_wcy = scx*wcy
    
    scy_wcx_c = les_filter(nx,ny,nxc,nyc,scy_wcx)
    scx_wcy_c = les_filter(nx,ny,nxc,nyc,scx_wcy)
    
    h = (sccy*wccx - sccx*wccy) - (scy_wcx_c - scx_wcy_c)
    
    t = dac*(wcxx + wcyy)
    tc = les_filter(nx,ny,nxc,nyc,t)
    
    m = kappa**2*dacc*(wccxx + wccyy) - tc
    
    hm = h*m
    mm = m*m
    
    CS2 = (np.sum(0.5*(hm + abs(hm)))/np.sum(mm))
    
    ev = CS2*dac
    
    return ev

#%%
def stat_smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[1:nx+2,1:ny+2] + s[3:nx+4,3:ny+4] \
                                             -s[3:nx+4,1:ny+2] - s[1:nx+2,3:ny+4])
    
    dsdxx = (1.0/(dx*dx))*(s[3:nx+4,2:ny+3] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[1:nx+2,2:ny+3])
    
    dsdyy = (1.0/(dy*dy))*(s[2:nx+3,3:ny+4] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[2:nx+3,1:ny+2])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev    

#%%
def dnn_closure(nx,ny,w,s,max_min,model,ifeat):
    wx,wy = grad_spectral(nx,ny,w[2:nx+3,2:ny+3])
    wxx,wxy = grad_spectral(nx,ny,wx)
    wyx,wyy = grad_spectral(nx,ny,wy)
    
    sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
    sxx,sxy = grad_spectral(nx,ny,sx)
    syx,syy = grad_spectral(nx,ny,sy)
    
    kernel_w = np.sqrt(wx**2 + wy**2)
    kernel_s = np.sqrt(4.0*sxy**2 + (sxx - syy)**2)
        
    wc = (2.0*w[2:nx+3,2:ny+3] - (max_min[0,0] + max_min[0,1]))/(max_min[0,0] - max_min[0,1])
    sc = (2.0*s[2:nx+3,2:ny+3] - (max_min[1,0] + max_min[1,1]))/(max_min[1,0] - max_min[1,1])
    kwc = (2.0*kernel_w - (max_min[2,0] + max_min[2,1]))/(max_min[2,0] - max_min[2,1])
    ksc = (2.0*kernel_s - (max_min[3,0] + max_min[3,1]))/(max_min[3,0] - max_min[3,1])
    
    if ifeat == 3:
        wcx = (2.0*wx - (max_min[4,0] + max_min[4,1]))/(max_min[4,0] - max_min[4,1])
        wcy = (2.0*wy - (max_min[5,0] + max_min[5,1]))/(max_min[5,0] - max_min[5,1])
        wcxx = (2.0*wxx - (max_min[6,0] + max_min[6,1]))/(max_min[6,0] - max_min[6,1])
        wcyy = (2.0*wyy - (max_min[7,0] + max_min[7,1]))/(max_min[7,0] - max_min[7,1])
        wcxy = (2.0*wxy - (max_min[8,0] + max_min[8,1]))/(max_min[8,0] - max_min[8,1])
        
        scx = (2.0*sx - (max_min[9,0] + max_min[9,1]))/(max_min[9,0] - max_min[9,1])
        scy = (2.0*sy - (max_min[10,0] + max_min[10,1]))/(max_min[10,0] - max_min[10,1])
        scxx = (2.0*sxx - (max_min[11,0] + max_min[11,1]))/(max_min[11,0] - max_min[11,1])
        scyy = (2.0*syy - (max_min[12,0] + max_min[12,1]))/(max_min[12,0] - max_min[12,1])
        scxy = (2.0*sxy - (max_min[13,0] + max_min[13,1]))/(max_min[13,0] - max_min[13,1])

    if ifeat == 1:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,18))
        n = 0
        for i in range(2,nx+3):
            for j in range(2,ny+3):
                x_test[n,0:9] = w[i-1:i+2,j-1:j+2].flatten()
                x_test[n,9:18] = s[i-1:i+2,j-1:j+2].flatten()
                n = n+1         
    if ifeat == 2:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,20))
        n = 0
        kw = np.zeros((nx+5,ny+5))
        ks = np.zeros((nx+5,ny+5))
        kw[2:nx+3,2:ny+3] = kwc
        ks[2:nx+3,2:ny+3] = ksc
        for i in range(2,nx+3):
            for j in range(2,ny+3):
                x_test[n,0:9] = w[i-1:i+2,j-1:j+2].flatten()
                x_test[n,9:18] = s[i-1:i+2,j-1:j+2].flatten()
                x_test[n,18] = kw[i,j]
                x_test[n,19] = ks[i,j]
    if ifeat == 3:
        nt = int((nx+1)*(ny+1))
        x_test = np.zeros((nt,12))
        
        x_test[:,0] = wc.flatten()
        x_test[:,1] = sc.flatten()
        x_test[:,2] = wcx.flatten()
        x_test[:,3] = wcy.flatten()
        x_test[:,4] = wcxx.flatten()
        x_test[:,5] = wcyy.flatten()
        x_test[:,6] = wcxy.flatten()
        x_test[:,7] = scx.flatten()
        x_test[:,8] = scy.flatten()
        x_test[:,9] = scxx.flatten()
        x_test[:,10] = scyy.flatten()
        x_test[:,11] = scxy.flatten()
                
    y_pred_sc = model.predict(x_test)
    y_pred = 0.5*(y_pred_sc*(max_min[14,0] - max_min[14,1]) + (max_min[14,0] + max_min[14,1]))
    
    y_pred = np.reshape(y_pred,[nx+1,ny+1])
    
    return y_pred  

#%%
def cnn_closure(nx,ny,w,s,max_min,model,ifeat):
    wx,wy = grad_spectral(nx,ny,w[2:nx+3,2:ny+3])
    wxx,wxy = grad_spectral(nx,ny,wx)
    wyx,wyy = grad_spectral(nx,ny,wy)
    
    sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
    sxx,sxy = grad_spectral(nx,ny,sx)
    syx,syy = grad_spectral(nx,ny,sy)
    
    kernel_w = np.sqrt(wx**2 + wy**2)
    kernel_s = np.sqrt(4.0*sxy**2 + (sxx - syy)**2)
    
    wc = (2.0*w[2:nx+3,2:ny+3] - (max_min[0,0] + max_min[0,1]))/(max_min[0,0] - max_min[0,1])
    sc = (2.0*s[2:nx+3,2:ny+3] - (max_min[1,0] + max_min[1,1]))/(max_min[1,0] - max_min[1,1])
    kwc = (2.0*kernel_w - (max_min[2,0] + max_min[2,1]))/(max_min[2,0] - max_min[2,1])
    ksc = (2.0*kernel_s - (max_min[3,0] + max_min[3,1]))/(max_min[3,0] - max_min[3,1])
    
    wcx = (2.0*wx - (max_min[4,0] + max_min[4,1]))/(max_min[4,0] - max_min[4,1])
    wcy = (2.0*wy - (max_min[5,0] + max_min[5,1]))/(max_min[5,0] - max_min[5,1])
    wcxx = (2.0*wxx - (max_min[6,0] + max_min[6,1]))/(max_min[6,0] - max_min[6,1])
    wcyy = (2.0*wyy - (max_min[7,0] + max_min[7,1]))/(max_min[7,0] - max_min[7,1])
    wcxy = (2.0*wxy - (max_min[8,0] + max_min[8,1]))/(max_min[8,0] - max_min[8,1])
    
    scx = (2.0*sx - (max_min[9,0] + max_min[9,1]))/(max_min[9,0] - max_min[9,1])
    scy = (2.0*sy - (max_min[10,0] + max_min[10,1]))/(max_min[10,0] - max_min[10,1])
    scxx = (2.0*sxx - (max_min[11,0] + max_min[11,1]))/(max_min[11,0] - max_min[11,1])
    scyy = (2.0*syy - (max_min[12,0] + max_min[12,1]))/(max_min[12,0] - max_min[12,1])
    scxy = (2.0*sxy - (max_min[13,0] + max_min[13,1]))/(max_min[13,0] - max_min[13,1])

    if ifeat == 1:
        x_test = np.zeros((1,nx+1,ny+1,2))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
    if ifeat == 2:
        x_test = np.zeros((1,nx+1,ny+1,4))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = kwc
        x_test[0,:,:,3] = ksc
    if ifeat == 3:
        x_test = np.zeros((1,nx+1,ny+1,12))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = wcx
        x_test[0,:,:,3] = wcy
        x_test[0,:,:,4] = wcxx
        x_test[0,:,:,5] = wcyy
        x_test[0,:,:,6] = wcxy
        x_test[0,:,:,7] = scx
        x_test[0,:,:,8] = scy
        x_test[0,:,:,9] = scxx
        x_test[0,:,:,10] = scyy
        x_test[0,:,:,11] = scxy
    
    x_test_f = x_test[:,:,:,:2]
    x_test_k = x_test[:,:,:,2:]
    y_pred_sc = model.predict(x = [x_test_f,x_test_k])
    y_pred = 0.5*(y_pred_sc[0,:,:,0]*(max_min[14,0] - max_min[14,1]) + (max_min[14,0] + max_min[14,1]))
    
    return y_pred

#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+5,ny+5))
    
    #Arakawa    
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
    
    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
    
    if ifm == 0:
        f[2:nx+3,2:ny+3] = -jac + lap/re 
        
    elif ifm == 1:
        ev = dyn_smag(nx,ny,kappa,s,w)
        f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap
    
    elif ifm == 2:
        kconvolve = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        pi_source = cnn_closure(nx,ny,w,s,max_min,model,ifeat)
        nue = pi_source/lap
        
        nue_p = np.where(nue > 0, nue, 0.0)
        
#        nue_loc_avg = ndimage.generic_filter(nue_p, np.nanmean, size=3, mode='constant', cval=np.NaN)
#        nue_loc_avg = ndimage.generic_filter(nue_p, np.mean, size=3, mode='constant', cval=0.0)
        nue_loc_avg = ndimage.convolve(nue_p, kconvolve, mode='mirror')#, cval=0.0)
        nue_loc_avg = nue_loc_avg/9.0
        
#        mask1 = nue_p < nue_loc_avg
#        nue_loc_avg_use = np.where(mask1[:,:] == True, nue_p, 0.0)
        nue_loc_avg_use = np.where(nue_p < nue_loc_avg, nue_p, 0.0)
        
#        mask2 = nue_loc_avg_use > 0.0
#        pi_source = np.where(mask2[:,:] == True, pi_source[:,:], 0.0)
        pi_source = np.where(nue_loc_avg_use > 0.0, pi_source[:,:],0.0)
        
        f[2:nx+3,2:ny+3] = -jac + lap/re + pi_source
    
    elif ifm == 3:
        kconvolve = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        pi_source = dnn_closure(nx,ny,w,s,max_min,model,ifeat)
        nue = pi_source/lap
        
        nue_p = np.where(nue > 0, nue, 0.0)
        
        nue_loc_avg = ndimage.convolve(nue_p, kconvolve, mode='mirror')#, cval=0.0)
        nue_loc_avg = nue_loc_avg/9.0
        
        nue_loc_avg_use = np.where(nue_p < nue_loc_avg, nue_p, 0.0)
        
        pi_source = np.where(nue_loc_avg_use > 0.0, pi_source[:,:],0.0)
        
        f[2:nx+3,2:ny+3] = -jac + lap/re + pi_source
        
    return f

def jacobian(nx,ny,dx,dy,re,w,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    # Arakawa
    j1 = gg*((w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))

    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    return jac

#%%
# set initial condition for decay of turbulence problem
def ic_decay(nx,ny,dx,dy):
    #w = np.empty((nx+3,ny+3))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #w = np.zeros((nx+3,ny+3))
    
    #periodicity
    w = np.zeros((nx+5,ny+5)) 
    w[2:nx+2,2:ny+2] = ut
    w[:,ny+2] = w[:,2]
    w[nx+2,:] = w[2,:]
    w[nx+2,ny+2] = w[2,2] 
    
    w = bc(nx,ny,w)    
    
    return w

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,dx,dy,w):
    epsilon = 1.0e-6

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
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%%
def tdma(a,b,c,r,s,e):
    
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    r_ = np.copy(r)
    
    un = np.zeros((np.shape(r)[0],np.shape(r)[1]))
    
    for i in range(s+1,e+1):
        b_[i,:] = b_[i,:] - a_[i,:]*(c_[i-1,:]/b_[i-1,:])
        r_[i,:] = r_[i,:] - a_[i,:]*(r_[i-1,:]/b_[i-1,:])
        
    un[e,:] = r_[e,:]/b_[e,:]
    
    for i in range(e-1,s-1,-1):
        un[i,:] = (r_[i,:] - c_[i,:]*un[i+1,:])/b_[i,:]
    
    del a_, b_, c_, r_
    
    return un

def tdmsv(a,b,c,r,s,e,n):
    gam = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    bet = np.zeros((1,n+1))
    
    bet[0,:] = b[s,:]
    u[s,:] = r[s,:]/bet[0,:]
    
    for i in range(s+1,e+1):
        gam[i,:] = c[i-1,:]/bet[0,:]
        bet[0,:] = b[i,:] - a[i,:]*gam[i,:]
        u[i,:] = (r[i,:] - a[i,:]*u[i-1,:])/bet[0,:]
    
    for i in range(e-1,s-1,-1):
        u[i,:] = u[i,:] - gam[i+1,:]*u[i+1,:]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdmsv(a,b,c,alpha,beta,r,s,e,n):
    bb = np.zeros((e+1,n+1))
    u = np.zeros((e+1,n+1))
    gamma = np.zeros((1,n+1))
    
    gamma[0,:] = -b[s,:]
    bb[s,:] = b[s,:] - gamma[0,:]
    bb[e,:] = b[e,:] - alpha*beta/gamma[0,:]
    bb[s+1:e,:] = b[s+1:e,:]
    
    x = tdmsv(a,bb,c,r,s,e,n)
    
    u[s,:] = gamma[0,:]
    u[e,:] = alpha[0,:]
    
    z = tdmsv(a,bb,c,u,s,e,n)
    
    fact = (x[s,:] + beta[0,:]*x[e,:]/gamma[0,:])/(1.0 + z[s,:] + beta[0,:]*z[e,:]/gamma[0,:])    
    x[s:e+1,:] = x[s:e+1,:] - fact*z[s:e+1,:]
        
    return x


def c4d_p(f,dx,dy,nx,ny,isign):
    
    if isign == 'X':
        u = np.copy(f)
        h = dx
    if isign == 'Y':
        u = np.copy(f.T)
        h = dy
        temp = nx
        nx = ny
        ny = temp
    
    a = np.zeros((nx,ny+1))
    b = np.zeros((nx,ny+1))
    c = np.zeros((nx,ny+1))
    r = np.zeros((nx,ny+1))

    ii = np.arange(0,nx)
    up = u[ii,:]
    a[ii,:] = 1.0/4.0
    b[ii,:] = 1.0
    c[ii,:] = 1.0/4.0
    r[ii,:] = 3.0*(up[(ii+1)%nx,:] - up[ii-1,:])/(4.0*h)
    
    start = 0
    end = nx
        
    alpha = np.zeros((1,ny+1))
    beta = np.zeros((1,ny+1))
    
    alpha[0,:] = 1.0/4.0
    beta[0,:] = 1.0/4.0
    
    x = ctdmsv(a,b,c,alpha,beta,r,start,end-1,ny)
    
    ud = np.zeros((nx+1,ny+1))
    ud[0:nx,:] = x[0:nx,:]
    ud[nx,:] = ud[0,:]

    if isign == 'X':
        fd = np.copy(ud)
    if isign == 'Y':
        fd = np.copy(ud.T)
        
    return fd

def compute_derivatives(f,dx,dy,nx,ny):
    fx = c4d_p(f,dx,dy,nx,ny,'X')
    fy = c4d_p(f,dx,dy,nx,ny,'Y')
    
    return fx, fy


def compute_history(nx,ny,dx,dy,x,y,lx,ly,re,s,w):
    
    sx = c4d_p(s,dx,dy,nx,ny,'X')
    sy = c4d_p(s,dx,dy,nx,ny,'Y')
     
    u = sy
    v = -sx
        
    # compute total energy
    en = 0.5*(u**2 + v**2)
    ene = simps(simps(en, y), x)/(lx*ly)
    
    # compute total enstrophy
    en = 0.5*(w**2)
    ens = simps(simps(en, y), x)/(lx*ly)
    
    # dissipation 
    dis = (2.0/re)*ens
    return ene, ens, dis

def store_apriori_data(nx,ny,nxc,nyc,dx,dy,dxc,dyc,w,s,filename):
    wc = np.zeros((nxc+5,nyc+5))
    sc = np.zeros((nxc+5,nyc+5))
    
    wc[2:nxc+3,2:nyc+3] = gaussian_coarsen(nx,ny,nxc,nyc,dx,dy,dxc,dyc,w[2:nx+3,2:ny+3]) 
    wc = bc(nxc,nyc,wc)
    
    sc[:,:] = fps(nxc,nyc,dxc,dyc,-wc)
    sc = bc(nxc,nyc,sc)
    
    jac_f = jacobian(nx,ny,dx,dy,re,w,s)                    # Jacobian on fine resolution
    jac_ff = gaussian_coarsen(nx,ny,nxc,nyc,dx,dy,dxc,dyc,jac_f)    # Filtered Jacobian on fine resolution
    jac_c = jacobian(nxc,nyc,dxc,dyc,re,wc,sc)              # Jacobian on coarse resolution
    
    pi_source = jac_c - jac_ff
    
    wcx,wcy = compute_derivatives(wc[2:nxc+3,2:nyc+3],dxc,dyc,nxc,nyc)
    wcxx,wcxy = compute_derivatives(wcx,dxc,dyc,nxc,nyc)
    wcyx,wcyy = compute_derivatives(wcy,dxc,dyc,nxc,nyc)
    scx,scy = compute_derivatives(sc[2:nxc+3,2:nyc+3],dxc,dyc,nxc,nyc)
    scxx,scxy = compute_derivatives(scx,dxc,dyc,nxc,nyc)
    scyx,scyy = compute_derivatives(scy,dxc,dyc,nxc,nyc)
    
    kernel_w = np.sqrt(wcx**2 + wcy**2)
    kernel_s = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2)
    
    lap = wcxx + wcyy
    
    nu_e = pi_source/lap
    
    np.savez(filename,wc=wc[2:nxc+3,2:nyc+3],sc=sc[2:nxc+3,2:nyc+3],
             nue=nu_e,lap=lap,pi=pi_source,ks=kernel_s,kw=kernel_w,
             wcx=wcx,wcy=wcy,wcxx=wcxx,wcyy=wcyy,wcxy=wcxy,
             scx=scx,scy=scy,scxx=scxx,scyy=scyy,scxy=scxy)
        
def plot_turbulent_parameters(time,ene,ens,dis,filename):
    fig, ax = plt.subplots(2,2,figsize=(8,6))
    axs = ax.flat
    axs[0].plot(time, ene, label='Energy')
    axs[1].plot(time, ens, label='Enstrophy')
    axs[2].plot(time, dis, label='Dissipation')
    
    for i in range(3):
        axs[i].legend()

    # plt.show()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)    


def plot_contour(X,Y,w,s,filename):
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    
    cs = axs[0].contourf(X,Y,w,120,cmap='RdBu_r', alpha=1.0,)
    fig.colorbar(cs, ax=axs[0], shrink=0.8, orientation='vertical')
    axs[0].set_aspect('equal')
    
    cs = axs[1].contourf(X,Y,s,120,cmap='RdBu_r', alpha=1.0,)
    fig.colorbar(cs, ax=axs[1], shrink=0.8, orientation='vertical')
    axs[1].set_aspect('equal')
        
    # plt.show()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)
    
#%%   
with open(config_file) as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
    
file.close()        

nx = input_data['nx']
ny = input_data['ny']
nxc = input_data['nxc']
nyc = input_data['nyc']
re = float(input_data['re'])
nt = input_data['nt']
dt = input_data['dt']
isolver = input_data['isolver']
icompact = input_data['icompact'] 
ifm = input_data['ifm']
ipr = input_data['ipr']
ip = input_data['ip']
its = input_data['its']
pfreq = input_data['pfreq']
sfreq = input_data['sfreq']
nsmovie = input_data['nsmovie']
esplot = input_data['esplot']
kappa = input_data['kappa']
pCU3 = input_data['pCU3']
ichkp = input_data['ichkp']
nschkp = input_data['nschkp']
iapriori = input_data['iapriori']
iaprsfreq = input_data['iaprsfreq']
imovie = input_data['imovie']
seedn = input_data['seedn']

seed(seedn)

model_dict = {}
model_dict[str(0)] = 'DNS'
model_dict[str(1)] = 'DSM'
model_dict[str(2)] = 'CNN'

directory = f'KT_{model_dict[str(ifm)]}'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(directory, f'solution_{nx}_{nxc}_{re:0.2e}_{seedn}')
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory_movie = os.path.join(directory, f'movie')
if not os.path.exists(directory_movie):
    os.makedirs(directory_movie)

directory_save = os.path.join(directory, f'save')
if not os.path.exists(directory_save):
    os.makedirs(directory_save)

directory_apriori = os.path.join(directory, f'apriori')
if not os.path.exists(directory_apriori):
    os.makedirs(directory_apriori)

     
filename = os.path.join(directory, f"kt_stats_{nx}_{ny}_{re:0.2e}.txt")
fstats = open(filename,"w+")


ifeat = 2

if ifm == 0:
    model = None
    max_min = None
else:
    model = load_model(f'./CNN/nn_history/TF{tf_version}_{nx}/CNN_model_{ifeat}',
                           custom_objects={'coeff_determination': coeff_determination})
        
    max_min = np.load(f'./CNN/nn_history/TF{tf_version}_{nx}/scaling.npy')

#%% 
# DA parameters
npe = 1

# assign parameters
pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

xc = np.linspace(0.0,2.0*np.pi,nxc+1)
yc = np.linspace(0.0,2.0*np.pi,nyc+1)
Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

#%% 
# allocate the vorticity and streamfunction arrays
wen = np.zeros((nx+5,ny+5,npe)) 
sen = np.zeros((nx+5,ny+5,npe))

ten = np.zeros((nx+5,ny+5,npe))
ren = np.zeros((nx+5,ny+5,npe))
time = np.zeros(nt+1)

for n in range(npe):
    if ichkp == 0:
        w0 = ic_decay(nx,ny,dx,dy)
        ks = 0
        time[ks] = ks*dt
        filename = os.path.join(directory, f"kt_res_{nx}_{ny}_{re:0.2e}_{ks}.txt")
        log = open(filename, "w")
    elif ichkp == 1:
        filename = os.path.join(directory_save, f'ws_{nschkp}.npz')
        data = np.load(filename)
        w0 = data['w']
        ks = nschkp 
        time[ks] = ks*dt
        filename = os.path.join(directory, f"kt_res_{nx}_{ny}_{re:0.2e}_{ks}.txt")
        log = open(filename, "w")
        
    wen[:,:,n] = w0 
    sen[:,:,n] = fps(nx, ny, dx, dy, -wen[:,:,n])
    sen[:,:,n] = bc(nx,ny,sen[:,:,n])

if print_log:
    sys.stdout = log
    
mfreq = int(nt/nsmovie)
km = input_data['mvchkp'] + 1

if (ks % pfreq == 0):
    print('%0.5i %0.3f %0.3f %0.3f' % (ks, time[ks], np.max(wen[:,:,n]), np.min(wen[:,:,n])))

n = 0                    
if ks == 0:
    filename = os.path.join(directory_save, f'ws_{ks}.npz')
    np.savez(filename,w = wen[:,:,n], s = sen[:,:,n])
    
    filename = os.path.join(directory_apriori, f'ws_{ks}.npz')
    store_apriori_data(nx,ny,nxc,nyc,dx,dy,dxc,dyc,wen[:,:,n],sen[:,:,n],filename)

if ks == 0:
    filename = os.path.join(directory_movie, f'ws_{km}.npz')
    np.savez(filename, w = wen[:,:,n], s = sen[:,:,n]) 
    km = km + 1
            
#%%
def rhs(nx,ny,dx,dy,re,pCU3,w,s,ifm,isolver,max_min,model,ifeat):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat)
    
# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

for k in range(ks+1,nt+1):
    time[k] = time[k-1] + dt
    for n in range(npe):
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,wen[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
        
        #stage-1
        ten[2:nx+3,2:ny+3,n] = wen[2:nx+3,2:ny+3,n] + dt*ren[2:nx+3,2:ny+3,n]
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,ten[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
    
        #stage-2
        ten[2:nx+3,2:ny+3,n] = 0.75*wen[2:nx+3,2:ny+3,n] + 0.25*ten[2:nx+3,2:ny+3,n] + 0.25*dt*ren[2:nx+3,2:ny+3,n]
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,ten[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
    
        #stage-3
        wen[2:nx+3,2:ny+3,n] = aa*wen[2:nx+3,2:ny+3,n] + bb*ten[2:nx+3,2:ny+3,n] + bb*dt*ren[2:nx+3,2:ny+3,n]
        wen[:,:,n] = bc(nx,ny,wen[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -wen[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        if (k % pfreq == 0):
            print('%0.5i %0.3f %0.3f %0.3f' % (k, time[k], np.max(wen[:,:,n]), np.min(wen[:,:,n])))
            sys.stdout.flush()
                    
        if k % sfreq == 0:
            filename = os.path.join(directory_save, f'ws_{k}.npz')
            np.savez(filename,w = wen[:,:,n], s = sen[:,:,n])
        
        if imovie == 1:
            if k % mfreq == 0:
                filename = os.path.join(directory_movie, f'ws_{km}.npz')
                np.savez(filename, w = wen[:,:,n]) 
                km = km + 1
        
        if k % iaprsfreq == 0:
            if iapriori == 1:
                filename = os.path.join(directory_apriori, f'ws_{k}.npz')
                store_apriori_data(nx,ny,nxc,nyc,dx,dy,dxc,dyc,wen[:,:,n],sen[:,:,n],filename)
        
    
total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)
input_data['cpu_time'] = total_clock_time

filename = os.path.join(directory, f'cpu_time.yaml')
with open(filename, 'w') as outfile:
    yaml.dump(input_data, outfile, default_flow_style=False)
    
#%%
ns = int(nt/sfreq)
ene, ens, dis, time = [np.zeros(ns+1) for i in range(4)]

for k in range(ns+1):
    filename = os.path.join(directory_save, f'ws_{k*sfreq}.npz')
    data = np.load(filename)
    w = data['w']
    s = data['s']
    ene[k], ens[k], dis[k] = compute_history(nx,ny,dx,dy,x,y,lx,ly,re,
                                         s[2:nx+3,2:ny+3],w[2:nx+3,2:ny+3])
    time[k] = k*dt*sfreq

#%%    
filename = os.path.join(directory, f'contour_{nx}_{ny}_{re:0.2e}.png')
plot_contour(X,Y,wen[2:nx+3,2:ny+3,n],sen[2:nx+3,2:ny+3,n],filename)

filename = os.path.join(directory, f'turb_params_{nx}_{ny}_{re:0.2e}.png')
plot_turbulent_parameters(time,ene,ens,dis,filename)

filename = os.path.join(directory, f'statistics_{nx}_{ny}_{re:0.2e}.npz')
np.savez(filename, time=time, ene=ene, ens=ens, dis=dis)  

#%%
wes_p = np.zeros((nx+5,ny+5,esplot+1))
wes_p_c = np.zeros((nxc+5,nyc+5,esplot+1))
espfreq = int(nt/esplot)
for j in range(esplot+1):
    filename = os.path.join(directory_save, f'ws_{j*espfreq}.npz')
    data = np.load(filename)
    wes_p[:,:,j] =  data['w']
    
    filename = os.path.join(directory_apriori, f'ws_{j*espfreq}.npz')
    data = np.load(filename)
    wes_p_c[2:nxc+3,2:nyc+3,j] =  data['wc']
    wes_p_c[:,:,j] = bc(nxc,nyc,wes_p_c[:,:,j])

#%%    
if (ipr == 4):

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    for j in range(esplot+1):
        en, n = energy_spectrum(nx,ny,dx,dy,wes_p[:,:,j])  
        k = np.linspace(1,n,n)
        if j == 0:
            ax[0].loglog(k,en[1:], 'k', lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$')
        else:
            ax[0].loglog(k,en[1:], lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$')
    
    kl = np.linspace(8,int(n/2),int(n/2)-7)
    line = 100*kl**(-3.0)
    
    ax[0].loglog(kl,line, 'k--', lw = 2, )
    ax[0].text(0.65, 0.95, '$k^{-3}$', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')
    
    for j in range(esplot+1):
        en, n = energy_spectrum(nxc,nyc,dxc,dyc,wes_p_c[:,:,j])  
        k = np.linspace(1,n,n)
        if j == 0:
            ax[1].loglog(k,en[1:], 'k', lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$')
        else:
            ax[1].loglog(k,en[1:], lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$')
            
    for i in range(2):
        ax[i].set_xlabel('$K$')
        ax[i].set_ylabel('$E(K)$')
        ax[i].legend(loc=3)
        ax[i].set_ylim(1e-12,1e0)
        ax[i].set_xlim(1e0,1e3)
        ax[i].set_title('Energy spectrum')
        
    plt.show()
    filename = os.path.join(directory, f'es_{nx}_{ny}_{re:0.2e}.png')
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0, dpi = 300)

    fig, ax = plt.subplots(2,2,figsize=(12,12))
    axs = ax.flat
    for j in range(1,esplot):
        en, n = energy_spectrum(nx,ny,dx,dy,wes_p[:,:,j])  
        k = np.linspace(1,n,n)
        axs[j-1].loglog(k,en[1:], lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$ DNS')
        
        en, n = energy_spectrum(nxc,nyc,dxc,dyc,wes_p_c[:,:,j])  
        k = np.linspace(1,n,n)
        axs[j-1].loglog(k,en[1:], lw = 2, alpha = 1.0, label=f'$t={j*dt*espfreq}$ FDNS')
        
        axs[j-1].set_xlabel('$K$')
        axs[j-1].set_ylabel('$E(K)$')
        axs[j-1].legend(loc=3)
        axs[j-1].set_ylim(1e-12,1e0)
        axs[j-1].set_xlim(1e0,1e3)
    plt.show()
    filename = os.path.join(directory, f'es_filtered_{nx}_{nxc}_{re:0.2e}.png')
    fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0, dpi = 300)