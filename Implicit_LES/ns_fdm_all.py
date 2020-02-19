#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:08:41 2019

@author: Suraj Pawar

Two-dimensional navier-stokes solver  
Vorticity-stream function formulation
Arakawa scheme (or compact scheme or explicit) for nonlinear term
3rd order Runge-Kutta for temporal discritization
Periodic boundary conditions only

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

from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

#%%
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using Thomas algorithm
#-----------------------------------------------------------------------------#
def tdms(a,b,c,r,s,e):
    gam = np.zeros((e+1))
    u = np.zeros((e+1))
    
    bet = b[s]
    u[s] = r[s]/bet
    
    for i in range(s+1,e+1):
        gam[i] = c[i-1]/bet
        bet = b[i] - a[i]*gam[i]
        u[i] = (r[i] - a[i]*u[i-1])/bet
    
    for i in range(e-1,s-1,-1):
        u[i] = u[i] - gam[i+1]*u[i+1]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdms(a,b,c,alpha,beta,r,s,e):
    bb = np.zeros((e+1))
    u = np.zeros((e+1))
    
    gamma = -b[s]
    bb[s] = b[s] - gamma
    bb[e] = b[e] - alpha*beta/gamma
    
#    for i in range(s+1,e):
#        bb[i] = b[i]
    
    bb[s+1:e] = b[s+1:e]
    
    x = tdms(a,bb,c,r,s,e)
    
    u[s] = gamma
    u[e] = alpha
    
    z = tdms(a,bb,c,u,s,e)
    
    fact = (x[s] + beta*x[e]/gamma)/(1.0 + z[s] + beta*z[e]/gamma)
    
#    for i in range(s,e+1):
#        x[i] = x[i] - fact*z[i]
    
    x[s:e+1] = x[s:e+1] - fact*z[s:e+1]
        
    return x

#-----------------------------------------------------------------------------#
#cu3dp: 3rd-order compact upwind scheme for the first derivative(up)
#       periodic boundary conditions (0=n), h=grid spacing
#       p: free upwind paramater suggested (p>0 for upwind)
#                                           p=0.25 in Zhong (JCP 1998)
#		
#-----------------------------------------------------------------------------#
def cu3dp(u,p,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    up = np.zeros((n+1))
    
    a[:] = 1.0 + p
    b[:] = 4.0
    c[:] = 1.0 - p

#    for i in range(1,n):
#        r[i] = ((-3.0-2.0*p)*u[i-1] + 4.0*p*u[i] + (3.0-2.0*p)*u[i+1])/h
    r[1:n] = ((-3.0-2.0*p)*u[0:n-1] + 4.0*p*u[1:n] + (3.0-2.0*p)*u[2:n+1])/h
    r[0] = ((-3.0-2.0*p)*u[n-1] + 4.0*p*u[0] + (3.0-2.0*p)*u[1])/h  
    
    alpha = 1.0 - p
    beta = 1.0 + p
    
    x = ctdms(a,b,c,alpha,beta,r,0,n-1)
    
    up[0:n] = x[0:n]
    
    up[n] = up[0]
    
    return up

#-----------------------------------------------------------------------------#
# c4dp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4dp(u,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    up = np.zeros((n+1))
    
    a[:] = 1.0/4.0
    b[:] = 1.0
    c[:] = 1.0/4.0

#    for i in range(1,n):
#        r[i] = (3.0/2.0)*(u[i+1] - u[i-1])/(2.0*h)
    r[1:n] = (3.0/2.0)*(u[2:n+1] - u[0:n-1])/(2.0*h)
    r[0] = (3.0/2.0)*(u[1] - u[n-1])/(2.0*h)
    
    alpha = 1.0/4.0
    beta = 1.0/4.0
    
    x = ctdms(a,b,c,alpha,beta,r,0,n-1)
    
    up[0:n] = x[0:n]
    
    up[n] = up[0]
    
    return up

#-----------------------------------------------------------------------------#
# c4ddp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4ddp(u,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    upp = np.zeros((n+1))
    
    a[:] = 1.0/10.0
    b[:] = 1.0
    c[:] = 1.0/10.0

#    for i in range(1,n):
#        r[i] = (6.0/5.0)*(u[i-1] - 2.0*u[i] + u[i+1])/(h*h)
    
    r[1:n] = (6.0/5.0)*(u[0:n-1] - 2.0*u[1:n] + u[2:n+1])/(h*h)
    r[0] = (6.0/5.0)*(u[n-1] - 2.0*u[0] + u[1])/(h*h)
    
    alp = 1.0/10.0
    beta = 1.0/10.0
    
    x = ctdms(a,b,c,alp,beta,r,0,n-1)
    
    upp[0:n] = x[0:n]
    
    upp[n] = upp[0]
    
    return upp      
        
        
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
def smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[1:nx+2,1:ny+2] + s[3:nx+4,3:ny+4] \
                                             -s[3:nx+4,1:ny+2] - s[1:nx+2,3:ny+4])
    
    dsdxx = (1.0/(dx*dx))*(s[3:nx+4,2:ny+3] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[1:nx+2,2:ny+3])
    
    dsdyy = (1.0/(dy*dy))*(s[2:nx+3,3:ny+4] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[2:nx+3,1:ny+2])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev
    
#%%
# compute jacobian using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def jacobian(nx,ny,dx,dy,re,w,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    # Arakawa
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
    
    return jac
    
    
#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa):
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
    
    #call Smagorinsky model       
    #cs = 0.18
    #ev = dyn_smag(nx,ny,dx,dy,s)
    
    #Central difference for Laplacian
    # f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap if using eddy viscosity model for LES
    if ifm == 0:
        f[2:nx+3,2:ny+3] = -jac + lap/re 
        
    elif ifm == 1:
        ev = dyn_smag(nx,ny,kappa,s,w)
        f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap
                        
    return f

#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhsa(nx,ny,dx,dy,re,we,se):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+5,ny+5))
    
    w = we[1:nx+4,1:ny+4]
    s = se[1:nx+4,1:ny+4]
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    #call Smagorinsky model       
    #cs = 0.18
    #ev = smag(nx,ny,dx,dy,s,cs)
    
    #Central difference for Laplacian
    # f[1:nx+2,1:ny+2] = -jac + lap/re + ev*lap if using eddy viscosity model for LES
    
    f[2:nx+3,2:ny+3] = -jac + lap/re 
                        
    return f

#%%
def rhs_cu3(nx,ny,dx,dy,re,pCU3,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))
    
    # compute wxx
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wxx = c4ddp(a,dx,nx)
        
        lap[2:nx+3,j] = wxx[:]
    
    # compute wyy
    for i in range(2,nx+3):
        a = w[i,2:ny+3]        
        wyy = c4ddp(a,dx,nx)
        
        lap[i,2:ny+3] = lap[i,2:ny+3] + wyy[:]
    
    # Jacobian (convective term): upwind
    
    # sy: u
    sy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = s[i,2:ny+3]        
        sy[i-2,:] = c4dp(a,dx,nx)
    
    # computation of wx
    wxp = np.zeros((nx+1,ny+1))
    wxn = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        
        # upwind for wx
        wxp[:,j-2] = cu3dp(a, pCU3, dx, nx)
        # downwind for wx        
        wxn[:,j-2] = cu3dp(a, -pCU3, dx, nx)
    
    # upwinding
    syp = np.where(sy>0,sy,0) # max(sy[i,j],0)
    syn = np.where(sy<0,sy,0) # min(sy[i,j],0)

    #jac[2:nx+3,2:ny+3] = syp*wxp + syn*wxn
    
    # sx: -v
    sx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = s[2:nx+3,j]
        sx[:,j-2] = -c4dp(a, dx, nx)
    
    # computation of wy
    wyp = np.zeros((nx+1,ny+1))
    wyn = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = w[i,2:ny+3]
        
        # upwind for wy
        wyp[i-2,:] = cu3dp(a, pCU3, dy, ny)
        # downwind for wy        
        wyn[i-2,:] = cu3dp(a, -pCU3, dy, ny)
    
    # upwinding
    sxp = np.where(sx>0,sx,0) # max(sx[i,j],0)
    sxn = np.where(sx<0,sx,0) # min(sx[i,j],0)
    
    jac[2:nx+3,2:ny+3] = (syp*wxp + syn*wxn) + (sxp*wyp + sxn*wyn)
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re 
    
    del sy, sx, syp, syn, sxp, sxn, wxp, wxn, wyp, wyn
    
    return f

#%%
def rhs_compact(nx,ny,dx,dy,re,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))
    
    # compute wxx
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wxx = c4ddp(a,dx,nx)
        
        lap[2:nx+3,j] = wxx[:]
    
    # compute wyy
    for i in range(2,nx+3):
        a = w[i,2:ny+3]        
        wyy = c4ddp(a,dx,nx)
        
        lap[i,2:ny+3] = lap[i,2:ny+3] + wyy[:]
    
    # Jacobian (convective term): upwind
    
    # sy
    sy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = s[i,2:ny+3]        
        sy[i-2,:] = c4dp(a,dx,nx)
    
    # computation of wx
    wx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wx[:,j-2] = c4dp(a,dx,nx)
        
    
    # sx
    sx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = s[2:nx+3,j]
        sx[:,j-2] = c4dp(a, dx, nx)
    
    # computation of wy
    wy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = w[i,2:ny+3]
        wy[i-2,:] = c4dp(a, dx, nx)
    
    jac[2:nx+3,2:ny+3] = (sy*wx - sx*wy)
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re
    
    del sy, wx, sx, wy
    
    return f

#%%
# compute exact solution for TGV problem
def exact_tgv(nx,ny,x,y,time,re):
    ue = np.zeros((nx+5,ny+5))
    
    nq = 4.0
    ue[2:nx+3, 2:ny+3] = 2.0*nq*np.cos(nq*x[0:nx+1, 0:ny+1])*np.cos(nq*y[0:nx+1, 0:ny+1])*np.exp(-2.0*nq*nq*time/re)
    
    ue = bc(nx,ny,ue)
    return ue

#%%
# set initial condition for TGV problem
def ic_tgv(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    nq = 4.0
    w[2:nx+3, 2:ny+3] = 2.0*nq*np.cos(nq*x[0:nx+1, 0:ny+1])*np.cos(nq*y[0:nx+1, 0:ny+1])
    
    w = bc(nx,ny,w)

    return w

#%%
# set initial condition for vortex merger problem
def ic_vm(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    sigma = np.pi
    xc1 = np.pi-np.pi/4.0
    yc1 = np.pi
    xc2 = np.pi+np.pi/4.0
    yc2 = np.pi
    
    w[2:nx+3, 2:ny+3] = np.exp(-sigma*((x[0:nx+1, 0:ny+1]-xc1)**2 + (y[0:nx+1, 0:ny+1]-yc1)**2)) \
                        + np.exp(-sigma*((x[0:nx+1, 0:ny+1]-xc2)**2 + (y[0:nx+1, 0:ny+1]-yc2)**2))
    
    w = bc(nx,ny,w)

    return w

#%%
def ic_shear(nx,ny,x,y):
    w = np.zeros((nx+5,ny+5))
    delta = 0.05
    sigma = 15/np.pi
    
#    for j in range(2,ny+3):
#        for i in range(2,nx+3):
#            if y[i-2,j-2] <= np.pi:
#                w[i,j] = delta*np.cos(x[i-2,j-2]) - sigma/  \
#                        (np.cosh(sigma*(y[i-2,j-2] - np.pi/2)))**2
#            else:
#                w[i,j] = delta*np.cos(x[i-2,j-2]) + sigma/  \
#                        (np.cosh(sigma*(3*np.pi/2 - y[i-2,j-2])))**2
    
    indy = np.array(np.where(y[0,:] <= np.pi))
    indy = indy.flatten()
    
    w[2:nx+3, indy+2] = delta*np.cos(x[0:nx+1, indy]) - sigma/  \
                        (np.cosh(sigma*(y[0:nx+1, indy] - np.pi/2)))**2
        
    indy = np.array(np.where(y[0,:] > np.pi))
    indy = indy.flatten()    
    w[2:nx+3, indy+2] = delta*np.cos(x[0:nx+1, indy]) + sigma/(np.cosh(sigma*(3.0*np.pi/2 - y[0:nx+1, indy])))**2
    
    w = bc(nx,ny,w)
    
    return w                
    #plt.contourf(x,y,w[1:nx+2, 1:ny+2],100,cmap='jet')
                    
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
def energy_spectrum(nx,ny,w):
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
def plotimage(x,y):
    fig, ax = plt.subplots(1,1,sharey=True,figsize=(6,5))
    cs1 = ax.contourf(x.T, 120, cmap = 'jet', interpolation='bilinear')
    ax.set_title("True")
    plt.colorbar(cs1, ax=ax)
    plt.show()
    
    fig, ax = plt.subplots(1,1,sharey=True,figsize=(6,5))
    cs2 = ax.contourf(y.T, 120, cmap = 'jet', interpolation='bilinear')
    ax.set_title("Coarsened")
    plt.colorbar(cs2, ax=ax)
    plt.show()

#%%
def coarsen(nx,ny,nxc,nyc,w,wc):
    wf = np.fft.fft2(w[1:nx+1,1:ny+1])
    
    wfc = np.zeros((nxc,nyc),dtype='complex')
    
    wfc[0:int(nxc/2),0:int(nyc/2)] = wf[0:int(nxc/2),0:int(nyc/2)]
        
    wfc[int(nxc/2):,0:int(nyc/2)] = wf[int(nx-nxc/2):,0:int(nyc/2)]
    
    wfc[0:int(nxc/2),int(nyc/2):] = wf[0:int(nxc/2),int(ny-nyc/2):]
    
    wfc[int(nxc/2):,int(nyc/2):] =  wf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    wfc = wfc*(nxc*nyc)/(nx*ny)
    
    wtc = np.real(np.fft.ifft2(wfc))
    
    wc[1:nxc+1,1:nyc+1] = np.real(wtc)
    wc[:,nyc+1] = wc[:,1]
    wc[nxc+1,:] = wc[1,:]
    wc[nxc+1,nyc+1] = wc[1,1]
    
    wc = bc(nxc,nyc,wc)
    

#%%
def export_data(nx,ny,re,n,w,s):
    folder = 'data_'+str(int(re)) + '_'+ str(nx) + '_' + str(ny)
    
    if not os.path.exists('./results/'+folder):
        os.makedirs('./results/'+folder)
        
    filename = './results/'+folder+'/results_' + str(int(n))+'.npz'
    np.savez(filename,w=w[2:nx+3,2:ny+3],s=s[2:nx+3,2:ny+3])
        
    
#%% 
# read input file
l1 = []
with open('input.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ifm = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])
ichkp = np.int64(l1[10][0])
istart = np.int64(l1[11][0])
kappa = np.int64(l1[12][0])
pCU3 = np.float64(l1[13][0])

freq = int(nt/ns)

#%% 
# assign parameters
nx = nd
ny = nd

nxc = ndc
nyc = ndc

#%%
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

x, y = np.meshgrid(x, y, indexing='ij')

#%% 
# allocate the vorticity and streamfunction arrays
w = np.empty((nx+5,ny+5)) 
s = np.empty((nx+5,ny+5))

t = np.empty((nx+5,ny+5))

r = np.empty((nx+5,ny+5))

#%%
# set the initial condition based on the problem selected
if (ipr == 1):
    w0 = ic_tgv(nx,ny,x,y)
elif (ipr == 2):
    w0 = ic_vm(nx,ny,x,y)
elif (ipr == 3):
    w0 = ic_shear(nx,ny,x,y)
elif (ipr == 4):
    w0 = ic_decay(nx,ny,dx,dy)
    
w = np.copy(w0)
s = fps(nx, ny, dx, dy, -w)
s = bc(nx,ny,s)

export_data(nx,ny,re,0,w,s)
 
def rhs(nx,ny,dx,dy,re,pCU3,w,s,isolver):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa)
        #return rhsa(nx,ny,dx,dy,re,w,s)
    if isolver == 2:
        return rhs_compact(nx,ny,dx,dy,re,w,s)
    if isolver == 3:
        return rhs_cu3(nx,ny,dx,dy,re,pCU3,w,s)
    

#%%
# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

for k in range(1,nt+1):
    time = time + dt
    
    r = rhs(nx,ny,dx,dy,re,pCU3,w,s,isolver)
    
    #stage-1
    t[2:nx+3,2:ny+3] = w[2:nx+3,2:ny+3] + dt*r[2:nx+3,2:ny+3]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,pCU3,t,s,isolver)

    #stage-2
    t[2:nx+3,2:ny+3] = 0.75*w[2:nx+3,2:ny+3] + 0.25*t[2:nx+3,2:ny+3] + 0.25*dt*r[2:nx+3,2:ny+3]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,pCU3,t,s,isolver)

    #stage-3
    w[2:nx+3,2:ny+3] = aa*w[2:nx+3,2:ny+3] + bb*t[2:nx+3,2:ny+3] + bb*dt*r[2:nx+3,2:ny+3]
    
    w = bc(nx,ny,w)
    
    s = fps(nx, ny, dx, dy, -w)
    s = bc(nx,ny,s)
    
    if (k%freq == 0):
        print(k, " ", time)
        export_data(nx,ny,re,int(k/freq),w,s)

total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)

#%%
# exact solution for TGV problem
if (ipr == 1):
    we = exact_tgv(nx,ny,x,y,time,re)
    
# compute the exact, initial and final energy spectrum
if (ipr == 4):
    ent, n = energy_spectrum(nx,ny,w)
    en0, n = energy_spectrum(nx,ny,w0)
    k = np.linspace(1,n,n)
    
    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    ese = c*(k**4)*np.exp(-(k/k0)**2)
    
    np.savetxt("energy_arakawa_"+str(int(re))+"_"+str(nd)+".csv", ent, delimiter=",")

#%%
# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

cs = axs[0].contourf(x,y,w0[2:nx+3,2:ny+3], 120, cmap = 'jet')
axs[0].set_title('$t=0$')

cs = axs[1].contourf(x,y,w[2:nx+3,2:ny+3], 120, cmap = 'jet')
axs[1].set_title('$t=4$')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.2, -0.03, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("field_fdm.png", bbox_inches = 'tight')


#%%
if (ipr == 4):

    fig, ax = plt.subplots()
    fig.set_size_inches(6,5)
    
    line = 100*k**(-3.0)
    
    ax.loglog(k,ese[:],'k', lw = 2, label='Exact')
    ax.loglog(k,en0[1:],'r', ls = '--', lw = 2, label='$t = 0.0$')
    ax.loglog(k,ent[1:], 'b', lw = 2, label = '$t = '+str(dt*nt)+'$')
    ax.loglog(k,line, 'g--', lw = 2, label = '$k^-3$')
    
    
    plt.xlabel('$K$')
    plt.ylabel('$E(K)$')
    plt.legend(loc=0)
    plt.ylim(1e-12,1e-1)
    fig.savefig('es_fdm.png', bbox_inches = 'tight', pad_inches = 0)
    

#%%
en_dns = np.loadtxt("energy_arakawa_"+str(int(re))+"_"+str(int(1024))+"_dns.csv")
en_coarse = np.loadtxt("energy_arakawa_"+str(int(re))+"_"+str(int(nd))+"_coarse.csv")
en_dsm = np.loadtxt("energy_arakawa_"+str(int(re))+"_"+str(int(nd))+"_dsm.csv")
 
fig, ax = plt.subplots()
fig.set_size_inches(6,5)


kf = np.linspace(1,en_dns.shape[0]-1,en_dns.shape[0]-1)

kl = kf[20:100]
line = 500*kl**(-3.0)

#ax.loglog(k,ese[:],'k', lw = 2, label='Exact')
#ax.loglog(k,en0[1:],'r', ls = '--', lw = 2, label='$t = 0.0$')

ax.loglog(kf,en_dns[1:], 'b', lw = 2, label = 'DNS')
ax.loglog(k,en_dsm[1:], 'g', lw = 2, label = 'DSM')
ax.loglog(k,ent[1:], 'm', lw = 2, label = 'CU3 ($p=0.25$)')
ax.loglog(k,en_coarse[1:], 'y', lw = 2, label = 'No model')

ax.loglog(kl,line, 'k-.', lw = 2)


plt.xlabel('$K$')
plt.ylabel('$E(K)$')
plt.legend(loc=0)
plt.xlim(1,200)
plt.ylim(1e-6,1e-1)
plt.text(50, 1e-2, '$k^-3$', color='k')

fig.savefig('es_fdm_all.png', bbox_inches = 'tight', pad_inches = 0)    
