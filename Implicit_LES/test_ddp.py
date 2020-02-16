#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:25:20 2020

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

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

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


def rhs_cs(nx,dx,alpha,u):
    r = np.zeros((nx+1))
    c = alpha/(dx*dx)
    
    r[:] = u[:]        
    
    r[1:nx] = c*(r[2:nx+1] - 2.0*r[1:nx] + r[0:nx-1]) 
    
    return r
    
#%%
scheme = 3 # [1] FT CS2 [2] RK4 CS2 [3] RK4 C4DPP 
x0 = 0
xL = np.pi
nx = 10
tmax = 0.5
alpha = 1.0


dx = (xL-x0)/nx
x = np.linspace(x0,xL,nx+1)
dt = 0.0001
nt = tmax/dt

u = np.sin(x)

t = 0

#%%
for k in range(1,int(nt)+1):
    if scheme == 1:
        
        v = np.zeros((nx+1))
        f = np.zeros((nx+1))
        c = alpha*dt/(dx*dx)
        
        v[:] = u[:]        
        f = alpha*(np.pi**2 - 1.0)*np.exp(-alpha*t)*np.sin(np.pi*x) 
        
#        for i in range(1,nx):
#            u[i] = v[i] + c*(v[i+1] - 2.0*v[i] + v[i-1]) + dt*f[i]
        u[1:nx] = v[1:nx] + c*(v[2:nx+1] - 2.0*v[1:nx] + v[0:nx-1]) 
        
    elif scheme == 2:
        
        r1 = rhs_cs(nx,dx,alpha,u)
        k1 = dt*(r1)
        
        r2 = rhs_cs(nx,dx,alpha,u+0.5*k1)
        k2 = dt*(r2)
        
        r3 = rhs_cs(nx,dx,alpha,u+0.5*k2)
        k3 = dt*(r3)
        
        r4 = rhs_cs(nx,dx,alpha,u+k3)
        k4 = dt*(r4)
        
        u[1:nx] = u[1:nx] + (k1[1:nx] + 2.0*(k2[1:nx] + k3[1:nx]) + k4[1:nx])/6.0
    
    elif scheme == 3:

        r1 = c4ddp(u,dx,nx)
        k1 = dt*(alpha*r1)
        
        r2 = c4ddp(u+0.5*k1,dx,nx)
        k2 = dt*(alpha*r2)
        
        r3 = c4ddp(u+0.5*k2,dx,nx)
        k3 = dt*(alpha*r3)
        
        r4 = c4ddp(u+k3,dx,nx)
        k4 = dt*(alpha*r4)
        
#        u = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
        u[1:nx] = u[1:nx] + (k1[1:nx] + 2.0*(k2[1:nx] + k3[1:nx]) + k4[1:nx])/6.0
    
    t = t + dt
    #print(k, ' ',  t,  ' ', np.max(u))
        
        
#%%
ue = np.exp(-alpha*tmax)*np.sin((x))
ui = np.sin(x)

print(np.linalg.norm(ue-u))

plt.plot(x,ui,'gs-',label='Initial')
plt.plot(x,ue,'ko-',label='Exact')
plt.plot(x,u,'r--',label='Numerical')
plt.legend()
plt.show()

#unf = np.genfromtxt('./fortran/numerical.plt',skip_header=1)
#
#print(np.linalg.norm(ue-unf[:,1]))
#
#a = u - unf[:,1]

#%%

#err1 = np.array([0.0012458834764954565,0.00010678372936789801,9.307855084871244e-06,8.172890312554292e-07])
#err3 = np.array([0.01,0.01/9,0.01/81,0.01/729])
#grid = np.array([20,40,80,160])
#
#plt.loglog(grid,err,'bo-')
#plt.loglog(grid+10,err3,'r--')
#plt.show()

#%%

#err = np.array([0.006791926863957603,0.0005917112530679781,5.178759729249218e-05,4.5519201564702094e-06])
#err3 = np.array([0.01,0.01/9,0.01/81,0.01/729])
#grid = np.array([20,40,80,160])
#
#plt.loglog(grid,err,'bo-')
#plt.loglog(grid,err1,'gs-')
#plt.loglog(grid+10,err3,'r--')
#plt.show() 0.09618854663456089, 0.05648325862124369, 0.00218486518225985

