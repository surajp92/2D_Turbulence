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
def c4dpp(u,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    upp = np.zeros((n+1))
    
    a[:] = 1.0/10.0
    b[:] = 1.0
    c[:] = 1.0/10.0

    for i in range(1,n):
        r[i] = (6.0/5.0)*(u[i-1] - 2.0*u[i] + u[i+1])/(h*h)
#    r[1:n] = (6.0/5.0)*(u[0:n-1] - 2.0*u[1:n] + u[2:n+1])/(2.0*h)
    r[0] = (6.0/5.0)*(u[n-1] - 2.0*u[0] + u[1])/(2.0*h)
    
    alpha = 1.0/4.0
    beta = 1.0/4.0
    
    x = ctdms(a,b,c,alpha,beta,r,0,n-1)
    
    upp[0:n] = x[0:n]
    
    upp[n] = upp[0]
    
    return upp


def rhs_c4dpp(nx,dx,alpha,u):
   
    upp = c4dpp(u,dx,nx)
    r = alpha*upp
    
    return r

def rhs_cs(nx,dx,a,u):
    r = np.zeros((nx+1))
    v = np.zeros((nx+5))
    
    v[2:nx+3] = u[:] 
    v[0] = v[nx]
    v[1] = v[nx+1]
    v[nx+3] = v[3]
    v[nx+4] = v[4]
    
    r[:] = -a*(-v[4:nx+5] + 8.0*v[3:nx+4] - 8.0*v[1:nx+2] + v[0:nx+1])/(12.0*dx)
    
#    for i in range(nx+1):
#        r[i] = -a*(-v[i+4] + 8.0*v[i+3] - 8.0*v[i+1] + v[i])/(12.0*dx)
    
    return r
    
#%%
scheme = 3 # [1] FT Upwind [2] RK4 C4 [3] RK4 C4DPP 
x0 = 0.0
xL = 1.0
nx = 40
tmax = 1.0
a = 1.0
cfl = 0.5

pCU3 = 0.0

dx = (xL-x0)/nx
x = np.linspace(x0,xL,nx+1)
dt = cfl*dx/a
nt = tmax/dt

u = np.sin(2.0*np.pi*x)

v = np.zeros((nx+3))

t = 0

#%%
for k in range(1,int(nt)+1):
    if scheme == 1:
        c = a*dt/dx
        v[1:nx+2] = u[:]
        v[0] = v[nx]
        v[nx+2] = v[2]
        
        u = v[1:nx+2] - c*(v[1:nx+2] - v[0:nx+1])
        
    elif scheme == 2:
        r1 = rhs_cs(nx,dx,a,u)
        k1 = dt*r1
        
        r2 = rhs_cs(nx,dx,a,u+0.5*k1)
        k2 = dt*r2
        
        r3 = rhs_cs(nx,dx,a,u+0.5*k2)
        k3 = dt*r3
        
        r4 = rhs_cs(nx,dx,a,u+k3)
        k4 = dt*r4
        
        u = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    elif scheme == 3:        
        r1 = rhs_c4dpp(nx,dx,a,u)
        k1 = dt*r1
        
        r2 = rhs_c4dpp(nx,dx,a,u+0.5*k1)
        k2 = dt*r2
        
        r3 = rhs_c4dpp(nx,dx,a,u+0.5*k2)
        k3 = dt*r3
        
        r4 = rhs_c4dpp(nx,dx,a,u+k3)
        k4 = dt*r4
        
        u = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    t = t + dt
    #print(k, ' ',  t,  ' ', np.max(u))
        
        
#%%
ue = np.sin(2.0*np.pi*(x-a*tmax))

print(np.linalg.norm(ue-u))

plt.plot(x,ue,'k')
plt.plot(x,u,'r--')
plt.show()

unf = np.genfromtxt('./fortran/numerical.plt',skip_header=1)

print(np.linalg.norm(ue-unf[:,1]))

a = u - unf[:,1]

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
#plt.show()

