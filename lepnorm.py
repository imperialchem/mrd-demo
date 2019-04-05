#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:17:56 2017

@author: Tristan Mackenzie

    This file is part of LepsPy.

    LepsPy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LepsPy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LepsPy.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

epsilon = 1.0E-15

# Using jit to speed up code
def use_jit(signature_or_function, nopython):
    def decorator(func):
        try:
            import numba
            print("Using jit for {0}".format(func.__name__))
            return numba.jit(signature_or_function=signature_or_function,nopython=nopython)(func)
        except:
            print("Not using jit for {0}".format(func.__name__))
            return func
    return decorator

@use_jit("Tuple((float64,float64))(float64,float64[:],float64[:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,boolean)", nopython=True)
def lepnorm(theta,Displacements,First_derivative,Hessian,Velocities,Accelerations,Masses,Reduced_Masses,dt,MEP):

    Momenta = Reduced_Masses * Velocities
    
    # G-Matrix
    y1 = 1/Masses[0]
    y2 = 1/Masses[1]
    y3 = 1/Masses[2]

    GM = np.zeros((3,3))
    GM[0][0]=y1+y2
    GM[0][1]=y2*np.cos(theta)
    GM[1][0]=GM[0][1]
    GM[1][1]=y3+y2
    GM[0][2]=-y2*np.sin(theta)/Displacements[1]
    GM[2][0]=GM[0][2]
    GM[1][2]=-y2*np.sin(theta)/Displacements[0]
    GM[2][1]=GM[1][2]
    gmt=1/(Displacements[0] ** 2)+1/(Displacements[1] ** 2)-(2*np.cos(theta)/(Displacements[0]*Displacements[1]))
    GM[2][2]=y1/(Displacements[0] ** 2)+y3/(Displacements[1] ** 2)+y2*gmt
    
    GMVal, GMVec = np.linalg.eig(GM)
    GMVal = np.diag(GMVal)
    GMVal1 = GMVal ** 0.5
    GMVal2 = np.diag(np.diag(GMVal) ** -0.5)
    GRR    = np.dot(np.dot(GMVec, GMVal1), GMVec.T)
    GROOT  = np.dot(np.dot(GMVec, GMVal2), GMVec.T)

    # G-Matrix Weighted Hessian;
    MWH = np.dot(np.dot(GRR, Hessian), GRR)
    W2, ALT = np.linalg.eig(MWH) #ALT is antisymmetric version in Fort code but that does not give the right G-Matrix!!!!
    
    # Gradient Vector in mass-weighted coordinates
    GRAD = -First_derivative
    GRADN = np.dot(ALT.T ,np.dot(GRR, GRAD))
    
    PCMO = np.dot(ALT.T,np.dot(GRR, Momenta))
    
    ktot = 0.5 * (PCMO[0] ** 2 + PCMO[1] ** 2 + PCMO[2] ** 2)
    
    q = np.zeros((3))
    for i in range(3):
        if W2[i] < - epsilon:
            wmod = abs(W2[i]) ** 0.5
            wmt = wmod * dt
            q[i]=PCMO[i] * np.sinh(wmt) / wmod + GRADN[i] * (1 - np.cosh(wmt)) / (wmod ** 2)
            PCMO[i] = PCMO[i] * np.cosh(wmt) - GRADN[i] * np.sinh(wmt) / wmod
        elif abs(W2[i]) < epsilon:
            q[i] = PCMO[i] * dt - (0.5 * GRADN[i] * (dt ** 2))
            PCMO[i] = PCMO[i] - GRADN[i] * dt
        else:
            wroot =W2[i] ** 0.5
            tfn1 = GRADN[i] * (1 - np.cos(wroot * dt)) / (wroot ** 2)
            q[i] = PCMO[i] * np.sin(wroot * dt) / wroot - tfn1
            PCMO[i] = PCMO[i] * np.cos(wroot * dt) - GRADN[i] * np.sin(wroot * dt) / wroot
            
    XX = np.dot(GRR, np.dot(ALT, q.T))
        
    if MEP:
      XX *= 5
      
      
        
    Displacements[0] += XX[0]
    Displacements[1] += XX[1]
    thetaf = theta + XX[2]
    Displacements[2] = ((Displacements[0] ** 2) + (Displacements[1] ** 2) - 2 * Displacements[0] * Displacements[1] * np.cos(thetaf)) ** 0.5
    Momenta = np.dot(np.dot(GROOT, ALT), PCMO)
    
    Velocities[0] = Momenta[0] / Reduced_Masses[0]
    Velocities[1] = Momenta[1] / Reduced_Masses[1]
    # This line needs to be changed
    Velocities[2] = 0
    
    Accelerations[0] = First_derivative[0] / Reduced_Masses[0]
    Accelerations[1] = First_derivative[1] / Reduced_Masses[1]
    if (Accelerations[0] + Accelerations[1] < 0):
        Accelerations[2] = - ((Accelerations[0] ** 2) + (Accelerations[1] ** 2) - 2 * Accelerations[0] * Accelerations[1] * np.cos(thetaf)) ** 0.5
    else:
        Accelerations[2] =   ((Accelerations[0] ** 2) + (Accelerations[1] ** 2) - 2 * Accelerations[0] * Accelerations[1] * np.cos(thetaf)) ** 0.5
    
    return thetaf,ktot
