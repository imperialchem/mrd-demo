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

def lepnorm(drab,drbc,theta,Frab,Frbc,Frac,vrabi,vrbci,vraci,hr1r1,hr1r2,hr1r3,hr2r2,hr2r3,hr3r3,ma,mb,mc,ti,dt,MEP):

    hessian = np.array([[hr1r1, hr1r2, hr1r3], [hr1r2, hr2r2, hr2r3], [hr1r3, hr2r3, hr3r3]])
    
    # Reduced masses.
    mab = (ma*mb)/(ma+mb)
    mbc = (mb*mc)/(mb+mc)
    mac = (ma*mc)/(ma+mc)

    prab = vrabi*mab
    prbc = vrbci*mbc
    prac = vraci*mac
    
    # G-Matrix
    # See E. B. Wildon Jr., J. C. Decius and P. C. Cross "Molecular Vibrations", McGraw-Hill (1955), sec. 4-6

    GM=np.array([[1/ma + 1/mb, np.cos(theta)/mb, -np.sin(theta)/(drbc*mb)],
                 [np.cos(theta)/mb, 1/mb + 1/mc, -np.sin(theta)/(drab*mb)],
                 [-np.sin(theta)/(drbc*mb), -np.sin(theta)/(drab*mb), 1/(drab**2 * ma) + 1/(drbc**2 * mc) + 1/(drab**2 * mb) + 1/(drbc**2 * mb) -2*np.cos(theta)/(drab*drbc*mb)]])

    
    GMVal, GMVec = np.linalg.eig(GM)

    GMVal1 = GMVal ** 0.5    # 1/sqrt(mass) for each mode
    GMVal2 = GMVal ** (-0.5) # sqrt(mass) for each mode

    GRR    = GMVec.dot(np.diag(GMVal1)).dot(GMVec.T)
    GROOT  = GMVec.dot(np.diag(GMVal2)).dot(GMVec.T)

    # G-Matrix Weighted Hessian;
    MWH = GRR.dot(hessian).dot(GRR)
    W2, ALT = np.linalg.eig(MWH); #ALT is antisymmetric version in Fort code but that does not give the right G-Matrix!!!!
    
    # Gradient Vector in mass-weighted coordinates
    GRAD = np.array([-Frab, -Frbc, -Frac])
    GRADN = ALT.T.dot(GRR).dot(GRAD)
    
    # Momentum Vector in Normal Coordinates
    MOM = np.array([prab, prbc, prac])
    PCMO = ALT.T.dot(GRR).dot(MOM)
    
    ktot = 0.5 * np.linalg.norm(PCMO)**2
    
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
            
    XX = GRR.dot(ALT).dot(q.T)
        
    if MEP:
      XX *= 5
      
      
        
    drabf = drab + XX[0]
    drbcf = drbc + XX[1]
    thetaf = theta + XX[2]
    dracf = ((drab ** 2) + (drbc ** 2) - 2 * drab * drbc * np.cos(thetaf)) ** 0.5
    MOM = GROOT.dot(ALT).dot(PCMO)
    
    tf = ti + dt
    vrabf = MOM[0] / mab
    vrbcf = MOM[1] / mbc
    vracf = 0
    
    arab = Frab / mab
    arbc = Frbc / mbc
    if (arab + arbc < 0):
        arac = - ((arab ** 2) + (arbc ** 2) - 2 * arab * arbc * np.cos(thetaf)) ** 0.5
    else:
        arac =   ((arab ** 2) + (arbc ** 2) - 2 * arab * arbc * np.cos(thetaf)) ** 0.5
    
    return drabf,drbcf,dracf,thetaf,vrabf,vrbcf,vracf,tf,arab,arbc,arac,ktot
