#Created on Wed May 17 10:17:56 2017
#
#@author: Tristan Mackenzie
#
#    This file is part of LepsPy.
#
#    LepsPy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    LepsPy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LepsPy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

def lepnorm(coord,mom,masses,gradient,hessian,dt,MEP):
    '''
    Updates coordinates and momenta by a time step (or an arbitraty step in the
    case of a minimum energy path (MEP)).

    coord, mom, gradient and hessian are all arrays in internal coordinates rAB,
    rBC and theta.
    mass is an array with masses of atoms A, B and C.
    dt is the size of the timestep.
    MEP is a boolian defining whether the calculation is a MEP.

    The function first converts from internal coordinates into mass-weighted
    normal modes, the displacement is calculated in normal modes, and converted
    back to internal coordinates.
    '''

    theta = coord[2]

    # G-Matrix
    # See E. B. Wildon Jr., J. C. Decius and P. C. Cross "Molecular Vibrations", McGraw-Hill (1955), sec. 4-6

    GM=np.array([[np.sum(1/masses[0:2]), np.cos(theta)/masses[1], -np.sin(theta)/(coord[1]*masses[1])],
                 [np.cos(theta)/masses[1], np.sum(1/masses[1:]), -np.sin(theta)/(coord[0]*masses[1])],
                 [-np.sin(theta)/(coord[1]*masses[1]), -np.sin(theta)/(coord[0]*masses[1]),
                   np.sum(1/(coord[0:2]**2 * masses[0:2])) + np.sum(1/(coord[0:2]**2 * masses[1:])) -2*np.cos(theta)/np.prod(coord[0:2]*masses[1])]])
    
    GMVal, GMVec = np.linalg.eig(GM)

    GMVal1 = GMVal ** 0.5    # 1/sqrt(mass) for each mode
    GMVal2 = GMVal ** (-0.5) # sqrt(mass) for each mode

    GRR    = GMVec.dot(np.diag(GMVal1)).dot(GMVec.T)
    GROOT  = GMVec.dot(np.diag(GMVal2)).dot(GMVec.T)

    # Transform into normal modes and do dynamics in normal modes
    # Follow dynamics algorithm in:
    # T. Helgaker, E. Uggerud, H.J. Aa. Jensen, Chem. Phys Lett. 173(2,3):145-150 (1990)

    # G-Matrix Weighted Hessian;
    mwhessian = GRR.dot(hessian).dot(GRR)
    w2, transf = np.linalg.eig(mwhessian) #transf is antisymmetric version in Fort code but that does not give the right G-Matrix!!!!
    
    # Gradient Vector in mass-weighted normal modes
    gradN = transf.T.dot(GRR).dot(gradient)
    
    if not MEP: 
        # Momentum Vector in normal modes
        momN = transf.T.dot(GRR).dot(mom)
    else:
        # enforce zero momentum
        momN = np.zeros(3)
        # effectivelly increase step to compensate absence of inertial term
        dt = dt * 15
    
    # Calculate kinetic energy
    # (Calculate before updating momenta as this fits better with running of the rest of the code)
    ktot = 0.5 * np.dot(momN,momN)

    displacementN = np.zeros(3)

    epsilon = 1e-15

    for i in range(3):
        if w2[i] < - epsilon: # negative curvature of the potential
            wmod = abs(w2[i]) ** 0.5
            displacementN[i]=momN[i] * np.sinh(wmod*dt) / wmod + gradN[i] * (1 - np.cosh(wmod*dt)) / (wmod**2)
            if not MEP:
                momN[i] = momN[i] * np.cosh(wmod*dt) - gradN[i] * np.sinh(wmod*dt) / wmod
        elif abs(w2[i]) < epsilon: # no curvature in potential
            displacementN[i] = momN[i] * dt - (0.5 * gradN[i] * (dt ** 2))
            if not MEP:
                momN[i] = momN[i] - gradN[i] * dt
        else: # positive curvature of the potential
            wroot =w2[i] ** 0.5 
            displacementN[i]=momN[i] * np.sin(wroot*dt) / wroot - gradN[i] * (1 - np.cos(wroot*dt)) / (wroot**2)
            if not MEP:
                momN[i] = momN[i] * np.cos(wroot*dt) - gradN[i] * np.sin(wroot*dt) / wroot
            
    # update coordinates by first transforming displacementN into internal coordinates
    coord = coord + GRR.dot(transf).dot(displacementN) 
    
    # transform updated momentum back into internal coordinates
    mom = GROOT.dot(transf).dot(momN)
            
    return (coord,mom,ktot)
