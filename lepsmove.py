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
from numpy.linalg.linalg import LinAlgError
from lepspoint import leps_gradient,leps_hessian


def _gmat(coord,masses):
    '''Calculate the G-matrix of the system.'''
    # See E. B. Wildon Jr., J. C. Decius and P. C. Cross "Molecular Vibrations", McGraw-Hill (1955), sec. 4-6

    rAB,rBC,theta = coord

    G12 = np.cos(theta)/masses[1]
    G13 = -np.sin(theta)/(rBC*masses[1])
    G23 = -np.sin(theta)/(rAB*masses[1])

    G=np.array([[np.sum(1/masses[0:2]), G12, G13],
                [G12, np.sum(1/masses[1:3]), G23],
                [G13, G23,
                 np.sum(1/(rAB**2 * masses[0:2])) + \
                  np.sum(1/(rBC**2 * masses[1:3])) - \
                  2*np.cos(theta)/(rAB*rBC*masses[1]**2)]])

    return G


def kinetic_energy(coord,mom,masses):
    '''
    Calculate the kineic energy:

    K = 1/2 mom^T G mom

    coord and mom are arrays in internal coordinates.
    masses is an array with masses of the 3 atoms.
    '''

    G = _gmat(coord,masses)

    return 0.5 * mom.dot(G).dot(mom)


def velocities(coord,mom,masses):
    '''
    Calculate velocities in internal coordinates.
    These don't have a simple relation to momenta.
    They are calculate from Hamilton's equations by differentiating the kinetic
    energy with respect to momenta.
    '''

    G = _gmat(coord,masses)

    return mom.dot(G)


def velocity_AC(coord,vAB,vBC):
    '''
    Calculate internuclear velocity between A and C atoms, by doing
    the vecor sum of the AB and BC velocities and projecting it onto
    the AC axis.
    '''
    # setup frame vectors
    AB_vec=np.array([coord[0],0])
    BC_vec=np.array([-np.cos(coord[2]),np.sin(coord[2])])*coord[1]
    AC_vec=AB_vec+BC_vec
    # normalise frame vectors
    AB_nvec=AB_vec/np.linalg.norm(AB_vec)
    BC_nvec=BC_vec/np.linalg.norm(BC_vec)
    AC_nvec=AC_vec/np.linalg.norm(AC_vec)

    # velocity vectors
    AB_vvec=-vAB*AB_nvec
    BC_vvec=-vBC*BC_nvec
    AC_vvec=AB_vvec+BC_vvec #note this is not along AC_nvec

    return -np.dot(AC_vvec,AC_nvec)


def lepsnorm(coord,mom,masses,gradient,hessian,dt):
    '''
    Updates coordinates and momenta by a time step.

    coord, mom, gradient and hessian are all arrays in internal coordinates rAB,
    rBC and theta.
    mass is an array with masses of atoms A, B and C.
    dt is the size of the timestep.

    The function first converts from internal coordinates into mass-weighted
    normal modes, the displacement is calculated in normal modes, and converted
    back to internal coordinates.
    '''

    # G-Matrix
    GM=_gmat(coord,masses)
    
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
    
    # Momentum Vector in normal modes
    momN = transf.T.dot(GRR).dot(mom)
    
    displacementN = np.zeros(3)

    epsilon = 1e-14

    for i in range(3):
        if w2[i] < - epsilon: # negative curvature of the potential
            wmod = abs(w2[i]) ** 0.5
            displacementN[i]=momN[i] * np.sinh(wmod*dt) / wmod + gradN[i] * (1 - np.cosh(wmod*dt)) / (wmod**2)
            momN[i] = momN[i] * np.cosh(wmod*dt) - gradN[i] * np.sinh(wmod*dt) / wmod
        elif abs(w2[i]) < epsilon: # no curvature in potential
            displacementN[i] = momN[i] * dt - (0.5 * gradN[i] * (dt ** 2))
            momN[i] = momN[i] - gradN[i] * dt
        else: # positive curvature of the potential
            wroot =w2[i] ** 0.5 
            displacementN[i]=momN[i] * np.sin(wroot*dt) / wroot - gradN[i] * (1 - np.cos(wroot*dt)) / (wroot**2)
            momN[i] = momN[i] * np.cos(wroot*dt) - gradN[i] * np.sin(wroot*dt) / wroot
            
    # update coordinates by first transforming displacementN into internal coordinates
    coord = coord + GRR.dot(transf).dot(displacementN) 
    
    # transform updated momentum back into internal coordinates
    mom = GROOT.dot(transf).dot(momN)
            
    return (coord,mom)


def calc_trajectory(coord_init,mom_init,masses,morse_params,H_param,steps,dt,calc_type):
    '''
    Make the system move. This may be the calculation of a inertial trajectory
    (calc_type="Dynamics"), a minimum energy path (calc_type="MEP"), a local
    monimum or transitions state search (calc_type="Opt Min" or calc_type="Opt TS").

    The function outputs a trajectory array with position and momenta, and a string
    which may contain an error message.
    '''
    
    # Rewrite inputs to avoid side effects
    coord=coord_init
    mom=mom_init
    step_size=dt

    # If doing an MEP, set initial momenta to zero and increase step size to compensate
    # absence of inertial term
    if calc_type == "MEP":
        mom=np.zeros(3)
        step_size = 15*dt 
    
    #Initialise outputs
    trajectory = [np.column_stack((coord,mom))]
    error = ""

    #Flag to stop appending to output in case of a crash
    terminate = False        

    itcounter = 0
    while itcounter < steps and not terminate:
        itcounter = itcounter+1            

        #Get current gradient, and Hessian
        gradient = leps_gradient(*coord,morse_params,H_param)
        hessian = leps_hessian(*coord,morse_params,H_param)

        if calc_type in ["Opt Min", "Opt TS"]: #Optimisation calculations
            
            #Diagonalise Hessian
            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            
            #Eigenvalue test
            neg_eig_i = [i for i,eig in enumerate(eigenvalues) if eig < -0.01]
            if len(neg_eig_i) == 0 and self.calc_type == "Opt TS":
                error="Eigenvalues Info::No negative curvatures at this geometry"
                terminate = True
            elif len(neg_eig_i) > 1 and self.calc_type == "Opt Min":
                error="Eigenvalues Error::Too many negative curvatures at this geometry"
                terminate = True                    
            elif len(neg_eig_i) > 1:
                error="Eigenvalues Error::Too many negative curvatures at this geometry"
                terminate = True
            
            #Optimiser
            disps = np.zeros(3)
            for mode in range(len(eigenvalues)):
                e_val = eigenvalues[mode]
                e_vec = eigenvectors[mode]

                disp = np.dot(np.dot((e_vec.T), -gradient), e_vec) / e_val
                disps += disp

            # update positions
            coord = coord + disps
            
        else: #Dynamics/MEP

            try:
                coord,mom = lepsnorm(coord,mom,masses,gradient,hessian,step_size)
            except LinAlgError:
                error="Surface Error::Energy could not be evaluated at step {}. Positions might be beyond the validity of the surface. Steps truncated".format(itcounter + 1)
                terminate = True

            if calc_type=="MEP":
                # reset momenta to zero if doing a MEP
                mom=np.zeros(3)
            
        if not terminate:
            # Update records
            trajectory.append(np.column_stack((coord,mom)))
    
    # convert to array and return
    return (np.array(trajectory),error)
