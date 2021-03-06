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


# There are problems with energy conservation when theta is not 180 degrees.
# This might be due to energy transfer into/from rotational motion which is
# not properly described in this treatment (no Coriolis coupling).
# The variation in total energy seems significant, so a bug is not to be rules out.

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

    # Transform into normal modes and do dynamics in mass-weighted normal modes
    # The transformation does not seem so trivial. Has similarities with the Miyazawa 
    # method described in: S. Caligano "Vibrational States" John Wiley & Sons (1976) sec. 4.6
    # but it is not the same

    GM=_gmat(coord,masses)
    
    GMVal, GMVec = np.linalg.eigh(GM)

    GMVal1 = GMVal ** 0.5
    GMVal2 = GMVal ** (-0.5)

    GRR    = GMVec.dot(np.diag(GMVal1)).dot(GMVec.T)
    GROOT  = GMVec.dot(np.diag(GMVal2)).dot(GMVec.T)

    # G-Matrix Weighted Hessian;
    mwhessian = GRR.dot(hessian).dot(GRR)
    w2, ALT = np.linalg.eigh(mwhessian) #ALT is antisymmetric version in Fort code but that does not give the right G-Matrix!!!!
    
    # Gradient Vector in mass-weighted normal modes
    gradN = ALT.T.dot(GRR).dot(gradient)
    
    # Momentum Vector in normal modes
    momN = ALT.T.dot(GRR).dot(mom)

    # The dynmics algorithm in mass-weighted normal modes follows:
    # T. Helgaker, E. Uggerud, H.J. Aa. Jensen, Chem. Phys Lett. 173(2,3):145-150 (1990)
   
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
    coord = coord + GRR.dot(ALT).dot(displacementN) 
    
    # transform updated momentum back into internal coordinates
    mom = GROOT.dot(ALT).dot(momN)
            
    return (coord,mom)


def calc_trajectory(coord_init,mom_init,masses,morse_params,sato,steps,dt,calc_type):
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

    # If not doing Dynamics (MEP, Opt Min or Opt TS), set initial momenta to zero
    if calc_type != "Dynamics":
        mom=np.zeros(3)

        # If doing a MEP increase step size to compensate absence of inertial term
        if calc_type == "MEP":
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
        #(array unpacking *coord used below only works for python 3.5 or higher)
        gradient = leps_gradient(*coord,morse_params,sato)
        hessian = leps_hessian(*coord,morse_params,sato)

        if calc_type in ["Opt Min", "Opt TS"]: #Optimisation calculations
            
            #Diagonalise Hessian
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            
            #Eigenvalue test
            neg_eig = [eig for eig in eigenvalues if eig < -1e-14]
            if len(neg_eig) == 0 and calc_type == "Opt TS":
                error="Eigenvalues Info::No negative curvatures at this geometry"
                terminate = True
            elif len(neg_eig) > 0 and calc_type == "Opt Min":
                error="Eigenvalues Error::Too many negative curvatures at this geometry"
                terminate = True                    
            elif len(neg_eig) > 1:
                error="Eigenvalues Error::Too many negative curvatures at this geometry"
                terminate = True
            
            #Optimiser
            disps = np.zeros(3)
            for mode in range(len(eigenvalues)):
                e_val = eigenvalues[mode]
                e_vec = eigenvectors[:,mode] #eigenvectors are the columns

                #disp is a vector with the same direction as e_vec
                disp = np.dot(e_vec, -gradient) * e_vec / e_val
                disps += disp

            # update positions
            # use dt to scale the step
            coord = coord + dt*disps

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
