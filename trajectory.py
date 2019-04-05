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
from lepspoint import lepspoint
from lepnorm import lepnorm
from numpy.linalg.linalg import LinAlgError
import tkinter.messagebox as msgbox

def get_surface(lepsgui):
        """Get Vmat (potential) for a given set of parameters"""
        lepsgui.get_params()
        
        #Check if params have changed. If not, no need to recalculate
        if lepsgui.old_params == lepsgui.params and not lepsgui._firstrun:
            return

        resl = 0.02 #Resolution
        lepsgui._firstrun = False
        
        #Get grid
        lepsgui.Grid_X = np.arange(lepsgui.params.mina,lepsgui.params.maxa,resl)
        lepsgui.Grid_Y = np.arange(lepsgui.params.minb,lepsgui.params.maxb,resl)
        lepsgui.Vmat = np.zeros((len(lepsgui.Grid_Y), len(lepsgui.Grid_X)))
        
        #Calculate potential for each gridpoint
        for drabcount, drab in enumerate(lepsgui.Grid_X):
            lepsgui.Current_Separations[0] = drab
            for drbccount, drbc in enumerate(lepsgui.Grid_Y):
                lepsgui.Current_Separations[1] = drbc
                
                V = lepspoint(
                    np.deg2rad(lepsgui.theta),
                    lepsgui.params.Dissociation_energies,
                    lepsgui.params.Reqs,
                    lepsgui.params.Morse_Parameters,
                    lepsgui.params.surface_param,
                    lepsgui.Current_Separations,
                    lepsgui.Current_First_derivative,
                    lepsgui.Current_Hessian,
                    derivative=0
                )
                lepsgui.Vmat[drbccount, drabcount] = V

        lepsgui.old_params = lepsgui.params


def get_first(lepsgui):
        """1 step of trajectory to get geometry properties"""
        lepsgui._read_entries()
        lepsgui.get_params()
        
        lepsgui.current_potential_energy = lepspoint(
            np.deg2rad(lepsgui.theta),
            lepsgui.params.Dissociation_energies,
            lepsgui.params.Reqs,
            lepsgui.params.Morse_Parameters,
            lepsgui.params.surface_param,
            lepsgui.Current_Separations,
            lepsgui.Current_First_derivative,
            lepsgui.Current_Hessian,
            2
        )
        
        lepsgui.theta,lepsgui.current_kinetic_energy = lepnorm(
            np.deg2rad(lepsgui.theta),
            lepsgui.Current_Separations,
            lepsgui.Current_First_derivative,
            lepsgui.Current_Hessian,
            lepsgui.Current_Velocities,
            lepsgui.Current_Accelerations,
            lepsgui.params.Masses,
            lepsgui.params.Reduced_masses,
            lepsgui.dt,
            False
        )


def get_trajectory(lepsgui):
        """Get dynamics, MEP or optimisation"""

        dt      = lepsgui.dt    #Time step
        lepsgui.current_time = 0
        
        thetai = np.deg2rad(lepsgui.theta) #Collision Angle
        grad = 2 #Calculating gradients and Hessian

        lepsgui.get_arrays()
        lepsgui.Animation_Positions = []
        lepsgui.append_animation_position()
        
        #Initial Velocity
        lepsgui.Current_Velocities = lepsgui.Current_Momenta / lepsgui.params.Reduced_masses
        lepsgui.Trajectory_Velocities = []
        
        #Initialise outputs
        lepsgui.Trajectory_times = []
        lepsgui.Trajectory_Separations = []
        lepsgui.Trajectory_Velocities = []
        lepsgui.Trajectory_Momenta = []
        lepsgui.Trajectory_Accelerations = []
        lepsgui.Trajectory_First_derivatives = []
        lepsgui.Trajectory_Hessians = []
        lepsgui.Potential_Energies = []
        lepsgui.Kinetic_Energies = []
        lepsgui.Potential_Energies = []
        lepsgui.Total_Energies = []
    
        #Initial conditions
        lepsgui.current_potential_energy = lepspoint(thetai,lepsgui.params.Dissociation_energies,lepsgui.params.Reqs,lepsgui.params.Morse_Parameters,lepsgui.params.surface_param,lepsgui.Current_Separations,lepsgui.Current_First_derivative,lepsgui.Current_Hessian,grad)
        
        lepsgui.add_trajectory_step()

        #Flag to stop appending to output in case of a crash
        terminate = False        

        for itcounter in range(lepsgui.steps):
            if lepsgui.calc_type != "Dynamics":
                lepsgui.Current_Velocities = np.zeros((3))
            
            #Get current potential, forces, and Hessian
            lepsgui.current_potential_energy = lepspoint(thetai,lepsgui.params.Dissociation_energies,lepsgui.params.Reqs,lepsgui.params.Morse_Parameters,lepsgui.params.surface_param,lepsgui.Current_Separations,lepsgui.Current_First_derivative,lepsgui.Current_Hessian,grad)
            
            if lepsgui.calc_type in ["Opt Min", "Opt TS"]: #Optimisation calculations
                
                #Diagonalise Hessian
                eigenvalues, eigenvectors = np.linalg.eig(lepsgui.Current_Hessian)
                
                #Eigenvalue test
                neg_eig_i = [i for i,eig in enumerate(eigenvalues) if eig < -0.01]
                if len(neg_eig_i) == 0 and lepsgui.calc_type == "Opt TS":
                    msgbox.showinfo("Eigenvalues Info", "No negative eigenvalues at this geometry")
                    terminate = True
                elif len(neg_eig_i) == 1 and lepsgui.calc_type == "Opt Min":
                    msgbox.showerror("Eigenvalues Error", "Too many negative eigenvalues at this geometry")
                    terminate = True                    
                elif len(neg_eig_i) > 1:
                    msgbox.showerror("Eigenvalues Error", "Too many negative eigenvalues at this geometry")
                    terminate = True
                
                #Optimiser
                Displacements = np.array([0.,0.,0.])
                for mode in range(len(eigenvalues)):
                    e_val = eigenvalues[mode]
                    e_vec = eigenvectors[mode]

                    disp = np.dot(np.dot((e_vec.T), lepsgui.Current_First_derivative), e_vec) / e_val
                    Displacements += disp
                    
                lepsgui.Current_Separations += Displacements
                #xrabf = xrabi + disps[0]
                #xrbcf = xrbci + disps[1]
                #xracf = ((xrabf ** 2) + (xrbcf ** 2) - 2 * xrabf * xrbcf * np.cos(thetai)) ** 0.5
                
            else: #Dynamics/MEP
                try:
                    lepsgui.theta,lepsgui.current_kinetic_energy = lepnorm(
                        lepsgui.theta,
                        lepsgui.Current_Separations,
                        lepsgui.Current_First_derivative,
                        lepsgui.Current_Hessian,
                        lepsgui.Current_Velocities,
                        lepsgui.Current_Accelerations,
                        lepsgui.params.Masses,
                        lepsgui.params.Reduced_masses,
                        lepsgui.dt,
                        False
                    )
                    lepsgui.current_time += lepsgui.dt

                except LinAlgError:
                    msgbox.showerror("Surface Error", "Energy could not be evaulated at step {}. Steps truncated".format(itcounter + 1))
                    terminate = True
                
            if lepsgui.Current_Separations[0] > lepsgui.lim or lepsgui.Current_Separations[1] > lepsgui.lim: #Stop calc if distance lim is exceeded
                msgbox.showerror("Surface Error", "Surface Limits exceeded at step {}. Steps truncated".format(itcounter + 1))
                terminate = True
                    
            if lepsgui.calc_type != "Dynamics":
                lepsgui.Current_Velocities = np.zeros((3))

            if itcounter != lepsgui.steps - 1 and not terminate:
                
                #As above
                lepsgui.append_animation_position()
                
                ### Maybe not necessary
                #Get A-C Velocity
                r0 = np.linalg.norm(lepsgui.Trajectory_Separations[-1][2] - lepsgui.Trajectory_Separations[-1][0])
                r1 = np.linalg.norm(lepsgui.Current_Separations[2] - lepsgui.Current_Separations[0])
                vrac = (r1 - r0) / dt

                lepsgui.add_trajectory_step()
            
            if terminate:
                break