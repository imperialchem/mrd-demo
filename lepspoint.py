#Created on Wed May 17 10:41:56 2017
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


# This file contains functions that calculate the London-Eyring-Polanyi-Sato (LEPS) 
# potential for a set of internal coordinates (2 bond distances and a bond angle),
# as well and the gradient and hessian of the surface with respect to the internal
# coordinates at that point of the surface.


import numpy as np

# Check consistency with H parameter used
k = 0.18 # In TRIATOMICS this is 0.18 but should be Sato parameter?
#this state variable looks strange
state = 1 # 1 for ground state, >1 for excited states


def cos_rule(side1,side2,angle):
    '''
    Use the cos rule to calculate the length of the side of a triangle,
    given the other 2 side lengths and the angle between them.
    '''

    return  (side1**2 + side2**2 - 2*side1*side2*np.cos(angle))**0.5


def _morse(r,D,B,re):
    '''Morse potential with a dissociation limit at 0.'''

    return D*(np.exp(-2*B*(r-re)) - 2*np.exp(-B*(r-re)))


def _morse_deriv1(r,D,B,re):
    '''First derivative of the Morse potential with respect to r.'''

    return -2*B*D*(np.exp(-2*B*(r-re)) - np.exp(-B*(r-re)))


def _morse_deriv2(r,D,B,re):
    '''Second derivative of the Morse potential with respect to r.'''

    return 2 * B**2 * D*(2*np.exp(-2*B*(r-re)) - np.exp(-B*(r-re)))


def _anti_morse(r,D,B,re):
    '''Potential with functional form similar to Morse, used as the triplet
    component of the LEPS potential.'''

    return 0.5*D*(np.exp(-2*B*(r-re)) + 2*np.exp(-B*(r-re))) 


def _anti_morse_deriv1(r,D,B,re):
    '''First derivative of the anti-Morse potential with respect to r.'''

    return -B*D*(np.exp(-2*B*(r-re)) + np.exp(-B*(r-re)))


def _anti_morse_deriv2(r,D,B,re):
    '''Second derivative of the anti-Morse potential with respect to r.'''

    return B**2 * D*(2*np.exp(-2*B*(r-re)) + np.exp(-B*(r-re)))


def _coulomb(morse,anti_morse,k):
    '''Calculate Coulomb (Q) integral in the LEPS approximation.
    The function is also used to compute the derivatives of Q since it is linear
    on the morse and anti_morse components.'''

    return 0.5*(morse + anti_morse + k*(morse - anti_morse))


def _exchange(morse,anti_morse,k):
    '''Calculate Exchange (J) integral in the LEPS approximation.
    The function is also used to compute the derivatives of J since it is linear
    on the morse and anti_morse components.'''

    return 0.5*(morse - anti_morse + k*(morse + anti_morse))


def leps_energy(int_coord,params,H):
    '''Calculate LEPS potential energy for a given point in internal coordinates
       int_coord=array([rAB,rBC,theta])
       params=array([[D_A,B_A,re_A],
                     [D_B.B_B,re_B],
                     [D_C,B_C,re_C]]).'''

    #Build array with distances rAB,rBC and rAC
    r=np.array([int_coord[0],int_coord[1],cos_rule(*int_coord)])
 
   #Coulomb and Exchange integrals
    Q=_coulomb(_morse(r,params[:,0],params[:,1],params[:,2]),
               _anti_morse(r,params[:,0],params[:,1],params[:,2]),k)
    J=_exchange(_morse(r,params[:,0],params[:,1],params[:,2]),
                _anti_morse(r,params[:,0],params[:,1],params[:,2]),k)
   
    return 1/(1+H**2) * (np.sum(Q) - state/2**0.5 *np.linalg.norm(J - np.roll(J,1)))


def leps_gradient(int_coord,params,H):
    '''Calculates the gradient of LEPS potential for a given point in internal coordinates
       int_coord=array([rAB,rBC,theta])
       params=array([[D_A,B_A,re_A],
                     [D_B.B_B,re_B],
                     [D_C,B_C,re_C]]).
       The gradient is given in internal coordinates:
       grad=array([dV/drAB,dV/drBC,dV/dtheta]).'''

    #Build array with distances rAB,rBC and rAC
    r=np.array([int_coord[0],int_coord[1],cos_rule(*int_coord)])

    #Exchange integrals (Coulomb not needed)
    J=_exchange(_morse(r,params[:,0],params[:,1],params[:,2]),
                _anti_morse(r,params[:,0],params[:,1],params[:,2]),k)

    #Partial derivative of Coulomb and Exchange integrals with respect to inter-atomic distances
    #this uses _coulomb() and _exchange() functions because they are linear functions
    partial_Q_r=_coulomb(_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),
                         _anti_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),k)
    partial_J_r=_exchange(_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),
                          _anti_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),k)

    #Note that Q_AC and J_AC depend on rAB and rBC. Calculated eg. dQ_AC/drAB = dQ_AC/drAC * drAC/drAB
    #Make array of the derivative of rAC with respect to the internal coordinate
    theta=int_coord[2]
    drACdint=np.array([r[0] - r[1]*np.cos(theta),r[1] - r[0]*np.cos(theta),r[0] * r[1]*np.sin(theta)])/r[2]

    #Calculate derivative of the Coulombic part of the potential with respect to the internal coordinates.
    #Only Q_AC depends on theta
    Qpart_grad_int=np.array([partial_Q_r[0],partial_Q_r[1],0]) + partial_Q_r[2] * drACdint
                    
    #Calculate derivative of the Exchange part of the potential with respect to the internal coordinates.
    #Only J_AC depends on theta
    Jdiff = J - np.roll(J,-1)
 
    Jpart_grad_rAB=np.sum(Jdiff*(np.array([1,0,-1])*partial_J_r[0]+np.array([0,-1,1])*partial_J_r[2]*drACdint[0]))
    Jpart_grad_rBC=np.sum(Jdiff*(np.array([-1,1,0])*partial_J_r[1]+np.array([0,-1,1])*partial_J_r[2]*drACdint[1]))
    Jpart_grad_theta=np.sum(Jdiff*np.array([0,-1,1])*partial_J_r[2]*drACdint[2])
 
    Jpart_grad_int=-state/(2**0.5 * np.linalg.norm(Jdiff)) * np.array([Jpart_grad_rAB,Jpart_grad_rBC,Jpart_grad_theta])
    
    return 1/(1+H**2) * (Qpart_grad_int + Jpart_grad_int)


def leps_hessian(int_coord,params,H):
    '''Calculates the Hessian of LEPS potential for a given point in internal coordinates
       int_coord=array([rAB,rBC,theta])
       params=array([[D_A,B_A,re_A],
                     [D_B.B_B,re_B],
                     [D_C,B_C,re_C]]).
       The gradient is given in internal coordinates,'''

    #Build array with distances rAB,rBC and rAC
    r=np.array([int_coord[0],int_coord[1],cos_rule(*int_coord)])

    #Exchange integrals (Coulomb not needed)
    J=_exchange(_morse(r,params[:,0],params[:,1],params[:,2]),
                _anti_morse(r,params[:,0],params[:,1],params[:,2]),k)

    #Partial first and second derivatives of Coulomb and Exchange integrals with respect to inter-atomic distances
    #this uses _coulomb() and _exchange() functions because they are linear functions
    partial_Q_r=_coulomb(_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),
                         _anti_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),k)
    partial_J_r=_exchange(_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),
                          _anti_morse_deriv1(r,params[:,0],params[:,1],params[:,2]),k)
    partial2_Q_r=_coulomb(_morse_deriv2(r,params[:,0],params[:,1],params[:,2]),
                          _anti_morse_deriv2(r,params[:,0],params[:,1],params[:,2]),k)
    partial2_J_r=_exchange(_morse_deriv2(r,params[:,0],params[:,1],params[:,2]),
                           _anti_morse_deriv2(r,params[:,0],params[:,1],params[:,2]),k)

    #Note that Q_AC and J_AC depend on rAB and rBC.
    #Make array of first derivatives of rAC with respect to the internal coordinate
    theta=int_coord[2]
    drACdint=np.array([r[0] - r[1]*np.cos(theta),r[1] - r[0]*np.cos(theta),r[0] * r[1]*np.sin(theta)])/r[2]

    #Make matrix of second derivatives of rAC with respect to the internal coordinate
    #d2rAC/dint2=array([[d2rAC/drAB2,d2rAC/(drABdrBC),d2rAC/(drABdtheta)],
    #                   [d2rAC/(drABdrBC),d2rAC/drBC2,d2rAC/(drBCdtheta)],
    #                   [d2rAC/(drABdtheta),d2rAC/(drBCdtheta),d2rAC/dtheta2]])
    d2rACdint2=np.array([[r[1]**2 * np.sin(theta)**2,-r[0]*r[1]*np.sin(theta)**2,r[1]**2 * np.sin(theta) * (r[1]-r[0]*np.cos(theta))],
                         [-r[0]*r[1]*np.sin(theta)**2,r[0]**2 * np.sin(theta)**2,r[0]**2 * np.sin(theta) * (r[0]-r[1]*np.cos(theta))],
                         [r[1]**2 * np.sin(theta) * (r[1]-r[0]*np.cos(theta)),r[0]**2 * np.sin(theta) * (r[0]-r[1]*np.cos(theta)),r[0]*r[1]*r[2]**2*np.cos(theta) - r[0]**2 * r[1]**2 *np.sin(theta)**2]]) / (r[2]**3)

    #Calculate Coulombic contribution to the Hessian
    Qpart_hess_int=np.diag(np.array([partial2_Q_r[0],partial2_Q_r[1],0])) + \
                   partial_Q_r[2] * d2rACdint2 + \
                   partial2_Q_r[2] * (drACdint * np.expand_dims(drACdint, axis=1)) #this last line is using broadcasting of 2 vectors to give a matrix

    #Calculate Exchange contribution to the Hessian
    #Devide into 2 parts.
    Jdiff = J - np.roll(J,-1)
    Jdiff_norm= np.linalg.norm(Jdiff)

    Jpart1_hess_drAB2=np.sum(Jdiff*(np.array([1,0,-1])*partial_J_r[0]+np.array([0,-1,1])*partial_J_r[2]*drACdint[0]))**2
    Jpart1_hess_drABdrBC=np.sum(Jdiff*(np.array([1,0,-1])*partial_J_r[0]+np.array([0,-1,1])*partial_J_r[2]*drACdint[0])) * \
                         np.sum(Jdiff*(np.array([-1,1,0])*partial_J_r[1]+np.array([0,-1,1])*partial_J_r[2]*drACdint[1]))
    Jpart1_hess_drABdtheta=np.sum(Jdiff*(np.array([1,0,-1])*partial_J_r[0]+np.array([0,-1,1])*partial_J_r[2]*drACdint[0])) * \
                           np.sum(Jdiff*np.array([0,-1,1])*partial_J_r[2]*drACdint[2])
    Jpart1_hess_drBC2=np.sum(Jdiff*(np.array([-1,1,0])*partial_J_r[1]+np.array([0,-1,1])*partial_J_r[2]*drACdint[1]))**2
    Jpart1_hess_drBCdtheta=np.sum(Jdiff*(np.array([-1,1,0])*partial_J_r[1]+np.array([0,-1,1])*partial_J_r[2]*drACdint[1])) * \
                           np.sum(Jdiff*np.array([0,-1,1])*partial_J_r[2]*drACdint[2])
    Jpart1_hess_dtheta2=np.sum(Jdiff*np.array([0,-1,1])*partial_J_r[2]*drACdint[2])**2

    Jpart1_hess_int=-1/Jdiff_norm**3 * np.array([[Jpart1_hess_drAB2,Jpart1_hess_drABdrBC,Jpart1_hess_drABdtheta],
                                                   [Jpart1_hess_drABdrBC,Jpart1_hess_drBC2,Jpart1_hess_drBCdtheta],
                                                   [Jpart1_hess_drABdtheta,Jpart1_hess_drBCdtheta,Jpart1_hess_dtheta2]])

    Jpart2_hess_drAB2=np.sum(Jdiff*(np.array([1,0,-1])*partial2_J_r[0]+np.array([0,-1,1])*(partial2_J_r[2]*drACdint[0]**2+partial_J_r[2]*d2rACdint2[0,0]))) + \
                      2*(partial_J_r[0]**2+(partial_J_r[2]*drACdint[0])**2-partial_J_r[0]*partial_J_r[2]*drACdint[0])
    Jpart2_hess_drABdrBC=np.sum(Jdiff*np.array([0,-1,1])*(partial2_J_r[2]*drACdint[0]*drACdint[1]+partial_J_r[2]*d2rACdint2[0,1])) + \
                         (-partial_J_r[0]*partial_J_r[1]-partial_J_r[0]*partial_J_r[2]*drACdint[1]-partial_J_r[1]*partial_J_r[2]*drACdint[0] + \
                          2*partial_J_r[2]**2*drACdint[0]*drACdint[1])
    Jpart2_hess_drABdtheta=np.sum(Jdiff*np.array([0,-1,1])*(partial2_J_r[2]*drACdint[0]*drACdint[2]+partial_J_r[2]*d2rACdint2[0,2])) + \
                           (-partial_J_r[0]*partial_J_r[2]*drACdint[2] + 2*partial_J_r[2]**2*drACdint[0]*drACdint[2])
    Jpart2_hess_drBC2=np.sum(Jdiff*(np.array([-1,1,0])*partial2_J_r[1]+np.array([0,-1,1])*(partial2_J_r[2]*drACdint[1]**2+partial_J_r[2]*d2rACdint2[1,1]))) + \
                      2*(partial_J_r[1]**2+(partial_J_r[2]*drACdint[1])**2-partial_J_r[1]*partial_J_r[2]*drACdint[1])
    Jpart2_hess_drBCdtheta=np.sum(Jdiff*np.array([0,-1,1])*(partial2_J_r[2]*drACdint[1]*drACdint[2]+partial_J_r[2]*d2rACdint2[1,2])) + \
                           (-partial_J_r[1]*partial_J_r[2]*drACdint[2] + 2*partial_J_r[2]**2*drACdint[1]*drACdint[2])
    Jpart2_hess_dtheta2=np.sum(Jdiff*np.array([0,-1,1])*(partial2_J_r[2]*drACdint[2]**2+partial_J_r[2]*d2rACdint2[2,2])) + \
                        2*(partial_J_r[2]*drACdint[2])**2

    Jpart2_hess_int=1/Jdiff_norm * np.array([[Jpart2_hess_drAB2,Jpart2_hess_drABdrBC,Jpart2_hess_drABdtheta],
                                             [Jpart2_hess_drABdrBC,Jpart2_hess_drBC2,Jpart2_hess_drBCdtheta],
                                             [Jpart2_hess_drABdtheta,Jpart2_hess_drBCdtheta,Jpart2_hess_dtheta2]])

    return 1/(1+H**2)*(Qpart_hess_int + (-state/2**0.5 * (Jpart1_hess_int+Jpart2_hess_int)))

