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


from lepspoint import leps_energy,cos_rule
from lepsmove import kinetic_energy,velocities,velocity_AC

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

            
def plot_contour(trajectory,x_grid,y_grid,Vmat,cutoff,spacing):
    """Contour Plot"""
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    
    plt.xlabel("AB Distance/pm")
    plt.ylabel("BC Distance/pm")
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    levels = np.arange(np.min(Vmat) -1, cutoff, spacing)
    plt.contour(X, Y, Vmat, levels = levels)
    plt.xlim([min(x_grid),max(x_grid)])
    plt.ylim([min(y_grid),max(y_grid)])
  
    if max(trajectory[:,2,0])-min(trajectory[:,2,0]) < 1e-7:
        plt.plot(trajectory[:,0,0], trajectory[:,1,0], linestyle='', marker='o', markersize=1.5, color='black')

    # highlight initial position
    plt.plot(trajectory[:1,0,0], trajectory[:1,1,0], marker='x', markersize=6, color="red")

    plt.draw()
    plt.pause(0.0001) #This stops matplotlib from blocking


def plot_skew(trajectory,masses,x_grid,y_grid,Vmat,cutoff,spacing):    
    """Skew Plot"""
    #
    #Taken from:
    #Introduction to Quantum Mechanics: A Time-Dependent Perspective
    #Chapter 12.3.3
    #
    #Transform X and Y to Q1 and Q2, where
    #Q1   = a*X + b*Y*cos(beta)
    #Q2   = b*Y*sin(beta)
    #a    = ((m_A * (m_B + m_C)) / (m_A + m_B + m_C)) ** 0.5
    #b    = ((m_C * (m_A + m_B)) / (m_A + m_B + m_C)) ** 0.5
    #beta = cos-1(((m_A * m_C) / ((m_B + m_C) * (m_A + m_B))) ** 0.5)
    #
    #m_i: mass of atom i
    
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    
    plt.xlabel("Q1/$pm.g^{1/2}.mol^{-1/2}$")
    plt.ylabel("Q2/$pm.g^{1/2}.mol^{-1/2}$")
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    ma,mb,mc = masses
    
    a    = ((ma * (mb + mc)) / np.sum(masses)) ** 0.5
    b    = ((mc * (ma + mb)) / np.sum(masses)) ** 0.5
    beta = np.arccos(((ma * mc) / ((mb + mc) * (ma + mb))) ** 0.5)

    #Transform grid
    Q1 = a * X + b * Y * np.cos(beta)
    Q2 = b * Y * np.sin(beta)
    
    #Plot gridlines every 30pm
    splot_grid_x = list(np.arange(x_grid[0], x_grid[-1], 30))+[x_grid[-1]]
    splot_grid_y = list(np.arange(y_grid[0], y_grid[-1], 30))+[y_grid[-1]]
    
    for x in splot_grid_x:
        r1 = [x, splot_grid_y[ 0]]
        r2 = [x, splot_grid_y[-1]]

        q1 = [a * r1[0] + b * r1[1] * np.cos(beta), b * r1[1] * np.sin(beta)]
        q2 = [a * r2[0] + b * r2[1] * np.cos(beta), b * r2[1] * np.sin(beta)]
              
        plt.plot([q1[0], q2[0]], [q1[1], q2[1]], linewidth=1, color='gray')
        plt.text(q2[0], q2[1], str(int(x))) #round label to integer
        
    for y in splot_grid_y:
        r1 = [splot_grid_x[ 0], y]
        r2 = [splot_grid_x[-1], y]

        q1 = [a * r1[0] + b * r1[1] * np.cos(beta), b * r1[1] * np.sin(beta)]
        q2 = [a * r2[0] + b * r2[1] * np.cos(beta), b * r2[1] * np.sin(beta)]
              
        plt.plot([q1[0], q2[0]], [q1[1], q2[1]], lw=1, color='gray')
        plt.text(q2[0], q2[1], str(int(y))) #round label to integer
        
    #Plot transformed PES
    levels = np.arange(np.min(Vmat) -1, cutoff, spacing)
    plt.contour(Q1, Q2, Vmat, levels = levels)
    plt.autoscale()
    plt.axes().set_aspect('equal')

    if max(trajectory[:,2,0])-min(trajectory[:,2,0]) < 1e-7:
        #Plot transformed trajectory
        srab = a * trajectory[:,0,0] + b * trajectory[:,1,0] * np.cos(beta)
        srbc = b * trajectory[:,1,0] * np.sin(beta)
    
        plt.plot(srab, srbc, linestyle='', marker='o', markersize=1.5, color='black')

    # highlight initial position
    plt.plot(srab[0], srbc[0], marker='x', markersize=6, color="red")
  
    plt.draw()
    plt.pause(0.0001)

 
def plot_surface(trajectory,morse_params,sato,x_grid,y_grid,Vmat,cutoff,spacing):
    """3d Surface Plot"""
    
    plt.close('all') #New figure needed for 3D axes
    fig_3d = plt.figure('Surface Plot', figsize=(5,5))
    
    ax = Axes3D(fig_3d)
    
    plt.xlabel("AB Distance/pm")
    plt.ylabel("BC Distance/pm")
    ax.set_zlabel("V/$kJ.mol^{-1}$")
    
    X, Y = np.meshgrid(x_grid, y_grid)
    ax.set_xlim3d([min(x_grid),max(x_grid)])
    ax.set_ylim3d([min(y_grid),max(y_grid)])
    
    Z = np.clip(Vmat, -800, cutoff)
    
    ax.plot_surface(X, Y, Z, rstride=int(spacing)+1, cstride=int(spacing)+1, cmap='jet', alpha=0.3, linewidth=0.25, edgecolor='black')

    levels = np.arange(np.min(Vmat) -1, cutoff, spacing)
    ax.contour(X, Y, Z, zdir='z', levels=levels, offset=ax.get_zlim()[0]-1)

    if max(trajectory[:,2,0])-min(trajectory[:,2,0]) < 1e-7:
        ax.plot(trajectory[:,0,0], trajectory[:,1,0],
                leps_energy(trajectory[:,0,0],trajectory[:,1,0],trajectory[:,2,0],morse_params,sato),
                color='black', linestyle='none', marker='o', markersize=2)
     
    plt.draw()
    plt.pause(0.0001)
    

def plot_ind_vs_t(trajectory,dt,calc_type):
    """Internuclear Distances VS Time"""
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    if calc_type == "Dynamics": 
        xaxis=dt*np.arange(len(trajectory))
        plt.xlabel("Time/fs")
    else:
        xaxis=np.arange(len(trajectory))
        plt.xlabel("Steps")

    plt.ylabel("Distance/pm")

    plt.plot(xaxis, trajectory[:,0,0], label = "A-B")
    plt.plot(xaxis, trajectory[:,1,0], label = "B-C")
    plt.plot(xaxis, cos_rule(trajectory[:,0,0],trajectory[:,1,0],trajectory[:,2,0]), label = "A-C")
    
    plt.legend()
    
    plt.draw()
    plt.pause(0.0001)


def plot_inv_vs_t(trajectory,masses,dt,calc_type):
    """Internuclear velocities VS time"""
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    # calculate velocities
    veloc=[]
    for point in trajectory:
        # velicities in internal coordinates
        internal_veloc=list(velocities(point[:,0],point[:,1],masses))
        # calculate magnitude of veloctity between AC
        vAC=velocity_AC(point[:,0],*internal_veloc[0:2])
        # make list of internuclear velocities
        in_veloc=internal_veloc[0:2]+[vAC]

        veloc.append(in_veloc)

    veloc=np.array(veloc)

    if calc_type == "Dynamics":
        xaxis=dt*np.arange(len(trajectory))
        plt.xlabel("Time/fs")
    else:
        xaxis=np.arange(len(trajectory))
        plt.xlabel("Steps")

    plt.ylabel("$Velocity/pm.fs^{-1}$")

    plt.plot(xaxis, veloc[:,0], label = "A-B")
    plt.plot(xaxis, veloc[:,1], label = "B-C")
    plt.plot(xaxis, veloc[:,2], label = "A-C")

    plt.legend()

    plt.draw()
    plt.pause(0.0001)
  
    
def plot_momenta_vs_t(trajectory,dt,calc_type):
    """Momenta VS Time"""
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    if calc_type == "Dynamics": 
        xaxis=dt*np.arange(len(trajectory))
        plt.xlabel("Time/fs")
    else:
        xaxis=np.arange(len(trajectory))
        plt.xlabel("Steps")

    plt.ylabel("$Momentum/g.mol^{-1}.pm.fs^{-1}$")

    plt.plot(xaxis, trajectory[:,0,1], label = "A-B")
    plt.plot(xaxis, trajectory[:,1,1], label = "B-C")
    plt.plot(xaxis, trajectory[:,2,1], label = "θ")
   
    plt.legend()
    
    plt.draw()
    plt.pause(0.0001)      


def plot_momenta(trajectory):
    """AB Momentum VS BC Momentum"""
    plt.clf()
    ax = plt.gca()
    
    plt.xlabel("AB Momentum/$g.mol^{-1}.pm.fs^{-1}$")
    plt.ylabel("BC Momentum/$g.mol^{-1}.pm.fs^{-1}$")
    
    lc = colorline(trajectory[:,0,1], trajectory[:,1,1], cmap = plt.get_cmap("jet"), linewidth=1)
    
    ax.add_collection(lc)
    ax.autoscale()
    plt.draw()
    plt.pause(0.0001)

 
def plot_velocities(trajectory,masses):
    """AB Velocity VS BC Velocity"""

    # calculate velocities in internal coordinates
    veloc=[]
    for point in trajectory:
        veloc.append(velocities(point[:,0],point[:,1],masses))
    veloc=np.array(veloc)
  
    plt.clf()
    ax = plt.gca()
    
    plt.xlabel("AB Velocity/$pm.fs^{-1}$")
    plt.ylabel("BC Velocity/$pm.fs^{-1}$")
    
    lc = colorline(veloc[:,0], veloc[:,1], cmap = plt.get_cmap("jet"), linewidth=1)
    
    ax.add_collection(lc)
    ax.autoscale()
    plt.draw()
    plt.pause(0.0001)

    
def plot_e_vs_t(trajectory,masses,morse_params,sato,dt,calc_type):
    """Energy VS Time"""
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    if calc_type == "Dynamics": 
        xaxis=dt*np.arange(len(trajectory))
        plt.xlabel("Time/fs")
    else:
        xaxis=np.arange(len(trajectory))
        plt.xlabel("Steps")
   
    plt.ylabel("E/$kJ.mol^{-1}$")

    # calculate energies
    V=np.zeros(len(trajectory))
    K=np.zeros(len(trajectory))
    for i,point in enumerate(trajectory):
        V[i]=leps_energy(*point[:,0],morse_params,sato)
        K[i]=kinetic_energy(point[:,0],point[:,1],masses)

    plt.plot(xaxis, V, label = "Potential Energy")
    plt.plot(xaxis, K, label = "Kinetic Energy")
    plt.plot(xaxis, V+K, label = "Total Energy")
    
    plt.legend()
    
    plt.draw()
    plt.pause(0.0001)


def animation(trajectory,masses,atom_list,atom_map):
    """Animation"""
    plt.close('all')
    ani_fig = plt.figure('Animation', figsize=(5,5))
    
    #Positions in space of A, B and C relative to B
    frames = len(trajectory)
    positions = np.column_stack((- trajectory[:,0,0], np.zeros(frames),
                                 np.zeros(frames), np.zeros(frames),
                                 - np.cos(trajectory[:,2,0]) * trajectory[:,1,0],
                                 np.sin(trajectory[:,2,0]) * trajectory[:,1,0]))
    positions = np.reshape(positions,(frames,3,2))
    
    #Get centre of mass
    com = masses.dot(positions[:])/np.sum(masses)
    
    #Translate to centre of mass (there might be a way to do this only with array operations)
    positions = positions - np.reshape(np.column_stack((com,com,com)),(frames,3,2))

    def init():
        ap, bp, cp = patches
        ax.add_patch(ap)
        ax.add_patch(bp)
        ax.add_patch(cp)
        return ap, bp, cp,
        
    def update(i):
        ap, bp, cp = patches
        ap.center = positions[i,0]
        bp.center = positions[i,1]
        cp.center = positions[i,2]
        return ap, bp, cp,
        
    ax = plt.axes(
    xlim = (min(np.ravel(positions[:,:,0])) - 100, max(np.ravel(positions[:,:,0])) + 100),
    ylim = (min(np.ravel(positions[:,:,1])) - 100, max(np.ravel(positions[:,:,1])) + 100)
    )
    ax.set_aspect('equal')
        
    patches = []
    
    for i,at_name in enumerate(atom_list):
        vdw, col = atom_map[at_name]
        patch = plt.Circle(positions[0,i], vdw * 0.25, color = col)
        patches.append(patch)
    
    anim = FuncAnimation(ani_fig, update, init_func=init, frames=len(positions), repeat=True, interval=5)
   
    # Try to show animation but be cautious about crashes 
    try:
        plt.show()
    except:
        pass


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments   
