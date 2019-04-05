#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:59:17 2017

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

from params import Params
from lepspoint import lepspoint
from lepnorm import lepnorm
from trajectory import get_trajectory, get_first, get_surface

import numpy as np
from numpy.linalg.linalg import LinAlgError
import copy

from configparser import ConfigParser

import tkinter as tk
import tkinter.messagebox as msgbox
from tkinter.filedialog import asksaveasfilename

from matplotlib import use as mpl_use
mpl_use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings

from argparse import ArgumentParser

class Interactive():
    
    def __init__(self, advanced=False): #Initialise Class
        
        ###Initialise tkinter###
        self.root = tk.Tk()
        self.root.title("LEPS GUI")
        self.root.resizable(0,0)
        
        ###Initialise defaults###
        config = ConfigParser(inline_comment_prefixes=(';', '#'))
        config.read('params.ini')
        
        #Atom: [Index, VdW radius, colour]
        #Index      - index in dropdown list in selection
        #VdW radius - used for animation
        #Colour     - used for animation
        atom_map = {}
        atoms = config['atoms']
        self.atoms_list = []
        for k, l in atoms.items():
            mass, vdw, colour, name = l.split(',')
            atom_map[name.strip()] = [
                int(k),
                float(vdw),
                '#' + colour.strip()            
            ]
            self.atoms_list.append(name.strip())
        self.atom_map = atom_map
        
        defaults = config['defaults']
        self.dt  = float(defaults['dt'])  #Time step in dynamics trajectory
        self.lim = float(defaults['lim']) #Calculation will stop once this distance is exceeded

        self.Vmat = None       #Array where potential is stored for each gridpoint
        self.old_params = None #Variable used to prevent surface being recalculated

        self.theta = None
        self.steps = None
        self.params = None
        
        self.a = 1
        self.b = 1
        self.c = 1
        self.xrabi = 0
        self.xrbci = 0
        self.xraci = 0
        self.xrabi = 0
        self.xrbci = 0
        self.prabi = 0
        self.prbci = 0
        self.steps = 0
        self.cutoff = 0
        self.spacing = 0
        self.calc_type = ""
        self.theta = 0
        self.plot_type = ""

        self.entries  = {} #Dictionary of entries to be read on refresh (user input)
        self.defaults = {  #Defaults for each entry
           #Key        : Default value  , type , processing function
            "a"        : ["H"           , str  , lambda x: self.atom_map[x][0]],
            "b"        : ["H"           , str  , lambda x: self.atom_map[x][0]],
            "c"        : ["H"           , str  , lambda x: self.atom_map[x][0]],
            "xrabi"    : ["2.3"         , float, None                         ],
            "xrbci"    : ["0.74"        , float, None                         ],
            "prabi"    : ["-2.5"        , float, None                         ],
            "prbci"    : ["-1.5"        , float, None                         ],
            "steps"    : ["500"         , int  , lambda x: max(1, x)          ],
            "cutoff"   : ["-20"         , float, None                         ],
            "spacing"  : ["5"           , int  , None                         ],
            "calc_type": ["Dynamics"    , str  , None                         ],
            "theta"    : ["180"         , float, None                         ],
            "plot_type": ["Contour Plot", str  , None                         ]
        }
        
        #Store variable as class attributes
        for key, l in self.defaults.items():
            val, vtype, procfunc = l
            val = vtype(val)
            if procfunc: #Check whether processing is needed
                val = procfunc(val)
            setattr(self, key, val)

        self.xraci = ((self.xrabi ** 2) + (self.xrbci ** 2) - 2 * self.xrabi * self.xrbci * np.cos(self.theta)) ** 0.5

        #Arrays
        self.Current_Separations = np.zeros((3))
        self.Animation_Positions = []
        self.Trajectory_Separations = []

        self.Current_Momenta = np.zeros((3))
        self.Trajectory_Momenta = []

        self.Current_First_derivative = np.zeros((3))
        self.Trajectory_First_derivatives = []

        self.Current_Hessian = np.zeros((3,3))
        self.Trajectory_Hessians = []

        self.Current_Accelerations = np.zeros((3))
        self.Trajectory_Accelerations = []

        self.current_time = 0
        self.Trajectory_times = []

        self.Current_Velocities = np.zeros((3))
        self.Trajectory_Velocities = []

        self.current_potential_energy = 0
        self.Potential_Energies = []

        self.current_kinetic_energy = 0
        self.Kinetic_Energies = []

        self.current_total_energy = 0
        self.Total_Energies = []

        self.get_arrays()
        
        #This is needed to allow surface to be calculated on the first run
        self._firstrun = True
        
        ###GUI###
        
        #Default frame format
        sunken = dict(height = 2, bd = 1, relief = "sunken")
        def gk(string):
            grid   = "".join([s for s in string if s.isdigit()])
            sticky = "".join([s for s in string if s in "news"])
            grid = grid.ljust(6, '0')
            r,c,rs,cs,px,py = [int(s) for s in grid]
            g = {"row": r, "column": c}
            if rs: g["rowspan"]    = rs
            if cs: g["columnspan"] = cs
            if px: g["padx"]       = px
            if py: g["pady"]       = px

            if sticky: g["sticky"]   = sticky
            
            return g
        
        #Atoms Selection Frame
        selection_frame = self._add_frame(dict(master=self.root, text="Atoms", **sunken), gk('002055news'))
        
        self._add_label(selection_frame, {"text": "Atom A:"}, gk('00'))
        self._add_label(selection_frame, {"text": "Atom B:"}, gk('10'))
        self._add_label(selection_frame, {"text": "Atom C:"}, gk('20'))
        
        self._add_optionmenu(selection_frame, "a", self.atoms_list, {}, gk('01'))
        self._add_optionmenu(selection_frame, "b", self.atoms_list, {}, gk('11'))
        self._add_optionmenu(selection_frame, "c", self.atoms_list, {}, gk('21'))
        
        #Initial Conditions Frame
        values_frame = self._add_frame(dict(master=self.root, text="Initial Conditions", **sunken), gk('202055news'))
        
        self._add_label(values_frame, {"text": "AB Distance:     "}, gk('00'))
        self._add_label(values_frame, {"text": "BC Distance:     "}, gk('10'))
        self._add_label(values_frame, {"text": "AB Momentum:   "  }, gk('20'))
        self._add_label(values_frame, {"text": "BC Momentum:   "  }, gk('30'))
        
        self._add_entry(values_frame, "xrabi", {}, gk('01'), {"width":10}, self.update_geometry_info)
        self._add_entry(values_frame, "xrbci", {}, gk('11'), {"width":10}, self.update_geometry_info)
        self._add_entry(values_frame, "prabi", {}, gk('21'), {"width":10}, self.update_geometry_info)
        self._add_entry(values_frame, "prbci", {}, gk('31'), {"width":10}, self.update_geometry_info)
        
        #Angle Frame
        angle_frame = self._add_frame(dict(master=self.root, text="Collision Angle", **sunken), gk('40news'))
        
        self._add_scale(angle_frame, "theta", {"from_":0, "to":180, "orient":"horizontal"}, gk('00ew'), {"length":200})
        
        #Update and Export
        update_frame = self._add_frame(dict(master=self.root, **sunken), gk('500355news'))
        self._add_button(update_frame, {"text": "Update Plot"}      , gk('000055'), {"<Button-1>": self.update_plot })
        self._add_button(update_frame, {"text": "Get Last Geometry"}, gk('010055'), {"<Button-1>": self.get_last_geo})
        self._add_button(update_frame, {"text": "Export Data"}      , gk('020055'), {"<Button-1>": self.export      })
        
        #Calculation Type Frame
        calc_type_frame = self._add_frame(dict(master=self.root, text="Calculation Type", **sunken), gk('010055news'))
        
        if advanced:
            calc_types = [ "Dynamics", "MEP", "Opt TS", "Opt Min"]
        else:
            calc_types = [ "Dynamics", "MEP"]
        
        self._add_optionmenu(calc_type_frame, "calc_type", calc_types, {}, gk('00'), {"width":20})
        
        #Plot Type Frame
        type_frame = self._add_frame(dict(master=self.root, text="Plot Type", **sunken), gk('110055news'))
        
        if advanced:
            plot_types = ["Contour Plot", "Skew Plot", "Surface Plot", "Internuclear Distances vs Time", 
                "Internuclear Current_Momenta vs Time", "Energy vs Time", "p(AB) vs p(BC)", "v(AB) vs v(BC)", "Animation"]     
        else:
            plot_types = ["Contour Plot", "Skew Plot", "Surface Plot", "Internuclear Distances vs Time", 
                "Internuclear Current_Momenta vs Time", "Energy vs Time", "Animation"]     
        
        self._add_optionmenu(type_frame, "plot_type", plot_types , {}, gk('00'), {"width":20})
        
        #Steps Frame
        steps_frame = self._add_frame(dict(master=self.root, text="Steps", **sunken), gk('210055news'))
        self._add_entry(steps_frame, "steps", {}, {"row":0, "column":0}, {"width":6})
        
        #Cutoff Frame
        cutoff_frame = self._add_frame(dict(master=self.root, text="Cutoff (Kcal/mol)", **sunken), gk('310055news'))
        self._add_scale(cutoff_frame, "cutoff",{"from_":-100, "to":0, "orient":"horizontal"}, gk('00ew'), {"length":200})
        
        #Contour Spacing Frame
        spacing_frame = self._add_frame(dict(master=self.root, text="Contour Spacing", **sunken), gk('410055news'))
        self._add_scale( spacing_frame, "spacing", {"from_":1, "to":10, "orient":"horizontal"}, gk('00ew'), {"length":200})
        
        #Geometry Info Frame
        
        geometry_frame = self._add_frame(dict(master=self.root, text="Initial Geometry Information", **sunken), gk('025055news'))
        
        self._add_button(geometry_frame, {"text": "Refresh"}, gk('000055'), {"<Button-1>": self.update_geometry_info})
        
        energy_frame = self._add_frame(dict(master=geometry_frame, text="Energy", **sunken), gk('100055news'))
        self._add_label(energy_frame, {"text": "Kinetic:   "}, gk('00'))
        self._add_label(energy_frame, {"text": "Potential: "}, gk('01'))
        self._add_label(energy_frame, {"text": "Total:     "}, gk('02')) 
        
        self.i_ke   = self._add_label(energy_frame, {"text": ""}, gk('10'))
        self.i_pe   = self._add_label(energy_frame, {"text": ""}, gk('11'))
        self.i_etot = self._add_label(energy_frame, {"text": ""}, gk('12'))
        
        forces_frame = self._add_frame(dict(master=geometry_frame, text="Forces", **sunken), gk('200055news'))
        self._add_label(forces_frame, {"text": "AB:        "}, gk('00'))
        self._add_label(forces_frame, {"text": "BC:        "}, gk('01'))
        self._add_label(forces_frame, {"text": "Total:     "}, gk('02'))
        
        self.i_fab  = self._add_label(forces_frame, {"text": ""}, gk('10'))
        self.i_fbc  = self._add_label(forces_frame, {"text": ""}, gk('11'))
        self.i_ftot = self._add_label(forces_frame, {"text": ""}, gk('12'))
        
        hessian_frame = self._add_frame(dict(master=geometry_frame, text="Hessian", **sunken), gk('300055news'))
        self._add_label(hessian_frame, {"text": "1:         "}, gk('01'))
        self._add_label(hessian_frame, {"text": "2:         "}, gk('02'))
        self._add_label(hessian_frame, {"text": "Eigenvalue:"}, gk('10'))
        self._add_label(hessian_frame, {"text": "AB Vector: "}, gk('20'))
        self._add_label(hessian_frame, {"text": "BC Vector: "}, gk('30'))
        
        self.i_eval1 = self._add_label(hessian_frame, {"text": ""}, gk('11'))
        self.i_eval2 = self._add_label(hessian_frame, {"text": ""}, gk('12'))
        
        self.i_evec11 = self._add_label(hessian_frame, {"text": ""}, gk('21'))
        self.i_evec12 = self._add_label(hessian_frame, {"text": ""}, gk('22'))
        self.i_evec21 = self._add_label(hessian_frame, {"text": ""}, gk('31'))
        self.i_evec22 = self._add_label(hessian_frame, {"text": ""}, gk('32'))
        
        self._add_button(geometry_frame, {"text": "Plot"}, gk('400055'), {"<Button-1>": self.plot_eigen})
        
        ###First Run###
        
        # Initialise params and info
        self.get_params()
        self.update_geometry_info()
        
        #Plot
        warnings.filterwarnings("ignore")
        self.fig = plt.figure('Plot', figsize=(5,5))
        self.update_plot()
        
        #Make sure all plots are closed on exit
        def cl():            
            plt.close('all')
            self.root.destroy()
            
        self.root.protocol("WM_DELETE_WINDOW", cl)
        self.root.mainloop()
        
    def _read_entries(self): 
        """Read entries from GUI, process and set attributes"""
        for key, l in self.entries.items():
            entry, type, procfunc = l
            try:
                val = self._cast(entry, type)
                if procfunc:
                    val = procfunc(val)
                setattr(self, key, val)
            except:
                pass
            
    def _cast(self, entry, type): 
        """Read entry and cast to type"""
        val = type(entry.get())
        return val
            
    def _add_frame(self, frame_kwargs={}, grid_kwargs={}):
        """Insert a frame (box) into parent.
        With text, a labelled frame is used"""
        
        if "text" in frame_kwargs:
            frame = tk.LabelFrame(**frame_kwargs)
        else:
            frame = tk.Frame(**frame_kwargs)
            
        frame.grid(**grid_kwargs)
        return frame
        
    def _add_label(self, frame, text_kwargs={}, grid_kwargs={}, config_kwargs={}):
        """Insert a label"""
        label = tk.Label(frame, **text_kwargs)
        label.grid(**grid_kwargs)
        label.config(**config_kwargs)
        return label
        
    def _add_scale(self, frame, key, scale_kwargs={}, grid_kwargs={}, config_kwargs={}):
        """Insert a scrollable bar"""
        val, vtype, procfunc = self.defaults[key]
        variable = tk.StringVar()
        variable.set(val)
        
        scale = tk.Scale(frame, **scale_kwargs)
        scale.set(variable.get())
        scale.grid(**grid_kwargs)
        scale.config(**config_kwargs)
        scale.grid_columnconfigure(0, weight = 1)
        
        self.entries[key] = [scale, vtype, procfunc]

    def _add_button(self, frame, button_kwargs={}, grid_kwargs={}, bind_kwargs={}, config_kwargs={}):
        "Insert a button"""
        button = tk.Button(frame, **button_kwargs)
        button.grid(**grid_kwargs)
        for k, v in bind_kwargs.items():
            button.bind(k, v)
        button.config(**config_kwargs)
        
    def _add_entry(self, frame, key, entry_kwargs={}, grid_kwargs={}, config_kwargs={}, attach_func=None):
        """Add a text entry"""
        val, vtype, procfunc = self.defaults[key]
        variable = tk.StringVar()
        variable.set(val)
        if attach_func:
            variable.trace("w", attach_func)
        
        entry = tk.Entry(frame, textvariable=variable, **entry_kwargs)
        entry.grid(**grid_kwargs)
        entry.config(**config_kwargs)
        
        self.entries[key] = [entry, vtype, procfunc]
        
    def _add_optionmenu(self, frame, key, items, optionmenu_kwargs={}, grid_kwargs={}, config_kwargs={}):
        """Add a dropdown menu"""
        val, vtype, procfunc = self.defaults[key]
        variable = tk.StringVar()
        variable.set(val)
        
        optionmenu = tk.OptionMenu(frame, variable, *items, **optionmenu_kwargs)
        optionmenu.grid(**grid_kwargs)
        optionmenu.config(**config_kwargs)
        
        self.entries[key] = [variable, vtype, procfunc]
        
    def _add_radio(self, frame, key, radio_kwargs={}, grid_kwargs={}, config_kwargs={}, variable=None):
        """Add a radio button"""
        val, vtype, procfunc = self.defaults[key]
        if variable is None:
            variable = tk.StringVar()
            variable.set(val)
        
        radio  = tk.Radiobutton(frame, variable=variable, **radio_kwargs)
        radio.grid(**grid_kwargs)
        radio.config(**config_kwargs)
        
        self.entries[key] = [radio, vtype, procfunc]
            
    def get_params(self):
        """This gets parameters for a given set of atoms"""
        #Params
        try:
            self.params = Params(self.a,self.b,self.c)
        except Exception:
            msgbox.showerror("Error", "Parameters for this atom combination not available!")
            raise
            
    def get_surface(self):
        """Get Vmat (potential) for a given set of parameters"""
        get_surface(self)

    def append_animation_position(self):

        theta_rad = np.deg2rad(self.theta) #Collision Angle

        #Positions of A, B and C relative to B
        X = np.array([
            [- self.Current_Separations[0], 0.],
            [0., 0.],
            [- np.cos(theta_rad) * self.Current_Separations[1], np.sin(theta_rad) * self.Current_Separations[1]]
        ])
        
        #Get centre of mass
        com = (X.T * self.params.Masses).T / self.params.total_mass
        
        #Translate to centre of mass (for animation)
        X -= com

        self.Animation_Positions.append(X)

    def get_arrays(self):

        self.xraci = ((self.xrabi ** 2) + (self.xrbci ** 2) - 2 * self.xrabi * self.xrbci * np.cos(self.theta)) ** 0.5
        self.Current_Separations[0] = self.xrabi
        self.Current_Separations[1] = self.xrbci
        self.Current_Separations[2] = self.xraci

        self.Current_Momenta[0] = self.prabi
        self.Current_Momenta[1] = self.prbci

        self.Current_Velocities[0] = self.prabi
        self.Current_Velocities[1] = self.prbci

        self.current_kinetic_energy = 0

    def add_trajectory_step(self):
        self.Trajectory_times.append(self.current_time)
        self.Trajectory_Separations.append(copy.deepcopy(self.Current_Separations))
        self.Trajectory_Velocities.append(copy.deepcopy(self.Current_Velocities))
        self.Trajectory_Momenta.append(copy.deepcopy(self.Current_Momenta))
        self.Trajectory_Accelerations.append(copy.deepcopy(self.Current_Accelerations))
        self.Trajectory_First_derivatives.append(copy.deepcopy(self.Current_First_derivative))
        self.Trajectory_Hessians.append(copy.deepcopy(self.Current_Hessian))

        self.Kinetic_Energies.append(self.current_kinetic_energy)
        self.Potential_Energies.append(self.current_potential_energy)

        self.current_total_energy = self.current_kinetic_energy + self.current_potential_energy
        self.Total_Energies.append(self.current_total_energy)
                        
    def get_trajectory(self):
        """Get dynamics, MEP or optimisation"""
        
        get_trajectory(self)
        
    def get_last_geo(self, *args):
        """Copy last geometry and momenta"""
        self.entries["xrabi"][0].delete(0, tk.END)
        self.entries["xrabi"][0].insert(0, self.Trajectory_Separations[-1][0])
        
        self.entries["xrbci"][0].delete(0, tk.END)
        self.entries["xrbci"][0].insert(0, self.Trajectory_Separations[-1][1])
        
        self.entries["prabi"][0].delete(0, tk.END)
        self.entries["prabi"][0].insert(0, self.Trajectory_Momenta[-1][0])
        
        self.entries["prbci"][0].delete(0, tk.END)
        self.entries["prbci"][0].insert(0, self.Trajectory_Momenta[-1][1])
            
    def export(self, *args):
        """Run calculation and print output in CSV format"""
        self._read_entries()
        self.get_arrays()
        self.get_trajectory()
        
        filename = asksaveasfilename(defaultextension=".csv")
        if not filename:
            return
            
        sources = [
            ["Time",            self.current_time                ],
            ["AB Distance",     self.Current_Separations[0]      ],
            ["BC Distance",     self.Current_Separations[1]      ],
            ["AC Distance",     self.Current_Separations[2]      ],
            ["AB Velocity",     self.Current_Velocities[0]       ],
            ["BC Velocity",     self.Current_Velocities[1]       ],
            ["AC Velocity",     self.Current_Velocities[2]       ],
            ["AB Momentum",     self.Current_Momenta[0]          ],
            ["BC Momentum",     self.Current_Momenta[1]          ],
            ["AC Momentum",     self.Current_Momenta[2]          ],
            ["AB dE/dx",        self.Current_First_derivative[0] ],
            ["BC dE/dx",        self.Current_First_derivative[1] ],
            ["AC dE/dx",        self.Current_First_derivative[2] ],
            ["Total Potential", self.current_potential_energy    ],
            ["Total Kinetic",   self.current_kinetic_energy      ],
            ["Total Energy",    self.current_total_energy        ],
            ["AB AB Hess Comp", self.Current_Hessian[0][0]       ],
            ["AB BC Hess Comp", self.Current_Hessian[0][1]       ],
            ["AB AC Hess Comp", self.Current_Hessian[0][2]       ],
            ["BC BC Hess Comp", self.Current_Hessian[1][1]       ],
            ["BC AC Hess Comp", self.Current_Hessian[1][2]       ],
            ["AC AC Hess Comp", self.Current_Hessian[2][2]       ]
        ]
        
        out = ",".join([t for t, s in sources]) + "\n"
        
        for step in range(len(self.Trajectory_times)):
            data = []
            for t, s in sources:
                try:
                    point = str(s[step])
                except:
                    point = ""
                data.append(point)
            out += ",".join(data) + "\n"
        
        with open(filename, "w") as f:
            f.write(out)
            
    def update_plot(self, *args):
        """Generate plot based on what type has been selected"""
        self._read_entries()
        self.get_surface()
        self.get_trajectory()
        
        if self.plot_type == "Contour Plot":
            self.plot_contour()
            self.plot_init_pos()
        elif self.plot_type == "Surface Plot":
            self.plot_surface()
            self.plot_init_pos()
        elif self.plot_type == "Skew Plot":
            self.plot_skew()
            self.plot_init_pos()
        elif self.plot_type == "Internuclear Distances vs Time":
            self.plot_ind_vs_t()
        elif self.plot_type == "Internuclear Current_Momenta vs Time":
            self.plot_inm_vs_t()
        elif self.plot_type == "Energy vs Time":
            self.plot_e_vs_t()
        elif self.plot_type == "p(AB) vs p(BC)":
            self.plot_momenta()
        elif self.plot_type == "v(AB) vs v(BC)":
            self.plot_velocities()
        elif self.plot_type == "Animation":
            self.animation()
            
    def plot_contour(self):    
        """Contour Plot"""
        plt.clf()
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.xlabel("AB Distance")
        plt.ylabel("BC Distance")
        
        X, Y = np.meshgrid(self.Grid_X, self.Grid_Y)
        
        levels = np.arange(np.min(self.Vmat) -1, float(self.cutoff), self.spacing)
        plt.contour(X, Y, self.Vmat, levels = levels)
        plt.xlim([min(self.Grid_X),max(self.Grid_X)])
        plt.ylim([min(self.Grid_Y),max(self.Grid_Y)])
        
        lc = colorline([x[0] for x in self.Trajectory_Separations], [x[1] for x in self.Trajectory_Separations], cmap = plt.get_cmap("jet"), linewidth=1)
        
        ax.add_collection(lc)
        plt.draw()
        plt.pause(0.0001) #This stops MPL from blocking
            
    def plot_skew(self):    
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
        
        plt.xlabel("Q1")
        plt.ylabel("Q2")
        
        X, Y = np.meshgrid(self.Grid_X, self.Grid_Y)
        
        ma = self.params.mass_a
        mb = self.params.mass_b
        mc = self.params.mass_c
        
        a    = ((ma * (mb + mc)) / (self.params.total_mass)) ** 0.5
        b    = ((mc * (ma + mb)) / (self.params.total_mass)) ** 0.5
        beta = np.arccos(((ma * mc) / ((mb + mc) * (ma + mb))) ** 0.5)

        #Transform grid
        Q1 = a * X + b * Y * np.cos(beta)
        Q2 = b * Y * np.sin(beta)
        
        #Plot gridlines every 0.5A
        grid_x = [self.Grid_X[0]] + list(np.arange(np.ceil(min(self.Grid_X) * 2) / 2, np.floor(max(self.Grid_X) * 2) / 2 + 0.5, 0.5)) + [self.Grid_X[-1]]
        grid_y = [self.Grid_Y[0]] + list(np.arange(np.ceil(min(self.Grid_Y) * 2) / 2, np.floor(max(self.Grid_Y) * 2) / 2 + 0.5, 0.5)) + [self.Grid_Y[-1]]
        
        for x in grid_x:
            r1 = [x, grid_y[ 0]]
            r2 = [x, grid_y[-1]]

            q1 = [a * r1[0] + b * r1[1] * np.cos(beta), b * r1[1] * np.sin(beta)]
            q2 = [a * r2[0] + b * r2[1] * np.cos(beta), b * r2[1] * np.sin(beta)]
                  
            plt.plot([q1[0], q2[0]], [q1[1], q2[1]], lw=1, color='gray')
            plt.text(q2[0], q2[1], str(x))
            
        for y in grid_y:
            r1 = [grid_x[ 0], y]
            r2 = [grid_x[-1], y]

            q1 = [a * r1[0] + b * r1[1] * np.cos(beta), b * r1[1] * np.sin(beta)]
            q2 = [a * r2[0] + b * r2[1] * np.cos(beta), b * r2[1] * np.sin(beta)]
                  
            plt.plot([q1[0], q2[0]], [q1[1], q2[1]], lw=1, color='gray')
            plt.text(q2[0], q2[1], str(y))
            
        #Plot transformed PES
        levels = np.arange(np.min(self.Vmat) -1, float(self.cutoff), self.spacing)
        plt.contour(Q1, Q2, self.Vmat, levels = levels)
        plt.autoscale()
        plt.axes().set_aspect('equal')
        
        #Plot transformed trajectory
        
        srab = a * np.array([x[0] for x in self.Trajectory_Separations]) + b * np.array([x[1] for x in self.Trajectory_Separations]) * np.cos(beta)
        srbc = b * np.array([x[1] for x in self.Trajectory_Separations]) * np.sin(beta)
        
        lc = colorline(srab, srbc, cmap = plt.get_cmap("jet"), linewidth=2)
        
        ax.add_collection(lc)
        
            
        plt.draw()
        plt.pause(0.0001)
        
    def plot_surface(self):
        """3d Surface Plot"""
        
        plt.close('all') #New figure needed for 3D axes
        self.fig_3d = plt.figure('Surface Plot', figsize=(5,5))
        
        ax = Axes3D(self.fig_3d)
        
        plt.xlabel("AB Distance")
        plt.ylabel("BC Distance")
        
        X, Y = np.meshgrid(self.Grid_X, self.Grid_Y)
        ax.set_xlim3d([min(self.Grid_X),max(self.Grid_X)])
        ax.set_ylim3d([min(self.Grid_Y),max(self.Grid_Y)])
        
        Z = np.clip(self.Vmat, -10000, self.cutoff)
        
        ax.plot_surface(X, Y, Z, rstride=self.spacing, cstride=self.spacing, cmap='jet', alpha=0.3, linewidth=0)
        ax.contour(X, Y, Z, zdir='z', cmap='jet', stride=self.spacing, offset=np.min(Z) - 10)
        ax.plot([x[0] for x in self.Trajectory_Separations], [x[1] for x in self.Trajectory_Separations], self.Potential_Energies)
         
        plt.draw()
        plt.pause(0.0001)
        
    def plot_ind_vs_t(self):
        """Internuclear Distances VS Time"""
        plt.clf()
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.xlabel("Time")
        plt.ylabel("Distance")
        
        ab, = plt.plot(self.Trajectory_times, [x[0] for x in self.Trajectory_Separations], label = "A-B")
        bc, = plt.plot(self.Trajectory_times, [x[1] for x in self.Trajectory_Separations], label = "B-C")
        ac, = plt.plot(self.Trajectory_times, [x[2] for x in self.Trajectory_Separations], label = "A-C")
        
        plt.legend(handles=[ab, bc, ac])
        
        plt.draw()
        plt.pause(0.0001)
        
    def plot_inm_vs_t(self):
        """Internuclear Current_Momenta VS Time"""
        plt.clf()
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.xlabel("Time")
        plt.ylabel("Momentum")

        ab, = plt.plot(self.Trajectory_times, [x[0] for x in self.Trajectory_Momenta], label = "A-B")
        bc, = plt.plot(self.Trajectory_times, [x[1] for x in self.Trajectory_Momenta], label = "B-C")
        ac, = plt.plot(self.Trajectory_times, [x[2] for x in self.Trajectory_Momenta], label = "A-C")
        
        plt.legend(handles=[ab, bc, ac])
        
        plt.draw()
        plt.pause(0.0001)      
        
    def plot_momenta(self):
        """AB Momentum VS BC Momentum"""
        plt.clf()
        ax = plt.gca()
        
        plt.xlabel("AB Momentum")
        plt.ylabel("BC Momentum")

        lc = colorline([x[0] for x in self.Trajectory_Momenta], [x[1] for x in self.Trajectory_Momenta], cmap = plt.get_cmap("jet"), linewidth=1)
        
        ax.add_collection(lc)
        ax.autoscale()
        plt.draw()
        plt.pause(0.0001)
        
    def plot_velocities(self):
        """AB Velocity VS BC Velocity"""
        plt.clf()
        ax = plt.gca()
        
        plt.xlabel("AB Velocity")
        plt.ylabel("BC Velocity")
        
        lc = colorline([x[0] for x in self.Trajectory_Velocities], [x[1] for x in self.Trajectory_Velocities], cmap = plt.get_cmap("jet"), linewidth=1)
        
        ax.add_collection(lc)
        ax.autoscale()
        plt.draw()
        plt.pause(0.0001)
        
    def plot_e_vs_t(self):
        """Energy VS Time"""
        plt.clf()
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.xlabel("Time")
        plt.ylabel("Energy")

        pot, = plt.plot(self.Trajectory_times, self.Potential_Energies, label = "Potential Energy")
        kin, = plt.plot(self.Trajectory_times, self.Kinetic_Energies,  label = "Kinetic Energy")
        
        plt.legend(handles=[pot, kin])
        
        plt.draw()
        plt.pause(0.0001)
        
    def animation(self):
        """Animation"""
        plt.close('all')
        self.ani_fig = plt.figure('Animation', figsize=(5,5))
        
        def init():
            ap, bp, cp = patches
            ax.add_patch(ap)
            ax.add_patch(bp)
            ax.add_patch(cp)
            return ap, bp, cp,
            
        def update(i):
            ap, bp, cp = patches
            ap.center = self.Animation_Positions[i][0]
            bp.center = self.Animation_Positions[i][1]
            cp.center = self.Animation_Positions[i][2]
            return ap, bp, cp,
            
        ax = plt.axes(
            xlim = (
                min([x[0] for x in self.Animation_Positions], key=lambda x: x[0])[0] - 1, 
                max([x[2] for x in self.Animation_Positions], key=lambda x: x[0])[0] + 1
            ),
            ylim = (
                min([x[0] for x in self.Animation_Positions], key=lambda x: x[1])[1] - 1, 
                max([x[2] for x in self.Animation_Positions], key=lambda x: x[1])[1] + 1
            )
        )
        ax.set_aspect('equal')
            
        patches = [None, None, None]
        
        for i, at_name in enumerate(["a", "b", "c"]):
            at = self.entries[at_name][0].get()
            index, vdw, c = self.atom_map[at]
            pos = self.Animation_Positions[0][i]
            patch = plt.Circle(pos, vdw * 0.25, fc = c)
            patches[i] = patch
        
        self.anim = FuncAnimation(self.ani_fig, update, init_func=init, frames=len(self.Trajectory_Separations), repeat=True, interval=20)
        
        try:
            plt.show()
        except:
            pass
        
    def plot_init_pos(self, *args):
        """Cross representing initial geometry"""
        if not self.plot_type == "Contour Plot":
            return
            
        self.init_pos_plot, = plt.plot([self.xrabi], [self.xrbci], marker='x', markersize=6, color="red")
        plt.draw()
        plt.pause(0.0001)
        
    def plot_eigen(self, *args):
        """Plot eigenvectors and eigenvalues on contour plot"""
        if not self.plot_type == "Contour Plot":
            return
            
        self.update_geometry_info()
        
        evecs = self._eigenvectors
        evals = self._eigenvalues
        
        self.eig1_plot = plt.arrow(
            self.xrabi, 
            self.xrbci, 
            evecs[0][0] / 10,
            evecs[0][1] / 10,
            color = "blue" if evals[0] > 0 else "red",
            label = "{:+7.3f}".format(evals[0])
        )
        self.eig2_plot = plt.arrow(
            self.xrabi, 
            self.xrbci, 
            evecs[1][0] / 10,
            evecs[1][1] / 10,
            color = "blue" if evals[1] > 0 else "red",
            label = "{:+7.3f}".format(evals[1])
        )
        
        plt.draw()
        plt.pause(0.0001)
        
    def get_first(self):
        """1 step of trajectory to get geometry properties"""
        self.get_arrays()
        get_first(self)
        
    def update_geometry_info(self, *args):
        """Updates the info pane"""
        self.get_first()
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.Current_Hessian)
            
            self._eigenvalues  = eigenvalues
            self._eigenvectors = eigenvectors
            
            ke     = "{:+7.3f}".format(self.current_kinetic_energy)
            pe     = "{:+7.3f}".format(self.current_potential_energy)
            etot   = "{:+7.3f}".format(self.current_total_energy)
            fab    = "{:+7.3f}".format(self.Current_First_derivative[0])
            fbc    = "{:+7.3f}".format(self.Current_First_derivative[1])
            
            eval1  = "{:+7.3f}".format(eigenvalues[0])
            eval2  = "{:+7.3f}".format(eigenvalues[1])
            
            evec11 = "{:+7.3f}".format(eigenvectors[0][0])
            evec12 = "{:+7.3f}".format(eigenvectors[0][1])
            evec21 = "{:+7.3f}".format(eigenvectors[1][0])
            evec22 = "{:+7.3f}".format(eigenvectors[1][1])
            
        except:
            ke     = "       "
            pe     = "       "
            etot   = "       "
            fab    = "       "
            fbc    = "       "
            eval1  = "       "
            eval2  = "       "
            evec11 = "       "
            evec12 = "       "
            evec21 = "       "
            evec22 = "       "
            
        self.i_ke["text"] = ke
        self.i_pe["text"] = pe
        self.i_etot["text"] = etot
        
        self.i_fab["text"] = fab
        self.i_fbc["text"] = fbc
        
        self.i_eval1["text"] = eval1
        self.i_eval2["text"] = eval2
        
        self.i_evec11["text"] = evec11
        self.i_evec12["text"] = evec12
        
        self.i_evec21["text"] = evec21
        self.i_evec22["text"] = evec22      
        
def colorline(
    Grid_X, Grid_Y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates Grid_X and Grid_Y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(Grid_X))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(Grid_X, Grid_Y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)


    return lc


def make_segments(Grid_X, Grid_Y):
    """
    Create list of line segments from Grid_X and Grid_Y coordinates, in the correct format
    for LineCollection: an array of the form numlines times (points per line) times 2 (Grid_X
    and Grid_Y) array
    """

    points = np.array([Grid_X, Grid_Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="Starts the Triatomic LEPS GUI")
    parser.add_argument("-a", "--advanced", action="store_true", help="Include additional features in the GUI")
    
    args = parser.parse_args()
    interactive = Interactive(advanced = args.advanced)
    
