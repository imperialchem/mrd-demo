#Created on Mon May 22 16:59:17 2017
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


from params import params
from lepspoint import leps_energy,leps_gradient,leps_hessian
from lepnorm import lepnorm

import numpy as np
from numpy.linalg.linalg import LinAlgError

from matplotlib import use as mpl_use
mpl_use("TkAgg") # might need to be changed on different operating systems

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings

import tkinter as tk
import tkinter.messagebox as msgbox
from tkinter.filedialog import asksaveasfilename

from configparser import ConfigParser
from argparse import ArgumentParser

class Interactive():
    
    def __init__(self, advanced=False): #Initialise Class
        
        ###Initialise tkinter###
        self.root = tk.Tk()
        self.root.title("LEPS GUI")
        self.root.resizable(0,0)
        
        ###Initialise defaults###
        
        config = ConfigParser(inline_comment_prefixes=(';', '#'))
        # The line below allows for dictionary keys with capital letters
        config.optionxform = lambda op:op
        config.read('params.ini')
        
        #Atom: [Index, VdW radius, colour]
        #VdW radius - used for animation
        #Colour     - used for animation
        atom_map = {}
        atoms = config['atoms']
        self.atoms_list = []
        for element, l in atoms.items():
            mass, vdw, colour = l.split(',')
            atom_map[element] = [
                float(vdw),
                '#' + colour.strip()            
            ]
            self.atoms_list.append(element)
        self.atom_map = atom_map
        
        defaults = config['defaults']
        self.dt  = float(defaults['dt'])  #Time step in dynamics trajectory
        self.H   = float(defaults['Hparam'])   #Surface parameter
        self.lim = float(defaults['lim']) #Calculation will stop once this distance is exceeded

        self.Vmat = None       #Array where potential is stored for each gridpoint
        self.old_params = None #Variable used to prevent surface being recalculated
        
        self.entries  = {} #Dictionary of entries to be read on refresh (user input)
        self.defaults = {  #Defaults for each entry
           #Key        : Default value  , type , processing function
            "a"        : ["H"           , str  , None                         ],
            "b"        : ["H"           , str  , None                         ],
            "c"        : ["H"           , str  , None                         ],
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
                "Internuclear Momenta vs Time", "Energy vs Time", "p(AB) vs p(BC)", "v(AB) vs v(BC)", "Animation"]     
        else:
            plot_types = ["Contour Plot", "Skew Plot", "Surface Plot", "Internuclear Distances vs Time", 
                "Internuclear Momenta vs Time", "Energy vs Time", "Animation"]     
        
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
            self.masses,self.morse_params,self.plot_limits = params(self.a,self.b,self.c)
        except Exception:
            msgbox.showerror("Error", "Parameters for this atom combination not available!")
            raise
        
        self.reduced_masses = np.array([(self.masses[0]*self.masses[1])/(self.masses[0]+self.masses[1]),
                                        (self.masses[1]*self.masses[2])/(self.masses[1]+self.masses[2]),
                                        (self.masses[0]*self.masses[2])/(self.masses[0]+self.masses[2])])
 
    def get_surface(self):
        """Get Vmat (potential) for a given set of parameters"""
        self.get_params()
        
        #Check if params have changed. If not, no need to recalculate
        new_params = [self.a, self.b, self.c, self.theta]
        if self.old_params == new_params and not self._firstrun:
            return

        resl = 0.02 #Resolution
        self._firstrun = False
        
        #Get grid
        self.x = np.arange(self.plot_limits[0,0],self.plot_limits[0,1],resl)
        self.y = np.arange(self.plot_limits[1,0],self.plot_limits[1,1],resl)

        self.Vmat = np.zeros((len(self.y), len(self.x)))
        
        #Calculate potential for each gridpoint
        for drabcount, drab in enumerate(self.x):
            for drbccount, drbc in enumerate(self.y):
    
                V = leps_energy(np.array([drab,drbc,np.deg2rad(self.theta)]),
                   self.morse_params,self.H)
                self.Vmat[drbccount, drabcount] = V

        self.old_params = new_params
                        
    def get_trajectory(self):
        """Get dynamics, MEP or optimisation"""
        
        itlimit = self.steps #Max number of steps
        dt      = self.dt    #Time step
        ti      = 0          #Initial time variable
        tf      = 0          #Final time variable
        
        thetai = np.deg2rad(self.theta) #Collision Angle
        
        xrabi = self.xrabi   #Initial AB separation
        xrbci = self.xrbci   #Initial BC separation
        xraci = ((xrabi ** 2) + (xrbci ** 2) - 2 * xrabi * xrbci * np.cos(thetai)) ** 0.5 #Initial AC separation
        prabi = self.prabi   #Initial AB momentum
        prbci = self.prbci   #Initial BC momentum
        
        vrabi = prabi / self.reduced_masses[0]  #Initial AB Velocity
        vrbci = prbci / self.reduced_masses[1]  #Initial BC Velocity
        
        #Positions of A, B and C relative to B
        positions = np.array([[- xrabi, 0.],[0., 0.],[- np.cos(thetai) * xrbci, np.sin(thetai) * xrbci]])
        
        #Get centre of mass
        com = self.masses.dot(positions)/np.sum(self.masses)
        
        #Translate to centre of mass (for animation)
        positions = positions - com
        
        self.ra    = [positions[0]]
        self.rb    = [positions[1]]
        self.rc    = [positions[2]]
        
        #Initial AC Velocity
        va = vrabi*(positions[0]-positions[1])/np.linalg.norm(positions[0]-positions[1])
        vc = vrbci*(positions[2]-positions[1])/np.linalg.norm(positions[2]-positions[1])
        vraci = np.linalg.norm(va + vc)
        
        #Initialise outputs
        self.xrab  = [xrabi]
        self.xrbc  = [xrbci]
        self.xrac  = [xraci]
        self.vrab  = [vrabi]
        self.vrbc  = [vrbci]
        self.vrac  = [vraci]
        self.t     = [ti]
    
        self.Vrint = Vrint = []
        self.Ktot  = Ktot  = []

        self.arab  = arab  = []
        self.arbc  = arbc  = []
        self.arac  = arac  = []
        self.etot  = etot  = []
    
        #Flag to stop appending to output in case of a crash
        terminate = False        

        for itcounter in range(itlimit):
            if self.calc_type != "Dynamics":
                vrabi = 0
                vrbci = 0
                vraci = 0
            
            #Get current potential, forces, and Hessian
            Vrinti = leps_energy(np.array([xrabi,xrbci,thetai]),self.morse_params,self.H)
            forces = -leps_gradient(np.array([xrabi,xrbci,thetai]),self.morse_params,self.H)
            hessian = leps_hessian(np.array([xrabi,xrbci,thetai]),self.morse_params,self.H)
            Vrint.append(Vrinti)

            if self.calc_type in ["Opt Min", "Opt TS"]: #Optimisation calculations
                
                #Diagonalise Hessian
                eigenvalues, eigenvectors = np.linalg.eig(hessian)
                
                #Eigenvalue test
                neg_eig_i = [i for i,eig in enumerate(eigenvalues) if eig < -0.01]
                if len(neg_eig_i) == 0 and self.calc_type == "Opt TS":
                    msgbox.showinfo("Eigenvalues Info", "No negative eigenvalues at this geometry")
                    terminate = True
                elif len(neg_eig_i) == 1 and self.calc_type == "Opt Min":
                    msgbox.showerror("Eigenvalues Error", "Too many negative eigenvalues at this geometry")
                    terminate = True                    
                elif len(neg_eig_i) > 1:
                    msgbox.showerror("Eigenvalues Error", "Too many negative eigenvalues at this geometry")
                    terminate = True
                
                #Optimiser
                disps = np.array([0.,0.,0.])
                for mode in range(len(eigenvalues)):
                    e_val = eigenvalues[mode]
                    e_vec = eigenvectors[mode]

                    disp = np.dot(np.dot((e_vec.T), forces), e_vec) / e_val
                    disps += disp
                    
                xrabf = xrabi + disps[0]
                xrbcf = xrbci + disps[1]
                xracf = ((xrabf ** 2) + (xrbcf ** 2) - 2 * xrabf * xrbcf * np.cos(thetai)) ** 0.5

                arabi  = 0
                arbci  = 0
                araci  = 0
                thetaf = thetai
                vrabf  = 0
                vrbcf  = 0
                vracf  = 0
                Ktoti  = 0
                tf    += dt
                
            else: #Dynamics/MEP
                try:
                    xrabf,xrbcf,xracf,thetaf,vrabf,vrbcf,vracf,tf,arabi,arbci,araci,Ktoti = lepnorm(xrabi,xrbci,thetai,forces[0],forces[1],forces[2],vrabi,vrbci,vraci,hessian[0,0],hessian[0,1],hessian[0,2],hessian[1,1],hessian[1,2],hessian[2,2],self.masses[0],self.masses[1],self.masses[2],ti,dt,self.calc_type == "MEP")
                except LinAlgError:
                    msgbox.showerror("Surface Error", "Energy could not be evaulated at step {}. Steps truncated".format(itcounter + 1))
                    terminate = True
                
            if xrabf > self.lim or xrbcf > self.lim: #Stop calc if distance lim is exceeded
                msgbox.showerror("Surface Error", "Surface Limits exceeded at step {}. Steps truncated".format(itcounter + 1))
                terminate = True
                    
            arab.append(arabi)
            arbc.append(arbci)
            arac.append(araci)
            Ktot.append(Ktoti)
            
            #Total energy
            etot.append(Vrint[itcounter] + Ktot[itcounter])
            
            if itcounter != itlimit - 1 and not terminate:
                
                #As above
	        #Positions of A, B and C relative to B
                positions = np.array([[- xrabf, 0.],[0., 0.],[- np.cos(thetaf) * xrbci, np.sin(thetaf) * xrbcf]])
        
	        #Get centre of mass
                com = self.masses.dot(positions)/np.sum(self.masses)
        
	        #Translate to centre of mass (for animation)
                positions = positions - com
        
                a    = [positions[0]]
                b    = [positions[1]]
                c    = [positions[2]]
                
                #Get A-C Velocity
                r0 = np.linalg.norm(self.rc[-1] - self.ra[-1])
                r1 = np.linalg.norm(positions[2] - positions[0])
                vrac = (r1 - r0) / dt
                
                self.ra.append(positions[0])            #A  Pos
                self.rb.append(positions[1])            #B  Pos
                self.rc.append(positions[2])            #C  Pos
                self.xrab.append(xrabf)                 #A-B Distance
                self.xrbc.append(xrbcf)                 #B-C Distance
                self.xrac.append(xracf)                 #A-C Distance
                if self.calc_type == "Dynamics":
                    self.vrab.append(vrabf)                 #A-B Velocity
                    self.vrbc.append(vrbcf)                 #B-C Velocity
                    self.vrac.append(vrac)                  #A-C Velocity
                else:
                    self.vrab.append(0.)                    #A-B Velocity
                    self.vrbc.append(0.)                    #B-C Velocity
                    self.vrac.append(0.)                    #A-C Velocity
                self.t.append(tf)                       #Time
                

            xrabi = xrabf
            xrbci = xrbcf
            xraci = xracf
            vrabi = vrabf
            vrbci = vrbcf
            vraci = vracf
            ti = tf
            
            if terminate:
                break
        
    def get_last_geo(self, *args):
        """Copy last geometry and momenta"""
        self.entries["xrabi"][0].delete(0, tk.END)
        self.entries["xrabi"][0].insert(0, self.xrab[-1])
        
        self.entries["xrbci"][0].delete(0, tk.END)
        self.entries["xrbci"][0].insert(0, self.xrbc[-1])
        
        self.entries["prabi"][0].delete(0, tk.END)
        self.entries["prabi"][0].insert(0, self.vrab[-1] * self.reduced_masses[0])
        
        self.entries["prbci"][0].delete(0, tk.END)
        self.entries["prbci"][0].insert(0, self.vrbc[-1] * self.reduced_masses[0])
            
    def export(self, *args):
        """Run calculation and print output in CSV format"""
        self._read_entries()
        self.get_trajectory()
        
        filename = asksaveasfilename(defaultextension=".csv")
        if not filename:
            return
            
        sources = [
            ["Time",            self.t                           ],
            ["AB Distance",     self.xrab                        ],
            ["BC Distance",     self.xrbc                        ],
            ["AC Distance",     self.xrac                        ],
            ["AB Velocity",     self.vrab                        ],
            ["BC Velocity",     self.vrbc                        ],
            ["AC Velocity",     self.vrac                        ],
            ["AB Momentum",     self.vrab*self.reduced_masses[0] ],
            ["BC Momentum",     self.vrab*self.reduced_masses[1] ],
            ["AC Momentum",     self.vrab*self.reduced_masses[2] ],
            ["AB Force",        -self.gradient[0]                ],
            ["BC Force",        -self.gradient[1]                ],
            ["theta Force",     -self.gradient[2]                ],
            ["Total Potential", self.Vrint                       ],
            ["Total Kinetic",   self.Ktot                        ],
            ["Total Energy",    self.etot                        ],
            ["AB AB Hess Comp", self.hessian[0,0]                ],
            ["AB BC Hess Comp", self.hessian[0,1]                ],
            ["AB theta Hess Comp", self.hessian[0,2]             ],
            ["BC BC Hess Comp", self.hessian[1,1]                ],
            ["BC theta Hess Comp", self.hessian[1,2]             ],
            ["theta theta Hess Comp", self.hessian[2,2]          ]
        ]
        
        out = ",".join([t for t, s in sources]) + "\n"
        
        for step in range(len(self.t)):
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
        elif self.plot_type == "Internuclear Momenta vs Time":
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
        
        X, Y = np.meshgrid(self.x, self.y)
        
        levels = np.arange(np.min(self.Vmat) -1, float(self.cutoff), self.spacing)
        plt.contour(X, Y, self.Vmat, levels = levels)
        plt.xlim([min(self.x),max(self.x)])
        plt.ylim([min(self.y),max(self.y)])
        
        lc = colorline(self.xrab, self.xrbc, cmap = plt.get_cmap("jet"), linewidth=1)
        
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
        
        X, Y = np.meshgrid(self.x, self.y)
        
        ma,mb,mc = self.masses
        
        a    = ((ma * (mb + mc)) / np.sum(self.masses)) ** 0.5
        b    = ((mc * (ma + mb)) / np.sum(self.masses)) ** 0.5
        beta = np.arccos(((ma * mc) / ((mb + mc) * (ma + mb))) ** 0.5)

        #Transform grid
        Q1 = a * X + b * Y * np.cos(beta)
        Q2 = b * Y * np.sin(beta)
        
        #Plot gridlines every 0.5A
        grid_x = [self.x[0]] + list(np.arange(np.ceil(min(self.x) * 2) / 2, np.floor(max(self.x) * 2) / 2 + 0.5, 0.5)) + [self.x[-1]]
        grid_y = [self.y[0]] + list(np.arange(np.ceil(min(self.y) * 2) / 2, np.floor(max(self.y) * 2) / 2 + 0.5, 0.5)) + [self.y[-1]]
        
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
        xrab = np.array(self.xrab)
        yrab = np.array(self.xrbc)
        
        srab = a * xrab + b * yrab * np.cos(beta)
        srbc = b * yrab * np.sin(beta)
        
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
        
        X, Y = np.meshgrid(self.x, self.y)
        ax.set_xlim3d([min(self.x),max(self.x)])
        ax.set_ylim3d([min(self.y),max(self.y)])
        
        Z = np.clip(self.Vmat, -10000, self.cutoff)
        
        ax.plot_surface(X, Y, Z, rstride=self.spacing, cstride=self.spacing, cmap='jet', alpha=0.3, linewidth=0.25, edgecolor='black')

        levels = np.arange(np.min(self.Vmat) -1, float(self.cutoff), self.spacing)
        ax.contour(X, Y, Z, zdir='z', levels=levels, offset=ax.get_zlim()[0]-1)
        ax.plot(self.xrab, self.xrbc, self.Vrint, color='black', linestyle='none', marker='o', markersize=2)
         
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
        
        ab, = plt.plot(self.t, self.xrab, label = "A-B")
        bc, = plt.plot(self.t, self.xrbc, label = "B-C")
        ac, = plt.plot(self.t, self.xrac, label = "A-C")
        
        plt.legend(handles=[ab, bc, ac])
        
        plt.draw()
        plt.pause(0.0001)
        
    def plot_inm_vs_t(self):
        """Internuclear Momenta VS Time"""
        plt.clf()
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.xlabel("Time")
        plt.ylabel("Momentum")
        
        momab = [v * self.mab for v in self.vrab]
        mombc = [v * self.mbc for v in self.vrbc]
        momac = [v * self.mac for v in self.vrac]

        ab, = plt.plot(self.t, momab, label = "A-B")
        bc, = plt.plot(self.t, mombc, label = "B-C")
        ac, = plt.plot(self.t, momac, label = "A-C")
        
        plt.legend(handles=[ab, bc, ac])
        
        plt.draw()
        plt.pause(0.0001)      
        
    def plot_momenta(self):
        """AB Momentum VS BC Momentum"""
        plt.clf()
        ax = plt.gca()
        
        plt.xlabel("AB Momentum")
        plt.ylabel("BC Momentum")
        
        momab = [v * self.mab for v in self.vrab]
        mombc = [v * self.mbc for v in self.vrbc]

        lc = colorline(momab, mombc, cmap = plt.get_cmap("jet"), linewidth=1)
        
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
        
        lc = colorline(self.vrab, self.vrbc, cmap = plt.get_cmap("jet"), linewidth=1)
        
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

        pot, = plt.plot(self.t, self.Vrint, label = "Potential Energy")
        kin, = plt.plot(self.t, self.Ktot,  label = "Kinetic Energy")
        
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
            ap.center = self.ra[i]
            bp.center = self.rb[i]
            cp.center = self.rc[i]
            return ap, bp, cp,
            
        ax = plt.axes(
        xlim = (min(self.ra, key=lambda x: x[0])[0] - 1, max(self.rc, key=lambda x: x[0])[0] + 1),
        ylim = (min(self.ra, key=lambda x: x[1])[1] - 1, max(self.rc, key=lambda x: x[1])[1] + 1)
        )
        ax.set_aspect('equal')
            
        patches = []
        
        for at_name in ["a", "b", "c"]:
            at = self.entries[at_name][0].get()
            vdw, col = self.atom_map[at]
            pos = getattr(self, "r" + at_name)[0]
            patch = plt.Circle(pos, vdw * 0.25, color = col)
            patches.append(patch)
        
        self.anim = FuncAnimation(self.ani_fig, update, init_func=init, frames=len(self.ra), repeat=True, interval=20)
        
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
        self._read_entries()
        self.get_params()
        dt   = self.dt
        
        xrabi = self.xrabi
        xrbci = self.xrbci
        prabi = self.prabi
        prbci = self.prbci
        
        thetai = np.deg2rad(self.theta)
        
        vrabi = prabi / self.reduce_masses[0]
        vrbci = prbci / self.reduced_masses[1]
        vraci = 0
        
        ti = 0
        
        Vrinti = leps_energy(np.array([xrabi,xrbci,thetai]),self.morse_params,self.H)
        gradient = leps_gradient(np.array([xrabi,xrbci,thetai]),self.morse_mparams,self.H)
        hessian = leps_hessian(np.array([xrabi,xrbci,thetai]),self.morse_params,self.H)
        # Call lepnorm just to get the kinetic energy
        Ktoti = lepnorm(xrabi,xrbci,thetai,i-gradient[0],-gradient[1],-gradient[2],vrabi,vrbci,vraci,hessian[0,0],hessian[0,1],hessian[0,2],hessian[1,1],hessian[1,2],hessian[2,2],lef,masses[0],self.masses[1],self_masses[2],ti,dt,False)[-1]
        
        return Vrinti,gradient,hessian,Ktoti
        
    def update_geometry_info(self, *args):
        """Updates the info pane"""
        try:
            Vrinti,gradient,hessian,Ktoti = self.get_first()
            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            
            self._eigenvalues  = eigenvalues
            self._eigenvectors = eigenvectors
            
            ke     = "{:+7.3f}".format(Ktoti)
            pe     = "{:+7.3f}".format(Vrinti)
            etot   = "{:+7.3f}".format(Vrinti + Ktoti)
            fab    = "{:+7.3f}".format(-gradient[0])
            fbc    = "{:+7.3f}".format(-gradient[1])
            
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
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
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
        
if __name__ == "__main__":
    
    parser = ArgumentParser(description="Starts the Triatomic LEPS GUI")
    parser.add_argument("-a", "--advanced", action="store_true", help="Include additional features in the GUI")
    
    args = parser.parse_args()
    interactive = Interactive(advanced = args.advanced)
    
