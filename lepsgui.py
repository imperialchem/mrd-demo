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
from lepspoint import leps_energy,leps_gradient,leps_hessian,cos_rule
from lepsmove import calc_trajectory,kinetic_energy,velocities,velocity_AC
from lepsplots import plot_contour,plot_skew,plot_surface,plot_ind_vs_t,plot_inv_vs_t,plot_momenta_vs_t,plot_momenta,plot_velocities,plot_e_vs_t,animation

import numpy as np

from matplotlib import use as mpl_use
mpl_use("TkAgg") # might need to be changed on different operating systems

import matplotlib.pyplot as plt
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
        self.H   = float(defaults['Hparam'])   #Surface parameter

        self.Vmat = None       #Array where potential is stored for each gridpoint
        self.surf_params = None #Variable used to prevent surface being recalculated
        self.traj_params = None #Variable used to prevent trajectory being recalculated

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
            "dt"       : ["0.002"       , float, lambda x: max(1e-7,x)        ],
            "cutoff"   : ["-20"         , float, None                         ],
            "spacing"  : ["5"           , float, None                         ],
            "calc_type": ["Dynamics"    , str  , None                         ],
            "theta"    : ["180"         , float, lambda x:np.deg2rad(x)       ],
            "plot_type": ["Contour Plot", str  , None                         ]
        }
        
        #Store variable as class attributes
        for key, l in self.defaults.items():
            val, vtype, procfunc = l
            val = vtype(val)
            if procfunc: #Check whether processing is needed
                val = procfunc(val)
            setattr(self, key, val)
        
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
            plot_types = ["Contour Plot", "Skew Plot", "Surface Plot", "Internuclear Distances vs Time", "Internuclear Velocities vs Time",
                "Momenta vs Time", "Energy vs Time", "p(AB) vs p(BC)", "v(AB) vs v(BC)", "Animation"]     
        else:
            plot_types = ["Contour Plot", "Skew Plot", "Surface Plot", "Internuclear Distances vs Time", "Internuclear Velocities vs Time", 
                "Momenta vs Time", "Energy vs Time", "Animation"]     
        
        self._add_optionmenu(type_frame, "plot_type", plot_types , {}, gk('00'), {"width":20})
        
        #Steps Frame
        steps_frame = self._add_frame(dict(master=self.root, text="Steps", **sunken), gk('210055news'))
        self._add_label(steps_frame, {"text": "  number"}, gk('00'))
        self._add_entry(steps_frame, "steps", {}, {"row":0, "column":1}, {"width":6})
        self._add_label(steps_frame, {"text": "  size"}, gk('02'))
        self._add_entry(steps_frame, "dt", {}, {"row":0, "column":3}, {"width":6})
        
        #Cutoff Frame
        cutoff_frame = self._add_frame(dict(master=self.root, text="Cutoff (kcal/mol)", **sunken), gk('310055news'))
        self._add_scale(cutoff_frame, "cutoff",{"from_":-100, "to":0, "orient":"horizontal"}, gk('00ew'), {"length":200})
        
        #Contour Spacing Frame
        spacing_frame = self._add_frame(dict(master=self.root, text="Contour Spacing", **sunken), gk('410055news'))
        self._add_scale( spacing_frame, "spacing", {"from_":0.5, "to":10, "resolution":0.5, "orient":"horizontal"}, gk('00ew'), {"length":200})
        
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
        self._add_label(forces_frame, {"text": "along AB: "}, gk('00'))
        self._add_label(forces_frame, {"text": "along BC: "}, gk('10'))
        
        self.i_fab  = self._add_label(forces_frame, {"text": ""}, gk('01'))
        self.i_fbc  = self._add_label(forces_frame, {"text": ""}, gk('11'))
        
        hessian_frame = self._add_frame(dict(master=geometry_frame, text="Hessian", **sunken), gk('300055news'))
        self._add_label(hessian_frame, {"text": "1:         "}, gk('01'))
        self._add_label(hessian_frame, {"text": "2:         "}, gk('02'))
        self._add_label(hessian_frame, {"text": "     ω²"}    , gk('10'))
        self._add_label(hessian_frame, {"text": "AB direction:"}, gk('20'))
        self._add_label(hessian_frame, {"text": "BC direction:"}, gk('30'))
        
        self.i_eval1 = self._add_label(hessian_frame, {"text": ""}, gk('11'))
        self.i_eval2 = self._add_label(hessian_frame, {"text": ""}, gk('12'))
        
        self.i_evec11 = self._add_label(hessian_frame, {"text": ""}, gk('21'))
        self.i_evec12 = self._add_label(hessian_frame, {"text": ""}, gk('22'))
        self.i_evec21 = self._add_label(hessian_frame, {"text": ""}, gk('31'))
        self.i_evec22 = self._add_label(hessian_frame, {"text": ""}, gk('32'))
        
        self._add_button(geometry_frame, {"text": "Plot"}, gk('400055'), {"<Button-1>": self.plot_eigen})
        
                
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
        """Gets parameters for a given set of atoms"""
        try:
            self.masses,self.morse_params,self.plot_limits = params(self.a,self.b,self.c)
        except Exception:
            msgbox.showerror("Error", "Parameters for this atom combination not available!")
            raise
         
    def get_surface(self):
        """Get the full potential energy surface (Vmat) at specified grid points or rAB and rBC."""
        
        resl = 0.02 #Resolution
        
        #Get grid
        self.x = np.arange(self.plot_limits[0,0],self.plot_limits[0,1],resl)
        self.y = np.arange(self.plot_limits[1,0],self.plot_limits[1,1],resl)

        X,Y=np.meshgrid(self.x,self.y)

        self.Vmat=leps_energy(X,Y,self.theta,self.morse_params,self.H)

    def get_last_geo(self, *args):
        """Copy last geometry and momenta"""
        self.entries["xrabi"][0].delete(0, tk.END)
        self.entries["xrabi"][0].insert(0, self.trajectory[-1,0,0])
        
        self.entries["xrbci"][0].delete(0, tk.END)
        self.entries["xrbci"][0].insert(0, self.trajectory[-1,1,0])
        
        self.entries["prabi"][0].delete(0, tk.END)
        self.entries["prabi"][0].insert(0, self.trajectory[-1,0,1])
        
        self.entries["prbci"][0].delete(0, tk.END)
        self.entries["prbci"][0].insert(0, self.trajectory[-1,1,1])
            
    def export(self, *args):
        """Run calculation and print output in CSV format"""
        self._read_entries()

        coord_init=np.array([self.xrabi,self.xrbci,self.theta])
        # Set initial momenta (theta component = 0)
        mom_init=np.array([self.prabi,self.prbci,0])

        self.trajectory,error=calc_trajectory(coord_init,mom_init,self.masses,self.morse_params,self.H,self.steps,self.dt,self.calc_type)
        if error!='':
            msgbox.showerror(*error.split('::'))
        
        filename = asksaveasfilename(defaultextension=".csv")
        if not filename:
            return
            
        if self.calc_type == "Dynamics":
            header1="Time"
            first_column=self.dt*np.arange(len(self.trajectory))
        else:
            header1="Step"
            first_column=np.arange(len(self.trajectory))

        line1=header1+",AB distance,AB momentum,BC distance,BC momentum,theta,theta momentum,V energy,K energy,Tot energy"

        # calculate energies
        V=np.zeros(len(self.trajectory))
        K=np.zeros(len(self.trajectory))
        for i,point in enumerate(self.trajectory):
            V[i]=leps_energy(*point[:,0],self.morse_params,self.H)
            K[i]=kinetic_energy(point[:,0],point[:,1],self.masses)

        data=np.column_stack((first_column,np.reshape(self.trajectory,(len(self.trajectory),6))
                              ,V,K,V+K))

        np.savetxt(filename,data,delimiter=',',header=line1)
            
    def update_plot(self, *args):
        """Generate plot based on what type has been selected"""

        # Besides setting up information about initial position
        # this also reads GUI entries and gets relevant parameters
        # as it calls _read_entries() and get_params()
        self.update_geometry_info()

        # Check if atom types and collision angle have changed
        new_surf_params = (self.a, self.b, self.c, self.theta)
        if self.surf_params != new_surf_params:
            self.get_surface()

        # Check if need to calculate new trajectory
        coord_init=(self.xrabi,self.xrbci,self.theta)
        mom_init=(self.prabi,self.prbci,0) #Set initial momenta (theta component = 0)
        new_traj_params=(coord_init,mom_init,self.steps,self.dt,self.calc_type)
        if self.surf_params!=new_surf_params or self.traj_params!=new_traj_params: 
            self.trajectory,error=calc_trajectory(np.array(coord_init),np.array(mom_init),self.masses,
                                            self.morse_params,self.H,self.steps,self.dt,self.calc_type)
            if error!='':
                msgbox.showerror(*error.split('::'))

        self.surf_params=new_surf_params
        self.traj_params=new_traj_params
        
        # Set message to show when trajectory not shown
        warnmessage="The angle between bonds is changing along the simulation. \
                     Likely the initial collision angle is not 180°. \
                     Potential energy surfaces will change with time: surface at time 0 shown. \
                     The trajectory is not drawn in this plot."

        if self.plot_type == "Contour Plot":
            plot_contour(self.trajectory,self.x,self.y,self.Vmat,self.cutoff,self.spacing)
            if max(self.trajectory[:,2,0])-min(self.trajectory[:,2,0]) > 1e-7:
                msgbox.showinfo("Changing energy surfaces", warnmessage) 

        elif self.plot_type == "Surface Plot":
            plot_surface(self.trajectory,self.morse_params,self.H,self.x,self.y,self.Vmat,self.cutoff,self.spacing)
            if max(self.trajectory[:,2,0])-min(self.trajectory[:,2,0]) > 1e-7:
                msgbox.showinfo("Changing energy surfaces", warnmessage) 

        elif self.plot_type == "Skew Plot":
            plot_skew(self.trajectory,self.masses,self.x,self.y,self.Vmat,self.cutoff,self.spacing)
            if max(self.trajectory[:,2,0])-min(self.trajectory[:,2,0]) > 1e-7:
                msgbox.showinfo("Changing energy surfaces", warnmessage) 

        elif self.plot_type == "Internuclear Distances vs Time":
            plot_ind_vs_t(self.trajectory,self.dt,self.calc_type)

        elif self.plot_type == "Internuclear Velocities vs Time":
            plot_inv_vs_t(self.trajectory,self.masses,self.dt,self.calc_type)

        elif self.plot_type == "Momenta vs Time":
            plot_momenta_vs_t(self.trajectory,self.dt,self.calc_type)

        elif self.plot_type == "Energy vs Time":
            plot_e_vs_t(self.trajectory,self.masses,self.morse_params,self.H,self.dt,self.calc_type)

        elif self.plot_type == "p(AB) vs p(BC)":
            plot_momenta(self.trajectory)

        elif self.plot_type == "v(AB) vs v(BC)":
            plot_velocities(self.trajectory,self.masses)

        elif self.plot_type == "Animation":
            animation(self.trajectory,self.masses,[self.a,self.b,self.c],self.atom_map)

    def get_first(self):
        """Gather information about the initial state."""
        
        coord=np.array([self.xrabi,self.xrbci,self.theta])
        mom=np.array([self.prabi,self.prbci,0])

        V = leps_energy(*coord,self.morse_params,self.H)
        gradient = leps_gradient(*coord,self.morse_params,self.H)
        hessian = leps_hessian(*coord,self.morse_params,self.H)
        K = kinetic_energy(coord,mom,self.masses)
        
        return (V,gradient,hessian,K)
        
    def update_geometry_info(self, *args):
        """Updates the info pane"""
        self._read_entries()
        self.get_params()

        try:
            V,gradient,hessian,K = self.get_first()
            eigenvalues, eigenvectors = np.linalg.eig(hessian)
 
            self.init_point_curvature  = eigenvalues
            self.init_point_nmodes = eigenvectors
           
            ke     = "{:+7.3f}".format(K)
            pe     = "{:+7.3f}".format(V)
            etot   = "{:+7.3f}".format(V + K)
            fab    = "{:+7.3f}".format(-gradient[0])
            fbc    = "{:+7.3f}".format(-gradient[1])
           
            eval1  = "{:+7.3f}".format(eigenvalues[0])
            eval2  = "{:+7.3f}".format(eigenvalues[1])
           
            evec11 = "{:+7.3f}".format(eigenvectors[0,0])
            evec12 = "{:+7.3f}".format(eigenvectors[0,1])
            evec21 = "{:+7.3f}".format(eigenvectors[1,0])
            evec22 = "{:+7.3f}".format(eigenvectors[1,1])
            
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

    def plot_eigen(self, *args):
        """Plot eigenvectors and eigenvalues of the hessian on contour plot"""
        if not self.plot_type == "Contour Plot":
            return

        evecs=self.init_point_nmodes
        evals=self.init_point_curvature        

        plt.arrow(self.trajectory[0,0,0], self.trajectory[0,1,0], 
            evecs[0][0] / 10, evecs[0][1] / 10,
            color = "blue" if evals[0] > 0 else "red",
            label = "{:+7.3f}".format(evals[0]))
     
        plt.arrow(self.trajectory[0,0,0], self.trajectory[0,1,0], 
            evecs[1][0] / 10, evecs[1][1] / 10,
            color = "blue" if evals[1] > 0 else "red",
            label = "{:+7.3f}".format(evals[1]))
        
        plt.draw()
        plt.pause(0.0001)


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Starts the Triatomic LEPS GUI")
    parser.add_argument("-a", "--advanced", action="store_true", help="Include additional features in the GUI")
    
    args = parser.parse_args()
    interactive = Interactive(advanced = args.advanced)
 
