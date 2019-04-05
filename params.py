#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:30:28 2017

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

from configparser import ConfigParser
import numpy as np

class Params():
    def __init__(self, a, b, c):

        # Gets the parameters for any atom set or returns error message if no
        # parameters exist.
        
        #Open parameter file
        config = ConfigParser(inline_comment_prefixes=(';', '#'))
        config.read('params.ini')
        try:
            isotopes = config['isotopes']
        except:
            isotopes = {}
        
        # labels:
        # H F Cl D I O
        # 1 2 3  4 5 6
        
        
        self.name_a   = str(a)
        self.name_b   = str(b)
        self.name_c   = str(c)
        self.name_ab  = (self.name_a + self.name_b)
        self.name_bc  = (self.name_b + self.name_c)
        self.name_ac  = (self.name_a + self.name_c)
        self.name_abc = (self.name_a + self.name_b + self.name_c)
        
        #Replace atoms set by the isotopes section in parameters file
        
        for i, o in isotopes.items():
            self.ab  = self.name_ab. replace(i, o)
            self.bc  = self.name_bc. replace(i, o)
            self.ac  = self.name_ac. replace(i, o)
            self.abc = self.name_abc.replace(i, o)

        # Masses
        self.mass_a = self._get_mass(config, self.name_a)
        self.mass_b = self._get_mass(config, self.name_b)
        self.mass_c = self._get_mass(config, self.name_c)
        
        self.Masses = np.array([self.mass_a,self.mass_b,self.mass_c])

        self.reduced_mass_ab = (self.mass_a * self.mass_b) / (self.mass_a + self.mass_b)
        self.reduced_mass_bc = (self.mass_b * self.mass_c) / (self.mass_b + self.mass_c)
        self.reduced_mass_ac = (self.mass_a * self.mass_c) / (self.mass_a + self.mass_c)

        self.Reduced_masses = np.array([self.reduced_mass_ab,self.reduced_mass_bc,self.reduced_mass_ac])

        self.total_mass = self.mass_a + self.mass_b + self.mass_c

        # Morse Parameters
        Drab, lrab, Brab = self._get_morse(config, self.ab)
        Drbc, lrbc, Brbc = self._get_morse(config, self.bc)
        Drac, lrac, Brac = self._get_morse(config, self.ac)

        self.Dissociation_energies = np.array([Drab,Drbc,Drac])
        self.Reqs = np.array([lrab,lrbc,lrac])
        self.Morse_Parameters = np.array([Brab,Brbc,Brac])
        
        # Plot Limits
        self.mina, self.maxa, self.minb, self.maxb = self._get_limits(config, self.abc)

        self.surface_param = 0.424

    def _get_mass(self, config, key):
        
        try:
            d = config['atoms']
            l = d[key]
            m = float(l.split(',')[0])
        except KeyError:
            raise KeyError('Mass not available for atom type {}'.format(key))
        except:
            raise RuntimeError('Parameter file corrupted. Cannot get mass for atom type {}'.format(key))
        
        return m
        
    def _get_morse(self, config, key):
        
        try:
            d = config['morse']
            m = d[key]
            m = [float(p) for p in m.split(',')]
            assert len(m) == 3
        except KeyError:
            raise KeyError('Morse potential not available for atom pair {}'.format(key))
        except:
            raise RuntimeError('Parameter file corrupted. Cannot get morse parater for atom pair {}'.format(key))
        
        return m
        
    def _get_limits(self, config, key):
        
        try:
            d = config['limits']
            l = d[key]
            l = [float(p) for p in l.split(',')]
            assert len(l) == 4
        except KeyError:
            raise KeyError('Limits not available for atoms {}'.format(key))
        except:
            raise RuntimeError('Parameter file corrupted. Cannot get morse parater for atom pair {}'.format(key))
        
        return l
