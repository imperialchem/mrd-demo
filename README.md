# LepsPy: A Molecular Reaction Dynamics Demonstration
A program to perform classical molecular reaction dynamics for a tri-atomic system using a London-Eyring-Polanyi-Sato (LEPS) potential parameterised for several atoms.

The first Python version of the code was written by Tristan Mackenzie, based on a Matlab version written by Lee Thompson, which in tern built upon a Fortran code written by Barry Smith.

### Running the program

Click on the **Clone or download** button to the right and download the zip archive with all the progeam files. You need to unpack the folder before you can run the program.

The program is run through a graphical user interface (GUI) which is started by running the file **lepsgui.py**.

#### On Windows

Double click on the **lepsgui.py** file to start the GUI.

#### On Linux and OSX

In a terminal, change directory to the LepsPy directory and execute "python lepsgui.py".


### Files

#### [lepsgui.py](./lepsgui.py)

This is the main program. lepsgui generates the GUI and plots, and drives the trajectory calculations.

#### [params.ini](./params.ini)

params.ini contains the parameter sets for a number of atom combinations. New atoms and parameters can be added to the program here.

#### [params.py](./params.py)

params.py reads params.ini and passes parameters to the lepsgui.

#### [lepspoint.py](./lepspoint.py)

lepspoint calculations the energy, first and second energy derivatives for any point on the surface.

#### [lepsmove.py](./lepsmove.py)

This file contains several functions related to the displacement of the system and its dynamic state.

