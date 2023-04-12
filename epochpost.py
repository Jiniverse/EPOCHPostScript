# epoch post module

### ========== Import Modules ========== ###
import numpy as np
import sdf
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.collections as mcoll
import matplotlib.path as mpth
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

font = {'size':14}
mpl.rc('font',**font)

from scipy import integrate
from scipy import special

from joblib import Parallel,delayed
### ==================================== ###


### =========== Constants =========== ###
# physics constants
qe	=	1.60217662e-19
me	= 	9.10938188e-31
epsilon0=  	8.8541878128e-12
mu0	= 	1.25663706212e-6
c 	= 	299792458
hbar 	= 	1.0545718e-34
# units
micro = 1.0e-6
micron = 1.0e-6
femto = 1.0e-15
# math constants
pi = 3.141592653589793
### ================================= ###

def printconst(): 	# print all the constants
    print('------------------------')
    print('Physics Constants:')
    print('-----')
    print('qe \t=',qe)
    print('me \t=',me)
    print('epsilon0=',epsilon0)
    print('mu0 \t=',mu0)
    print('c \t=',c)
    print('hbar \t=',hbar)
    print('------------------------')
    print('Units:')
    print('-----')
    print('micro \t=',micro)
    print('femto \t=',femto)
    print('------------------------')
    print('Math Constants:')
    print('-----')
    print('pi \t=',pi)
    print('------------------------')

### ========== Normalization factors ========== ###
def critical(lambda0):		# critical density
    omega0 = 2.0*pi*c/lambda0
    return omega0**2*me*epsilon0/qe**2

def norm_efield(lambda0):	# normalizaiton factor for electric field
    return 2*pi*me*c**2/qe/lambda0

def norm_bfield(lambda0):	# normalizaiton factor for magnetic field
    return 2*pi*me*c/qe/lambda0
### =========================================== ###

### ========== Read Data ========== ###
def readinput3d(inputfile):	# read inupt.deck file
    inputdata = {'size_x':[0.0,0.0],\
                 'size_y':[0.0,0.0],\
                 'size_z':[0.0,0.0],\
                 'grids':[0,0,0],\
                 'a0':0.0,'lambda0':0.0,\
                 'skip':[0,0,0]}
    with open(inputfile) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line.strip().startswith('a0'):
                a0 = float(line.split('=')[-1])
                inputdata['a0'] = a0
            elif line.strip().startswith('lambda0'):
                lambda0 = float( line.split('=')[-1].split('*')[0] )
                inputdata['lambda0'] = lambda0
            elif line.strip().startswith('x_min'):
                x_min = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_x'][0] = x_min
            elif line.strip().startswith('x_max'):
                x_max = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_x'][1] = x_max
            elif line.strip().startswith('y_min'):
                y_min = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_y'][0] = y_min
            elif line.strip().startswith('y_max'):
                y_max = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_y'][1] = y_max
            elif line.strip().startswith('z_min'):
                z_min = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_z'][0] = z_min
            elif line.strip().startswith('z_max'):
                z_max = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_z'][1] = z_max
            elif line.strip().startswith('nx'):
                nx = int(line.split('=')[-1])
                inputdata['grids'][0] = nx
            elif line.strip().startswith('ny'):
                ny = int(line.split('=')[-1])
                inputdata['grids'][1] = ny
            elif line.strip().startswith('nz'):
                nz = int(line.split('=')[-1])
                inputdata['grids'][2] = nz
            elif line.strip().startswith('skip_x'):
                skip_x = int(line.split('=')[-1])
                inputdata['skip'][0] = skip_x
            elif line.strip().startswith('skip_y'):
                skip_y = int(line.split('=')[-1])
                inputdata['skip'][1] = skip_y
            elif line.strip().startswith('skip_z'):
                skip_z = int(line.split('=')[-1])
                inputdata['skip'][2] = skip_z
    # inputdata = {'size_x':[x_min,x_max],\
    #              'size_y':[y_min,y_max],\
    #              'size_z':[z_min,z_max],\
    #              'grids':[nx,ny,nz],\
    #              'a0':a0,'lambda0':lambda0,\
    #              'skip':[skip_x,skip_y,skip_z]}
    return inputdata

def readinput2d(inputfile): # read inupt.deck file
    inputdata = {'size_x':[0.0,0.0],\
                 'size_y':[0.0,0.0],\
                 'grids':[0,0],\
                 'a0':0.0,'lambda0':0.0,\
                 'skip':[0,0]}
    with open(inputfile) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line.strip().startswith('a0'):
                a0 = float(line.split('=')[-1])
                inputdata['a0'] = a0
            elif line.strip().startswith('lambda0'):
                lambda0 = float( line.split('=')[-1].split('*')[0] )
                inputdata['lambda0'] = lambda0
            elif line.strip().startswith('x_min'):
                x_min = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_x'][0] = x_min
            elif line.strip().startswith('x_max'):
                x_max = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_x'][1] = x_max
            elif line.strip().startswith('y_min'):
                y_min = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_y'][0] = y_min
            elif line.strip().startswith('y_max'):
                y_max = float( line.split('=')[-1].split('*')[0] )
                inputdata['size_y'][1] = y_max
            elif line.strip().startswith('nx'):
                nx = int(line.split('=')[-1])
                inputdata['grids'][0] = nx
            elif line.strip().startswith('ny'):
                ny = int(line.split('=')[-1])
                inputdata['grids'][1] = ny
            elif line.strip().startswith('skip_x'):
                skip_x = int(line.split('=')[-1])
                inputdata['skip'][0] = skip_x
            elif line.strip().startswith('skip_y'):
                skip_y = int(line.split('=')[-1])
                inputdata['skip'][1] = skip_y
    # inputdata = {'size_x':[x_min,x_max],\
    #              'size_y':[y_min,y_max],\
    #              'size_z':[z_min,z_max],\
    #              'grids':[nx,ny,nz],\
    #              'a0':a0,'lambda0':lambda0,\
    #              'skip':[skip_x,skip_y,skip_z]}
    return inputdata


def sdfread(file_path='./',file_index=0,prefix=''): 
    """Read .sdf file."""
    file_name = file_path + prefix + str(file_index).zfill(4) + '.sdf'
    return sdf.read(file_name)
### =============================== ###

 
### ========== Plots ========== ###
label_map = {0:r'$t/T_0$',
             1:r'$x/\lambda_0$',
             2:r'$y/\lambda_0$',
             3:r'$z/\lambda_0$'}

cmap = plt.cm.jet
upper = cmap(np.linspace(0, 1, 1000))
middle = np.array([[1.-i*1./120, 1.-i*1./120, 1.-i*0.5/120,1.] for i in range(1,121)])
white = np.ones((10,4))
colors = np.vstack((white, middle, upper))
jetwhite = plc.LinearSegmentedColormap.from_list('jet_white', colors)

def set_label_xy(axis,x,y):
    axis.set_xlabel(label_map[x])
    axis.set_ylabel(label_map[y])
    
def EM_map(axis,x,y,EMmat,norm, field = r'$E_y$', xidx=1, yidx=2):
    emmap = axis.pcolormesh(x, y, EMmat, cmap='bwr', vmin=-norm,vmax=norm,shading='gouraud')
    axis.set_xlabel(label_map[xidx])
    axis.set_ylabel(label_map[yidx])
    cb_emmap = plt.colorbar(emmap,ax=axis)
    cb_emmap.set_label('Normalized '+field)
    return emmap,cb_emmap

def Density_map(axis,x,y,Densmat,vmin=0.0,vmax=0.0,particle = 'Electron', xidx=1, yidx=2,cmap=jetwhite):
    if(vmax==0.0):
        vmax = np.max(Densmat)
    profile_den = axis.pcolormesh(x, y, Densmat, cmap=cmap, vmin = vmin, vmax=vmax, shading='gouraud')
    axis.set_xlabel(label_map[xidx])
    axis.set_ylabel(label_map[yidx])
    cb_dens = plt.colorbar(profile_den,ax=axis)
    cb_dens.set_label(particle+' density'+' in '+r'$n_c$')
    return profile_den, cb_dens
### =========================== ###
