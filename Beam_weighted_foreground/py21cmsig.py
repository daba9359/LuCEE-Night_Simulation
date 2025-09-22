
# boiler plate for most pylinex 21-cm stuff










import healpy as hp
from PIL import Image
import matplotlib.animation as animation
from astropy.io import fits
import os
import copy
from pylinex import Fitter, BasisSum, PolynomialBasis, MetaFitter, AttributeQuantity
from pylinex import Basis
from pylinex import TrainedBasis
import pylinex
import py21cmsig
import importlib
import corner
import lochness
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from importlib import reload
from pylinex import RepeatExpander, ShapedExpander, NullExpander,\
    PadExpander, CompiledQuantity, Extractor
import spiceypy as spice
from datetime import datetime
import enlighten
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pylinex import Fitter, BasisSum, PolynomialBasis
import perses
from perses.models import PowerLawTimesPolynomialModel
from pylinex import Basis
import perses
import ares
import camb
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Some basic constants
kb = 1.3807e-16    # Boltzman's constant [ergs per Kelvin]
kb_ev = 8.617e-5   # Boltzman's constant [electron volts per Kelvin]
h_pl = 6.6260755e-27 # Planck's constant [erg seconds]
c = 3e10        # speed of light [cm/s]
# Parameters  (Taking the Lambda CDM ones from Planck collaboration 2020 pg.19 (without BAO because of no real reason))
H0 = 67.36     # Hubble constant 
h = H0/100     # H0
omM0 = 0.3152   # Omega matter today
omB0 = 0.0493   # Omega baryons today 
omK0 = 0        # Omega curvature today
omC0 = 0.2645   # Omega cold dark matter today 
#omR0 = 8.98e-5  # Omega radiation today
omR0=8.600000001024455e-05  # Omega radiation from 21cm FAST
omL0 = 0.6847   # Omega Dark Energy today
f_He = 0.08    # Fraction of Helium by number
y_P = 0.2454  # Fraction of helium by mass (from Planck VI 2018) 
p_crit = 1.88e-29*h**2  # [g / cm^-3] The critical density of the universe
m_p = 1.6727e-24   #[g]  mass of a proton
m_pMev = 938.28   # [MeV/c^2] mass of a proton
m_e = 9.1094e-28   #[g]  mass of an electron
m_eMev = 0.511      # [Mev/c^2] mass of an electron
m_He = 6.644e-24   # [g] mass of helium
fsc = 1/137        # dimensionless fine structure constant
z_array = np.arange(20,1100)
#n_b0 = 2.06e-7   #[baryons per cubic centimeter] Old version from Joshua's comps paper that isn't a function of Omega baryons
n_b0 = (1-y_P)*omB0*p_crit/m_p+y_P*omB0*p_crit/m_He  # number density of baryons (by definition doesn't include electrons)
T_gamma0 = 2.725   # [Kelvin] modern CMB temperature
T_star = 0.068    # [Kelvin] the temperature equivalent of the energy difference of the 21 cm hyperfine spin states
A_10 = 2.85e-15    # [inverse seconds] Einstein coefficient for the spontaneous emission of the hyperfine spin states
# Less neccesary (or at least less understood by me but needed by some code)
As = 2.099e-9  # Amplitude of a power spectrum of adiabatic perturbations
ns = 0.9649    # Spectra index of a power spectrum of adiabatic perturbations

### Common equations:
H = lambda z,omR0,omM0,omK0,omL0: (H0*3.24078e-20)*(1+z)*np.sqrt(omR0*(1+z)**2+omM0*(1+z)+omK0+(omL0/((1+z)**2)))  # Standard Lambda CDM hubble flow in inverse seconds
## Some important functions 

# x_e
# for starters, I'm just going to use CosmoRec. We'll use camb later so it's more flawlessly integrated into python
#NOTE: I don't use the CosmoRec stuff anymore, but it's here if you want it.

# Alternative x_e:

xe_alt = lambda z: 0*(z/z)

# x_e using camb
parameters_camb = camb.set_params(H0=H0, ombh2=omB0*h**2, omch2=omC0*h**2, As=As, ns=ns)
camb_data= camb.get_background(parameters_camb)
camb_xe = camb_data.get_background_redshift_evolution(z_array, ["x_e"], format="array")
# or if you want an interpolated version:
camb_xe_interp = scipy.interpolate.CubicSpline(z_array,camb_xe.flatten())  # Needs a redshift argument

# T_gamma
T_gamma = lambda z: T_gamma0*(1+z)

### Here is the stuff that generally doesn't change

## Here are the k tables from Furlanetto 2006b

k_HH_raw = np.array([[1,1.38e-13],[2,1.43e-13],[4,2.71e-13],[6,6.60e-13],[8,1.47e-12],[10,2.88e-12],[15,9.10e-12],[20,1.78e-11],[25,2.73e-11],[30,3.67e-11],[40,5.38e-11],[50,6.86e-11],[60,8.14e-11],[70,9.25e-11],\
                 [80,1.02e-10],[90,1.11e-10],[100,1.19e-10],[200,1.75e-10],[300,2.09e-10],[500,2.56e-10],[700,2.91e-10],[1000,3.31e-10],[2000,4.27e-10],[3000,4.97e-10],[5000,6.03e-10]])
k_eH_raw = np.array([[1, 2.39e-10],[2,3.37e-10],[5,5.30e-10],[10,7.46e-10],[20,1.05e-9],[50,1.63e-9],[100,2.26e-9],[200,3.11e-9],[500,4.59e-9],[1000,5.92e-9],[2000,7.15e-9],[3000,7.71e-9],[5000,8.17e-9]])

# let's write a function that interpolates this table given whatever value we put in.
k_HH = scipy.interpolate.CubicSpline(k_HH_raw.transpose()[0],k_HH_raw.transpose()[1])   # Needs a temperature (or array of temps) as an argument
k_eH = scipy.interpolate.CubicSpline(k_eH_raw.transpose()[0],k_eH_raw.transpose()[1])   # Needs a temperature (or array of temps) as an argument

## n_H and n_e

n_H = lambda z,x_e: n_b0*(1+z)**3*(1-x_e(z))
n_e = lambda z,x_e: n_b0*(1+z)**3*(x_e(z))

# x_c
x_c = lambda z,x_e,T_k: (T_star)/(T_gamma0*(1+z)*A_10)*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))   # HH and eH

# T_S
T_S = lambda z,x_e,T_k: (1+x_c(z,x_e,T_k))/((1/(T_gamma0*(1+z)))+(x_c(z,x_e,T_k)/T_k(z)))

# But that's not the actual data. Need to include optical depth:

###
# z is your redshift (can be an array or single value)
# x_e is your fraction of free electrons functions (with z as an argument)
# T_k is your gas temperature functions (with z as an argument)
dTb = lambda z,x_e,T_k,omB0,omM0: 27*(1-x_e(z))*((h**2*omB0)/(0.023))*(((0.15)/(h**2*omM0))*((1+z)/(10)))**(1/2)*(1-((T_gamma0*(1+z))/(T_S(z,x_e,T_k))))

########### Foreground and Beam Related Constants ###################

NSIDE = 64 # resolution of the map
NPIX = hp.nside2npix(NSIDE)
NPIX   # total number of pixels (size of the array being used)
location = (-23.815,182.25)  # The lat lon location of the moon landing site
spice_kernels = "/home/dbarker7752/lochness/input/spice_kernels/" #location of the spice kernels
frequencies = np.arange(6,50,0.1)

# Cosmological Parameters
H0 = 67.36     # Hubble constant 
h = H0/100     # H0
omM0 = 0.3152   # Omega matter today
omB0 = 0.0493   # Omega baryons today 
omK0 = 0        # Omega curvature today
omC0 = 0.2645   # Omega cold dark matter today 
#omR0 = 8.98e-5  # Omega radiation today
omR0=8.600000001024455e-05  # Omega radiation from 21cm FAST
omL0 = 0.6847   # Omega Dark Energy today

### Boilerplate arrays for healpy (changes with a change in resolution)
thetas = hp.pix2ang(NSIDE,np.arange(NPIX))[0]*(180/np.pi)
phis = hp.pix2ang(NSIDE,np.arange(NPIX))[1]*(180/np.pi)
coordinate_array = np.ones((NPIX,2))
for i in np.arange(NPIX):
    coordinate_array[i] = np.array([phis[i],thetas[i]])

# HASLAM map
gal = perses.foregrounds.HaslamGalaxy()
haslam_data=gal.get_map(39.93) # gets the actual array of the data for that haslam map

# ULSA map
# ULSA_direction_raw = fits.open("/home/dbarker7752/21_cm_group/ULSA Maps/000.fits") # 32 bit version
ULSA_direction_raw = fits.open("/home/dbarker7752/21_cm_group/ULSA Maps/100.fits") # 64 bit version
ULSA_frequency = fits.open("/home/dbarker7752/21_cm_group/ULSA Maps/210.fits")
ULSA_constant = fits.open("/home/dbarker7752/21_cm_group/ULSA Maps/220.fits")

# This cell fixes the hole in the ULSA data via an interpolation

# This identifies the pixels of the dead zone
vec = hp.ang2vec(np.pi/2*1.1, -np.pi/2*1.05)
indices=hp.query_disc(nside=NSIDE,vec=vec,radius=0.1954)
hole_map = copy.deepcopy(ULSA_direction_raw[0].data[7])
hole_map[indices] = 10000000
hp.mollview(ULSA_direction_raw[0].data[7])
# These indices will be our region 10 which is the region we ignore
indices_deadzone = indices


x = np.arange(NPIX)
x = np.delete(x,indices_deadzone) # Gets rid of the dead zone
ULSA_min_deadzone = copy.deepcopy(ULSA_direction_raw[0].data)
for i,data in enumerate(ULSA_direction_raw[0].data):
    y = data
    y = np.delete(y,indices_deadzone)
    interpolator = scipy.interpolate.CubicSpline(x,y)
    for j in indices_deadzone:
        ULSA_min_deadzone[i][j] = interpolator(j)
hp.mollview(ULSA_min_deadzone[7])

ULSA_direction = ULSA_min_deadzone

# creates a list of all the beam file names.
path = "/home/dbarker7752/21_cm_group/Varied_Regolith/Beams"
files = []
for file in os.listdir(path):
    files.append(path+"/"+file)

# some other useful global variables
galaxy_map = ULSA_direction    # default galaxy map
test_times1 = [[2026,12,22,1,0,0]]   # list of times LOCHNESS will rotate the sky for 
frequency_array = np.array(range(1,51))   # list of frequencies we're evaluating at   

# modifies the galaxy map to not have the CMB (to make it consistent with the delta CMB convention of the signal)
galaxy_map_minCMB = copy.deepcopy(galaxy_map)
redshift_array = 1420.4/frequency_array-1
# This loop creates a CMB subtracted galaxy map to input into LOCHNESS. I've commented it out so you don't have to take 5 min to import this module
# for i,j in enumerate(redshift_array):
#     galaxy_map_minCMB[i] = galaxy_map[i] - py21cmsig.T_gamma(j)
# galaxy_map_minCMB[np.where(galaxy_map_minCMB<0.0)] = 0   # Gets rid of the negatives that plague this ULSA map (not sure why they are they)
# foreground_array_minCMB = lochness.LOCHNESS(spice_kernels,test_times1,location,galaxy_map=galaxy_map_minCMB).lunar_frame_galaxy_maps
# foreground_array_minCMB[np.where(foreground_array_minCMB<0.0)] = 0

# radiometer noise
sigT = lambda T_b, dnu, dt: T_b/(np.sqrt(dnu*dt))
# Noise parameters
dnu = 1e6
dt = 10000*3600 # first number is the number of hours of integration time

# Synchrotron Equation
synch = lambda f,A,B,c : A*(f/408)**(B+c*np.log(f/408))  # taken from page 6 of Hibbard et al. 2023 Apj. Arbitrarily chose 25 as my v0

# This identifies the pixels of the absorption region
vec = hp.ang2vec(np.pi/2, 0)
indices=hp.query_disc(nside=NSIDE,vec=vec,radius=0.85)
absorp_map = copy.deepcopy(ULSA_direction[7])
# absorp_map[indices] = 10000000
absorp_indices = indices[np.where(absorp_map[indices] < 1450000)][750:906]
absorp_map[absorp_indices] = 10000000

manager = enlighten.get_manager()
pbar = manager.counter(total=100, desc='Progress')

n_regions = 5
reference_frequency = 25


# Kinetic gas temperature with a  Runge-Kutta method of order 5(4)

def Tk (z_array,omR0,omM0,omK0,omL0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling and compton scattering. Only works for the cosmic 
    Dark Ages, as it does not include UV
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01
    omR0: Density prameter of radiation today
    omM0: Density parameter of matter today
    omK0: Density pramameter of curvature today
    omL0: Density parameter of Dark Energy today
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
                 redshift for its argument. Useful for future calculations.
    cosmological_parameters: An output of the cosmo parameters used to make the curve (just returns your density parameter inputs)"""
### Let's code up T_k 
    # some important functions
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    x_e = camb_xe_interp   # this is our model for fraction of free electrons
    adiabatic = lambda z,T:(1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*(2*H(z,omR0,omM0,omK0,omL0)*T)
    compton = lambda z,T: (1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))
    z_start = z_array[-1]
    z_end = z_array[0]

    ### Let's code up T_k
    ## The heating / cooling processes ##

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')  # solve_ivp is WAY WAY WAY faster and plenty precise enough for what we're doing

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    T_array = np.array([z,T])  

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument 
    cosmological_parameters = np.array([omR0,omM0,omK0,omL0])   
    return T_array, Tk_function, cosmological_parameters

My_Tk = Tk(z_array,omR0,omM0,omK0,omL0)

def lambdaCDM_training_set(frequency_array,parameters,N):
    """"Creates a training set based on the error range of the Lambda CDM cosmological constants for the fiducial 21 cm signal
    (no exotic physics).
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters. Make sure the order matches the order of the function
                As of writing this that order is: omR0,omM0,omK0,omL0,omB0. Shape should be, in this example, (5,2), with the first 
                column being the mean and the second being the standard deviation.
    N: The number of curves you would like to have in your training set. Interger
    bin_number: The number of bins in frequency space used to make the curves. Recommend 250 for LuSEE-Night as of the time writing this.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied fiducial 21 cm curves"""

    training_set = np.ones((N,len(frequency_array)))    # dummy array for the expanded training set
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    # Note: The divide by 2 is there 

    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
            new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
        training_set_params[n] = new_params

    redshift_array = 1420.4/frequency_array-1 
    redshift_array = redshift_array[::-1]      # need to convert to redshift since all of our functions are in redshift
    for n in tqdm(range(N)):
        R,M,K,L,B=training_set_params[n]
        T_k = Tk(redshift_array,R,M,K,L)[1]   # calculate our kinetic temperature to plug into the dTb function
        dTb_element=dTb(redshift_array,camb_xe_interp,T_k,B,M)*10**(-3)   # Need to convert back to Kelvin
        dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
        training_set[n] = dTb_element
    
    return training_set, training_set_params

# This is our custom T_k code with Dark Matter Self-Annihilation

def Tk_DMAN (z_array,f_dman_e_0,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    f_dman_e_0: Parameter governing the effects of this exotic model.

    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations."""

    num=len(z_array)
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    z_start = z_array[-1]
    z_end = z_array[0]

    # This defines our right hand side function
    delta_z = np.abs(z_array[1]-z_array[0])
    standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
    annihilation_dxe_dz = lambda z,xe: (0.0735*f_dman_e_0*((1-xe)/3)*(1+(z))**2*1/(1+f_He)*1/H(z,omR0,omM0,omK0,omL0)*1/(1-xe))*0.03  # self-annihilation addition to the rate of change very non physical right now
    func_xe = lambda z,xe: standard_dxe_dz(z)-(1/2*scipy.special.erf(0.01*(z-900))+1/2)*annihilation_dxe_dz(z,xe)  # total rate of change of free electrons

    # Initial conditions
    xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

    # Time span
    z_span = (z_start, z_end)

    # Solve the differential equation
    sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    xe = sol.y[0]

    
    xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
    x_e = xe_function
    xe_array = np.array([z,xe])
         
    g_h = lambda z: (1+2*x_e(z))/3


### Let's code up T_k
    ## The heating / cooling processes ##
    
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    dman = lambda zs,T,f_dman_e_0,g_h: (2/3)*(1/(H(zs,omR0,omM0,omK0,omL0)*kb_ev))*f_dman_e_0*g_h(zs)*1/(1+f_He+x_e(zs))*(1+zs)**2*0.3     # dark matter self-annihilation


    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) - dman(z,T,f_dman_e_0,g_h)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])   
    return Tk_array, Tk_function,xe_function, xe_array

def DMAN_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0):
    """"Creates a training set of singal curves based on the parameter range of the dark matter self-annihilation model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each row is a parameter and each column represents a value of that parameter that will be included
                in the interpolation to get new parameters. I don't linearly sample this because it usually weights the distribution
                heavily towards one end of the parameter space. Better to interpolate and space out the curves equally.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    
    # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    parameter_interpolators = {}
    for p in range(len(parameters)):
        x = range(len(parameters[p]))
        y = parameters[p]
        parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
        parameter_interpolators[p] = parameter_interpolator
    
    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        if gaussian:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
            training_set_params[n] = new_params
        else:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
            training_set_params[n] = new_params

    for n in tqdm(range(N)):
        fDMAN=training_set_params[n]
        DMAN_Tk = Tk_DMAN(redshift_array,fDMAN)  # calculate our kinetic temperature to plug into the dTb function
        dTb_element=dTb(redshift_array,DMAN_Tk[2],DMAN_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
        training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

def Tk_DMD (z_array,time_scale,C,omC0=omC0,h=h,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01
    time_scale: time parameter for dark matter decay.
    C: An arbitrary factor to make up for the odd units they've used in the original equation. Need to better understand why I have to do this.
    omC0: dark matter density parameter
    h: cosmological h
    
    
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations."""

    num=len(z_array)
    z_start = z_array[-1]
    z_end = z_array[0]
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    f_dmd_tau = time_scale**-1     # inverse of the time scale for convenience later
    g_h = 1/3  # amount of energy going to heating

    #This defines our right hand side function
    delta_z = np.abs(z_array[1]-z_array[0])
    standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
    decay_dxe_dz = lambda z,xe : C*(1+1100/z)**2*1/H(z,omR0,omM0,omK0,omL0)*f_dmd_tau*(xe/xe)  # self-annihilation addition to the rate of change very non physical right now (xe/xe is because you HAVE TO have the y variable in there, even if it's pointless)
    func_xe = lambda z,xe: standard_dxe_dz(z)-decay_dxe_dz(z,xe)  # total rate of change of free electrons  # total rate of change of free electrons
    # This defines our right hand side function

    # Initial conditions
    xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

    # Time span
    z_span = (z_start, z_end)

    # Solve the differential equation
    sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    xe = sol.y[0]

    
    xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
    x_e = xe_function
    xe_array = np.array([z,xe])
         
    g_h = lambda z: (1+2*x_e(z))/3

### Let's code up T_k
    # The heating / cooling processes ##
        
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    dmd = lambda zs,T,f_dmd_tau,g_h: (2/3)*(1/(H(zs,omR0,omM0,omK0,omL0)*kb))*(1.69e-8*f_dmd_tau*g_h(zs)*(omC0*h**2/0.12)*(1+1100/zs)**2)*200

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z_span = (z_start, z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) - dmd(z,T,f_dmd_tau,g_h)
    
    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])      
    return Tk_array, Tk_function,xe_function, xe_array

def DMD_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0,C=100000):
    """"Creates a training set of singal curves based on the parameter range of the dark matter decay model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then the first column is the minimum value and the second is the maximum. Each row is a different parameter.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    C: An arbitrary factor to make up for the odd units they've used in the original equation. Need to better understand why I have to do this.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    parameter_interpolators = {}
    for p in range(len(parameters)):
        x = range(len(parameters[p]))
        y = parameters[p]
        parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
        parameter_interpolators[p] = parameter_interpolator
    
    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        if gaussian:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
            training_set_params[n] = new_params
        else:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
            training_set_params[n] = new_params

    for n in tqdm(range(N)):
        fDMD=training_set_params[n]
        DMD_Tk = Tk_DMD(redshift_array,fDMD,C)  # calculate our kinetic temperature to plug into the dTb function
        dTb_element=dTb(redshift_array,DMD_Tk[2],DMD_Tk[1],B,M)*1e-3 # Need to convert back to Kelvin
        training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

# Millicoulumb simplified
# This is our custom T_k code with additional cooling

def Tk_cool_simp (z_array,C):
    """Creates an array evolving the IGM temperature based on adiabatic cooling and compton scattering. Only works for the cosmic 
    Dark Ages, as it does not include UV
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values
    C: Phenomenological parameter that represents the effect of additional cooling from this model
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature"""
### Let's code up T_k
    num=len(z_array)
    t_c = lambda z: 1.2e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering

    x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.zeros((num-1,2))



    ## T_X stuff (Temperature of dark matter)

    TX_array = np.zeros((num-1,2))
    rate_dm = lambda z: 1/(C)*(1/(1+z)**(1))*10e16
    
    ### Heating and Cooling Processes ###
    adiabatic = lambda z,T:(1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*(2*H(z,omR0,omM0,omK0,omL0)*T)
    compton = lambda z,T: (1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))
    Millicharged = lambda z,T,T_X: (1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*((T-T_X)/(rate_dm(z)))

    # Define the differential equation system
    def equations(z, y):
        T, TX = y
        dT_dz = adiabatic(z,T)-compton(z,T) + Millicharged(z,T,TX)
        dTX_dz = adiabatic(z,TX)-(omB0/(10*omC0))*Millicharged(z,T,TX)
        return [dT_dz, dTX_dz]

    # Initial conditions
    z0 = [3000, 3000]

    # Time span
    z_span = (1100, 20)

    # Solve the differential equation
    sol = solve_ivp(equations, z_span, z0, dense_output=True, method='Radau')

    # Access the solution
    t = sol.t
    T, TX = sol.y

    Tk_array = np.array([t,T])
    TX_array = np.array([t,TX])
    Tk_function=scipy.interpolate.CubicSpline(t[::-1],T[::-1])  # Turns our output into a function with redshift as an argument    
    return Tk_array, Tk_function, TX_array

def MCDM_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0):
    """"Creates a training set of singal curves based on the parameter range of the dark matter decay model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each column is a different value within that parameter space that will be interpolated.
                 This allows the parameter space to be sampled in a nonlinear way in order to avoid too many curves in one area. Each row is a different parameter.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    parameter_interpolators = {}
    for p in range(len(parameters)):
        x = range(len(parameters[p]))
        y = parameters[p]
        parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
        parameter_interpolators[p] = parameter_interpolator
    
    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        if gaussian:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
            training_set_params[n] = new_params
        else:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
            training_set_params[n] = new_params

    for n in tqdm(range(N)):
        MCDM_C=training_set_params[n][0]
        MCDM_Tk = Tk_cool_simp(redshift_array,MCDM_C)  # calculate our kinetic temperature to plug into the dTb function
        dTb_element=dTb(redshift_array,camb_xe_interp,MCDM_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
        training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

def Tk_EDE (z_array,Omega_ee,z_c):
    """Creates an array evolving the IGM temperature based on adiabatic cooling and compton scattering. Only works for the cosmic 
    Dark Ages, as it does not include UV
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    Omega_ee:  The density parameter of early dark energy at z=0

    z_c:  The redshift at which early dark energy's equation of state switches from 1 to -1 (turning point in the H function)
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations."""
### Let's code up T_k
    num=len(z_array)
    oee = Omega_ee
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    a_c = lambda z: 1/(1+z)
    x_e = camb_xe_interp
    H_EDE = lambda z: (H0*3.24078e-20)*np.sqrt(omR0*(1+z)**4+omM0*(1+z)**3+omK0*(1+z)**2+oee*((1+a_c(z_c)**6)/((1/(1+z))**6+a_c(z_c)**6)))
    z_start = z_array[-1]
    z_end = z_array[0]
    adiabatic = lambda z,T:(1/(H_EDE(z)*(1+z)))*(2*H_EDE(z)*T)
    compton = lambda z,T: (1/(H_EDE(z)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))

    ### Let's code up T_k
    ## The heating / cooling processes ##

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')  # solve_ivp is WAY WAY WAY faster and plenty precise enough for what we're doing

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    T_array = np.array([z,T])  

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument 
    cosmological_parameters = np.array([omR0,omM0,omK0,omL0])


    return T_array, Tk_function

def EDE_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0):
    """"Creates a training set of singal curves based on the parameter range of the dark matter self-annihilation model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each row is a parameter and each column represents a value of that parameter that will be included
                in the interpolation to get new parameters. I don't linearly sample this because it usually weights the distribution
                heavily towards one end of the parameter space. Better to interpolate and space out the curves equally.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""

    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    
    # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    parameter_interpolators = {}
    for p in range(len(parameters)):
        x = range(len(parameters[p]))
        y = parameters[p]
        parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
        parameter_interpolators[p] = parameter_interpolator
    
    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        if gaussian:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
            training_set_params[n] = new_params
        else:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*(len(parameters[k])-1)))
            training_set_params[n] = new_params

    for n in tqdm(range(N)):
        oee,z_c=training_set_params[n]
        EDE_Tk = Tk_EDE(redshift_array,oee,z_c)  # calculate our kinetic temperature to plug into the dTb function
        dTb_element=dTb(redshift_array,camb_xe_interp,EDE_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
        training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

# This will be our ERB model


def ERB_model (z_array,A_r_value,frequency,starting_point,smoothing,T_k=My_Tk[1],x_e=camb_xe_interp,smoothing_factor = 0):
    """Creates the Excess Radio Background model to see how it effects the 21cm dark ages trough
    
    Parameters
    ===================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    A_r_value: Non-dimensional amplitude of the ERB

    frequency: Has to do with the shape. It makes sure you get a good trough at that frequency. In MHz

    starting_point = the redshift it whicht to turn the ERB on (it isn't necessarily ubiquitous at all z)

    smoothing = this determines the value of the error function which determines how dramatic the smoothing for the curve
                is.  This is done to emulate the fact that the source of the ERB isn't necessarily going to immediately turn on,
                but will instead slowly turn on depending on how you adjust this number.

    Tk:  The function that creates your gas temperature.  Takes z as an argument, though other parameters are necessary for non standard Tk's.

    x_e:  The equation that defines your evolution of free electron fraction. Requires z as an argument
    ===================================================
    returns an array and spline function to plot"""

    # The upper limit of the 21 cm temperature:
    lambda_21 = 21  # wavelength of 21cm line [cm]
    max_temp = lambda z,x_e,T_k: -1000*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))*(3*h_pl*c*lambda_21**2*n_H(z,x_e)*T_star)/(32*np.pi*kb*(1+z)*H(z,omR0,omM0,omK0,omL0)*T_k(z))  #[mK]
    max_temp_interp=scipy.interpolate.CubicSpline(z_array,max_temp(z_array,camb_xe_interp,My_Tk[1]))

    alpha = -2.6   # dimensionless quantity that defines the spectral index of the signal
    nu_obs = lambda z: 1420/(1+z)  # [MHz] observed frequency of the 21 cm line
    #T_k = Tk_ERB(z_array,A_r,frequency)[1]   # converts the T_k raw function into the spline function
    x_c = lambda z,x_e,T_k: (T_star)/(T_gamma0*(1+z)*A_10)*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))
    A_r = lambda z: -A_r_value/2*scipy.special.erf(smoothing*(z-starting_point+smoothing_factor))+A_r_value/2
    T_R = lambda z: T_gamma(z)*(1+A_r(z)*(nu_obs(z)/frequency)**alpha)
 
    dTb = lambda z,x_e,T_k: 27*(1-x_e(z))*((h**2*omB0)/(0.023))*(((0.15)/(h**2*omM0))*((1+z)/(10)))**(1/2)*((x_c(z,x_e,T_k)*T_gamma(z)/T_R(z))/\
                                                                                                            (1+((x_c(z,x_e,T_k)*T_gamma(z)/T_R(z)))))*(1-(T_R(z)/T_k(z)))*1e-3  # convert to Kelvin
    
    ERB_function = scipy.interpolate.CubicSpline(z_array,dTb(z_array,x_e,T_k))
    return ERB_function

def ERB_training_set(frequency_array,parameters,N,T_k=My_Tk[1],x_e=camb_xe_interp,gaussian=False,B = omB0, M=omM0):
    """"Creates a training set of singal curves based on the parameter range of the dark matter decay model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then the first column is the minimum value and the second is the maximum. Each row is a different parameter.
    N: The number of curves you would like to have in your training set. Interger
    Tk:  The function that creates your gas temperature.  Takes z as an argument, though other parameters are necessary for non standard Tk's.
    x_e:  The equation that defines your evolution of free electron fraction. Requires z as an argument
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    parameter_interpolators = {}
    for p in range(len(parameters)):
        x = range(len(parameters[p]))
        y = parameters[p]
        parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
        parameter_interpolators[p] = parameter_interpolator
    
    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        if gaussian:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
            training_set_params[n] = new_params
        else:
            for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
            training_set_params[n] = new_params

    for n in tqdm(range(N)):
        z_s,Ar=training_set_params[n]
        ERB_function = ERB_model(redshift_array,Ar,78,z_s,0.2,T_k)  # calculates the ERB model
        ERB_element=ERB_function(redshift_array)  # Need to convert back to Kelvin
        training_set_rs[n] = ERB_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params, training_set_rs


#####################################################################################################################################################################
# Foreground and Beam Stuff #

def signal_training_set(path,foreground_array,times,frequency_bins,number_of_parameters,mask_negatives=True):
    """ Creates an array that includes the beam-weighted foreground arrays with respective times of all the listed files. NOTE: Is not general and only applies to Fatima's beams. Need to change this at some point.
    Parameters
    ========================================================================================================
    path: the path that contains all of the files you wish to compute the signal for. Must be a string. Right now I've only designed the function to read an entire folder.
    foreground_array: Array of the rotated galaxy foreground. You can create this by inputting your desired galaxy map and times into the LOCHNESS function
                     Example: foreground_array = LOCHNESS(spice_kernels,time_array,location,galaxy_map = my_galaxy_map).lunar_frame_galaxy_maps
    times: the array of times that you used for your 
     function. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    frequency_bins: the number of frequency bins. Should be an interger.
    number_of_parameters: The number of parameters in your model.
    mask_negative":  Whether or not to mask the negative values. Should be true for galaxy maps and false for cosmological signals (since they are delta TCMB, which can be negative and often is)
    
    Returns
    ========================================================================================================
    signal_master_array: An array of all the signals for each file 
    sma_minCMB:  Same as signal_master_array, but with the CMB subtracted off"""

    files = []
    
    for file in os.listdir(path):
        files.append(path+"/"+file)

    parameter_set = np.ones((len(files),number_of_parameters))  # NOTE: the 3 corresponds to the 3 parameters in Fatima's beams, so not general to any beam.
    signal_master_array = np.zeros((len(files),len(times),frequency_bins))  # place holder for now
    for i,f in tqdm(enumerate(files)):
        array_element=signal_from_beams(fits_beam_master_array(f),foreground_array,times,frequency_bins,mask_negatives)
        signal_master_array[i] = array_element
        parameters = np.array(fits_beam_master_array(f)[2])
        parameter_set[i] = parameters

    return signal_master_array, parameter_set

## This function coverts a weighted foreground set (with all frequencies) into a signal (frequency vs temperature)
def signal_from_beams(beam_array,foreground_array,time_array,desired_frequency_bins,mask_negatives=True,normalize_beam=True,rotate_beams=True):
    """Converts a weighted beam array into a monopole signal of frequency vs temperature
    
    Parameters
    =================================================================
    beam_array: An array of healpy maps for each frequency. The input should be from fits_beam_master_array. That function's output will match the required format of this input.
    foreground_array: Array of the rotated galaxy foreground. You can create this by inputting your desired galaxy map and times into the LOCHNESS function
                     Example: foreground_array = LOCHNESS(spice_kernels,time_array,location,galaxy_map = my_galaxy_map).lunar_frame_galaxy_maps
    time_array: Array of times you wish to evaluate this at. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    desired_frequency_bins: The number of frequency bins you desire.
    normalize_beam = Boolean as to wheter or not you would like to normalize the beam so that all weights add to 1. Sometimes beams are packaged in a way that actually
                     includes both the beam and the response function (how the antenna responds at each frequency), but technically the beam should not include this.
                     Normalization is set to True as default to remove the response function from beams. If it's already removed, it won't change anything anyways.
    rotate_beams: Whether or not to rotate the beams from a top zenith to a center of mollview zenith (usually have to do this with CEM produced beams).
    
    Returns
    =================================================================
    signal: An array for the signal (frequency vs brightness temperature)"""

#### NOTE: we need to mask the beam to make the negative numbers equal to 0. There is a glitch with that blank spot in the ULSA map where it wants to massively inflate the negative value

# now let's apply the ULSA map and weight it using the beam and collect all the frequencies into a single array:
    frequency_bins = beam_array[1].shape[0]  # picks out the number of frequency bins in Fatima's beams. Won't work if she changes their format
    times = time_array
    signal = np.ones((len(times),frequency_bins,NPIX))
    for f in range(frequency_bins):
        signal_element = time_evolution(beam_array[1][f],foreground_array,"N/A",f,"N/A",times,animation=False,normalize_beam=normalize_beam,rotate_beams=rotate_beams)
        signal[:,f] = copy.deepcopy(signal_element)
    if mask_negatives:
        signal[np.where(signal<0.0)] = 0  # Makes all negatives 0's as they should be. Negative temperature makes no sense in this case, though negative signal does since its delta Tb
    # now let's add up all the pixels and bin them per frequency to get our monopole signa
    signal_sum = np.ones((len(times),frequency_bins))
    for t in range(len(times)):
        for f in range(frequency_bins):
            if normalize_beam:
                signal_sum[t][f] = np.sum(signal[t][f])
            else:
                signal_sum[t][f] = np.sum(signal[t][f])/(NPIX/2) # this creates our array of summed up signals, NPIX/2 is so that we don't add all the 0's that aren't in the beam NOTE: different if you add horizon
    
    new_signal_sum = np.ones((len(signal_sum),desired_frequency_bins))  # creates a dummy array for later
    for t in range(len(signal_sum)):  #picks out the time array length
        SS_element_interpolator = scipy.interpolate.CubicSpline(range(1,len(signal_sum[0])+1),signal_sum[t]) # assumes we start at 1 MHz, creates the interpolation function
        SS_element = SS_element_interpolator(np.arange(1,51,(50/desired_frequency_bins)))    # creates the element that will replace the index of the dummy array. Assumes a start and end frequency of 1 and 50 MHz
        new_signal_sum[t] = SS_element 
    
    return new_signal_sum

def fits_beam_master_array (file_path):
    """Converts Fatima's beam files into a larger 3-D array of beams.
    Parameters
    ==============================================================================
    file_path: the path of the fits file of the beam. Must be a string.

    Returns
    ==============================================================================
    healpy_array = A healpy array that combines all the desired values
    beam_functions = an array of interpolations per frequency"""
    
    file = fits.open(file_path)   # opens the fit file to be used in our function
    data = file[8].data/(4*np.pi)  # normalizes the data. Note that the [8] is the gain part of this particular fits file convention (Fatima decided this convention)
    beam_functions = []
    ### This portion of the code makes the interpolation objects that will be combined together and converted to healpy arrays later.

    ## This mess creates our y array. Its takes very little time just way more lines of code than I think I actually need most likely
    array1 = np.zeros(361)
    for i in range(len(data[0])+90):  # we have to add the 90 here to get the values below the horizon. Fatima doesn't include below horizon, have to add it in ourselves.
        if i != 0:
            array_element=np.ones(361)*i
            array1=np.append(array1,array_element)

    array2 = np.arange(0,361)
    for j in range(len(data[0])+90):
        if j != 0:
            array_element = np.arange(0,361)
            array2 = np.append(array2,array_element) 
    ## ## ##

    y = np.array((array2,array1)).transpose()
    
    zeros = np.zeros(32490)
    for j in range(len(data)):
        d = data[j].flatten()  # creates our data array for plugging into the interpolator
        d= np.append(d,zeros)
        beam_function = scipy.interpolate.RBFInterpolator(y,d,neighbors=10)  # creates a function for the beams using an interpolator.
                                                                        # requires a 2-D array: np.array([phi,theta]) as an input.
        beam_functions.append(beam_function)

    healpy_array = np.array([ang2pix_interpolator(beam_functions[0])])
    for i in range(len(data)):
        if i == 0:
            None
        else:
            healpy_array=np.concatenate((healpy_array,np.array([ang2pix_interpolator(beam_functions[i])])),axis=0)
    parameter_array = []
    parameter_array.append((file[0].header["L"],file[0].header["TOP"],file[0].header["BOTTOM"])) 
    return beam_functions, healpy_array, parameter_array

# let's create an animation of the time evolution of a specific beam weighted foreground at a specific frequency
def time_evolution (beam,foreground_array,save_location,frequency,label,time_array,location = location,norm=None,max=None,animation=True,normalize_beam=True,rotate_beams=True):
    """Creates the  for a specific master beam
    
    Parameters
    =============================================================================
    beam: the healpy array that is to be mapped onto the sky. Should be (NPIX) shape. Just the one healpy array. Assumes zenith is at the top 
          edge of the mollview (function will rotate it to the center).
    foreground_array: the healpy array that is the galactic foreground, but already rotated. Should be (time steps,freqeuncy bins,NPIX) shape.
                      NOTE: This could be calculated within this function, but it saves time to do it outside if your calculating 
                            multiple beams at the same timestep, then you can apply this to each of them instead of calculating each time.
    save_location: location you whish to save these plots (string)
    frequency: The frequency to evaluate at. Interger
    label: legend label of each plot (first part at least)
    time_array: The list of times that you wish to evaluate at.
    location: The lat lon of the LuSEE-Night lander. Default is the latest value I've seen from the mission details.
    max: The max value displayed on the mollview map.
    animation = Boolean as to whether or not you would like an animation of the beams to be made, cycling through frequency
    normalize_beam = Boolean as to wheter or not you would like to normalize the beam so that all weights add to 1. Sometimes beams are packaged in a way that actually
                     includes both the beam and the response function (how the antenna responds at each frequency), but technically the beam should not include this.
                     Normalization is set to True as default to remove the response function from beams. If it's already removed, it won't change anything anyways.
    rotate_beams: Whether or not to rotate the beams from a top zenith to a center of mollview zenith (usually have to do this with CEM produced beams).
    Return
    =============================================================================
    saves the plots to the designated folder and also creates an animation in the same folder"""

    # Let's do some plotting
    foreground_array_mod = copy.deepcopy(foreground_array[:,frequency])  # this assumes that each index number associates the the same frequency number
                                                                        # NOTE: The deepcopy makes sure changes to foreground_array_mod don't change the original foreground_array
    beam_euler_angle = [0,90,90] # this rotates only the beam, not the galaxy, in order to match the convention of zenith being the center of the map
    if rotate_beams:
        rotated_beam = hp.Rotator(rot=beam_euler_angle).rotate_map_pixel(beam)
    rotated_beam = beam
    if animation:
        for i,j in  enumerate(foreground_array_mod):
            if normalize_beam:
                foreground_array_mod[i] = j*rotated_beam/np.sum(rotated_beam) 
            else:
                foreground_array_mod[i] = j*rotated_beam
            hp.mollview(foreground_array_mod[i],title=label+ f" at {frequency}" + f" time step {i}",unit=r"$T_b$",min=0,norm=norm,max=max)
            plt.savefig(save_location+f"/{frequency}+MHz"+f"_time_step_{i}.png")
            plt.close()

        animate_images_time(save_location,save_location+"Animation.gif",time_array, frequency)
    else:
        for i,j in  enumerate(foreground_array_mod):
            if normalize_beam:
                foreground_array_mod[i] = j*rotated_beam/(np.sum(rotated_beam))
            else:
                foreground_array_mod[i] = j*rotated_beam       
    return foreground_array_mod

def ang2pix_interpolator (data,coordinates=coordinate_array,normalization = 1):
    """Converts a 3-D beam map into a healpy format (1-D array per frequency, so technically 3-D to 2-D)
    
    Parameters
    =============================================================================================
    data: an interpolation function that takes a 2D array as its argument such as np.array([altitude, azimuth])
    normalization: The number to divide by to normalize the data. Default = 1 (assumes a normalized gain array)
    =============================================================================================
    Returns
    =============================================================================================
    data_healpy_map:  a 2-D array in the shape (frequency, 1-D healpy_map)"""

    # the point is to be able to input any size of beam array and not have to worry about empty spaces due to pixels not being filled it
    # this means we need to fill in the data if it hasn't been provided, which is very easy with an interpolation
    ### Interpolation
    data_healpy_map = data(coordinates)/normalization

    return data_healpy_map

def animate_images_time(image_folder, output_path, time_array,frequency, frame_duration=200):
    """
    Animates images in a folder and saves the animation as a GIF.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_path (str): Path to save the output GIF file.
        time_array: The array of the times at which to evaluate.
        frequency: The frequency at which to evaluate.
        frame_duration (int, optional): Duration of each frame in milliseconds. Defaults to 200.
    """
    # image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))],key=int)
    image_files = []
    for i in range(len(time_array)):
        image_files.append(image_folder+f"{frequency}"+f"_time_step_{i}.png")

    fig, ax = plt.subplots()
    ims = []
    for image_file in image_files:
        img = Image.open(image_file)
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=frame_duration, blit=True, repeat_delay=1000)
    ani.save(output_path, writer='pillow')
    plt.close(fig)

def simulation_run_raw (frequencies,beam_file,foreground_array,time_array,dnu,dt,omR0,omM0,omK0,omL0,omB0):
    """Creates a simulated data curve.
    
    NOTE: Not general right now. Only works in the range of 1-50 MHz. Easy fix, but don't want to do that right now, since it's not needed.
    NOTE: Also not general because the radiometer noise is built into the function (which is fine for anything I'll be doing). 

    Parameters:
    ====================================================================
    training_set_curves: The full list of curves that Pylinex used to fit its model.  Shape (number of curves, time stamps, frequency bins)
    training_set_parameters: The associated parameters with the training set curves. Shape (number of curves, number of parameters)
    beam_file = The file that contains the beam_arrays
    foreground_array: the healpy array that is the galactic foreground, but already rotated. Should be (time steps,NPIX) shape.
                      NOTE: This could be calculated within this function, but it saves time to do it outside if your calculating 
                            multiple beams at the same timestep, then you can apply this to each of them instead of calculating each time.
    time_array: Array of times you wish to evaluate this at. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    dnu: The bin size of the frequency bins. For the noise function.
    dt: Integration time. For the noise function.
   
    
    Returns
    ====================================================================
    simulated_data = An array of the simulated data as Temperature vs Frequency"""

    # Noise function
    sigT = lambda T_b, dnu, dt: T_b/(np.sqrt(dnu*dt))

    # Loads in a beam and a signal
    beam=fits_beam_master_array(beam_file)  # loads in a test beam
    redshift_array = 1420.4/np.arange(1,51)
    redshift_array = redshift_array[::-1]      # need to convert to redshift since all of our functions are in redshift
    dTb=py21cmsig.dTb(redshift_array,py21cmsig.camb_xe_interp,py21cmsig.Tk(redshift_array,omR0,omM0,omK0,omL0)[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
    dTb=dTb[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
    redshift_array_expanded = 1420.4/frequencies
    redshift_array_expanded = redshift_array_expanded[::-1]      # need to convert to redshift since all of our functions are in redshift
    dTb_expanded=py21cmsig.dTb(redshift_array_expanded,py21cmsig.camb_xe_interp,py21cmsig.Tk(redshift_array_expanded,omR0,omM0,omK0,omL0)[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
    dTb_expanded=dTb_expanded[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)

    # Let's now add this to our galaxy map
    signal_foreground_array = np.zeros_like(foreground_array)
    for t in range(len(foreground_array)):
        for i,j in enumerate(dTb):
            signal_foreground_array[t][i] = foreground_array[t][i]+j   # adds the signal to each frequency

    # Now let's weight this using a beam and LOCHNESS to rotate it
    simulation_test_raw = signal_from_beams(beam,signal_foreground_array,time_array,50) #the non-interpolated version with only 50 frequency bins
    simulation_test_interp = scipy.interpolate.CubicSpline(range(1,51),simulation_test_raw[0])
    simulation_test = simulation_test_interp(frequencies)
    simulation_test_no_noise = copy.deepcopy(simulation_test)
    # Now we add radiometer noise

    for i in range(len(frequencies)):
        simulation_test[i] = np.random.normal(simulation_test[i],sigT(simulation_test[i],dnu,dt))

    signal_only = dTb_expanded   # the non-interpolated version with only 50 frequency bins
    foreground_only_raw = signal_from_beams(beam,foreground_array,time_array,50)
    foreground_interp = scipy.interpolate.CubicSpline(range(1,51),foreground_only_raw[0])
    foreground_only = foreground_interp(frequencies)
    noise_only = simulation_test - signal_only - foreground_only

    return simulation_test, signal_only, foreground_only, noise_only, simulation_test_no_noise, simulation_test_raw

def make_foreground_model(frequencies,n_regions,sky_map,reference_frequency,rms_mean,rms_std,absorption_region = True,\
                           absorp_indices = None,plot_regions=True,scale=0.1,ev_num=1e6,show_region_map=True):
    """Make a foreground model from a number of regions (add an absorption region if you'd like). Also can define a reference frequency.
    This function's primary purpose is to return a Bayesian evidence value that can be used to compare different foreground models.
    
    Parameters
    =============================================
    frequencies: Frequencies you're evaluating at. Array
    n_regions: The number of regions you want in your foreground model
    sky_map: Your reference model for your regions. Examples include Haslam, Guzman, GMS, ULSA, etc. Already rotated into the correct time.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    rms_std: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms. 
    synchrotron_parameters: The parameters for the synchrotron equation. Need to be an array of the shape (3,)
                            Parameters are (amplitude,spectral index,spectral curvature)
    absorption_region: Boolean of whether you want an additional region for the absoprtion zone at the center of the galaxy
                       NOTE: This isn't general right now, and only works if your resolution is 32
    absorpt_indices: The indices of the 32 bit map that include the absorption region
    plot_regions: Whether or not you'd like a plot of the regions on the sky.
    scale: The variation from the best fit of the best fit paramters when making new curves for the evidence.
    ev_num: Number of curves to make for the evidence set.
    show_region_map: Whether or not to show the sky view of the regions.
    

    Returns
    =============================================
    """

    # Synchrotron Equation:
    temps = np.sum(sky_map[frequencies[0]-1:frequencies[-1]+1],axis=1)/NPIX # temps from sky map
    noise = sigT(temps,dnu,dt)  # usually globally defined
    synchrotron = synch  # globally defines variable

    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency],n_regions)
    indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region

    # creates an absorption region at the center of the plane of the milky way (where the synchrotron radiation is heavily absorbed)
    if absorption_region:
        for r in range(len(indices)):
            for v in absorp_indices: # removes the absorption region from the low number region
                if v in indices[r]: 
                    indices[r].remove(v)
        indices["a"] = absorp_indices
    
    # plots the regions on the sky map
    if plot_regions:
        region_map=np.zeros_like(sky_map[reference_frequency])
        for r,v in zip(indices,range(len(indices))):
            region_map[indices[r]] = v
        if show_region_map:
            hp.mollview(region_map)

    # creates a curve of the sky map, per region, which will be the data we want to fit against
    total_signal = np.zeros((len(indices),len(frequencies)))
    for i,r in enumerate(indices):
        signal_sum = np.array([])
        for f in frequencies-1:
            signal_sum_element = np.sum(sky_map[f][indices[r]]/len(indices[r]))
            signal_sum = np.append(signal_sum,signal_sum_element)
        total_signal[i] = signal_sum
    # Identifies the best fit using a least squares for the synchrotron parameters 

    best_fit_params = np.zeros((len(indices),3)) # 3 parameters in the synchrotron equation
    best_fit_params_error = np.zeros((len(indices),3,3))
    best_fit_curves = np.zeros((len(indices),len(frequencies)))
    for p in range(len(indices)):
        best_fit_params[p]=scipy.optimize.curve_fit(synchrotron,frequencies,total_signal[p],sigma=noise)[0]
        best_fit_params_error[p] = scipy.optimize.curve_fit(synchrotron,frequencies,total_signal[p],sigma=noise)[1]
        best_fit_curves[p] = synchrotron(frequencies,best_fit_params[p][0],best_fit_params[p][1],best_fit_params[p][2])

    # root mean square of each regions best fit curve
    region_rms = np.array([])
    for j in range(len(indices)):
        rms = np.sqrt(np.mean((best_fit_curves[j]-total_signal[j])**2))
        region_rms = np.append(region_rms,rms)

    # evidence of this model
    model_priors = 1/ev_num
    model_likelihood = 0
    curves = np.zeros((ev_num,len(frequencies)))
    for n in tqdm(range(int(ev_num))):
        region_element = np.zeros(len(frequencies))
        for i,c in enumerate(indices): # creates the temperature vs frequency curves for new, varied parameters
            region_element += synchrotron(frequencies,best_fit_params[i][0]+best_fit_params[i][0]*(2*scale*np.random.random()-scale),best_fit_params[i][1]+best_fit_params[i][1]*(2*scale*np.random.random()-scale),\
                                          best_fit_params[i][2]+best_fit_params[i][2]*(2*scale*np.random.random()-scale))*len(indices[c])/NPIX
        curves[n] = region_element
    stats = calculate_rms(curves,temps,rms_mean,rms_std) # calculates p-values
    model_likelihood = np.sum(stats[2])
    evidence = model_likelihood*model_priors  # NOTE: only works because all parameters have the same probability

    return best_fit_params, best_fit_curves, region_rms, total_signal, region_element, best_fit_params_error, temps, curves, \
        stats, evidence, indices

def B_value_interp(beam_sky_training_set,beam_sky_training_set_params,\
                         frequencies,sky_map,reference_frequency,n_regions):
    """Interpolates the beam weighting per region for the synchrotron_foreground
    
    Parameters
    ============================================
    frequencies: The array of frequencies you wish to evaluate at.
    n_regions: The number of regions you want in your foreground model
    sky_map: Your reference model for your regions. Examples include Haslam, Guzman, GMS, ULSA, etc. Already rotated into the correct time.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_sky_training_set_params: The parameters associated with the beam_sky_training_set. Should be shape (n curves,n parameters per curve)
    beam_curve_training_set: This is the training set that is temperature vs frequency. You'll need this one as well. This one need not
                            be the raw training set. Should be shape (n curves, frequency bins)

    Returns
    ============================================"""

    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency],n_regions)
    new_region_indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region
    t = 0
    B_values_raw = np.zeros((len(beam_sky_training_set),len(beam_sky_training_set[0]),len(new_region_indices)))
    B_values = np.zeros((len(beam_sky_training_set),len(frequencies),len(new_region_indices)))
    for n in tqdm(range(len(beam_sky_training_set))):
        for f in range(len(beam_sky_training_set[0])):
            for i,r in enumerate(new_region_indices):
                B_values_raw[n][f][i]=np.sum(beam_sky_training_set[n][f][new_region_indices[r]])
        B_values_interp = scipy.interpolate.CubicSpline(np.arange(1,len(beam_sky_training_set[0])+1),B_values_raw[n])
        B_values[n] = B_values_interp(frequencies)
    expanded_B_values_interpolator = {}
    for f in range(len(frequencies)):
        values = B_values[:,f]
        params = beam_sky_training_set_params
        expanded_B_values_interp=scipy.interpolate.NearestNDInterpolator(params,values)
        expanded_B_values_interpolator[f]=expanded_B_values_interp

    return expanded_B_values_interpolator

def synchrotron_foreground_forsigex(n_regions,frequencies,reference_frequency,sky_map, BTS_curves, BTS_params,\
                           beam_sky_training_set,beam_sky_training_set_params,N,parameter_variation,B_value_functions\
                            ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True):
    """Creates a training set for the five region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_sky_training_set_params: The parameters associated with the beam_sky_training_set. Should be shape (n curves,n parameters per curve)
    N: Number of varied foregrounds you wish to have in the training set.
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    Returns
    ======================================================="""
    t=0
    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency-1],n_regions) # define the regional patches
    B_values = np.zeros((len(BTS_curves),len(frequencies),n_regions))
    for i,b in enumerate(BTS_params):
        for f in range(len(frequencies)):
            B_values[i][f] = B_value_functions[f](b)
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    masked_indices = np.where(beam_sky_training_set[0][-1] == 0)[0]

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synch(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synch(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp
    


    # This loop weights the new change in mean temperature per region with the beam value associated

    new_curves = np.zeros((len(B_values),N,len(frequencies)))
    for b in tqdm(range(len(BTS_curves))):
        weighted_deltaT = np.zeros((N,len(frequencies)))
        for n in range(N):
            for r in range(n_regions):
                weighted_deltaT[n] += new_foreground_deltaT[n][r]*B_values[b,:,r]
            #     training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            # training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            # training_set_params = np.concatenate((training_set_params,[training_set_params_row]),axis=0)
            new_curves[b][n] = BTS_curves[b]+weighted_deltaT[n]
            # training_set = np.concatenate((training_set,[new_curves[b][n]]),axis=0)



    # This loop takes a wierdly long amount of time to run and just massages the arrays into the proper format for PYLINEX

    training_set_size = len(BTS_curves)*N
    parameter_length = len(new_parameters[0][0])*n_regions+len(BTS_params[0])
    training_set = np.zeros((training_set_size,len(frequencies)))
    training_set_params = np.zeros((training_set_size,parameter_length))
    x = -1
    for b in tqdm(range(len(BTS_curves))):
        for n in range(N):
            training_set_params_row = np.array([])
            x += 1
            for r in range(n_regions):
                training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            training_set[x] = new_curves[b][n]
            training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            training_set_params[x] = training_set_params_row
    if print_parameter_variation:
        print(parameter_variation)

    

    return optimized_parameters,new_curves,masked_indices, training_set, training_set_params

## NOTE: This doesn't include multiple time stamps. Do we need to even bother? Something to think about.
def expanded_training_set_no_t(STS_data,STS_params,N,custom_parameter_range=np.array([0]),show_parameter_ranges=False):
    """Convert a signal_training_set output into a much larger training set by interpolating over the parameters per frequency
    This is basically a 1 dimensional MEDEA.
    
    Parameters
    ====================================================
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    custom_parameter_range: This will replace the automatically generator parameter range. Make sure it will still be within
                            the parameter range of the training set and is of the correct shape. Default to False since it's 
                            a bit more advanced of a parameter. Shape is (n parameters, 2(min and max))

    Returns
    ====================================================
    expanded_training_set: A new training set interpolated from the old with N curves
    expanded_training_set_params: The associated parameters of the new training set
    new_data: This is an interpolator that allows you to plug in any parameter value and get the curve. Handy for some investigation work"""

    param_value_ranges_array = np.ones((len(STS_params[0]),2))   # dummy arrray for parameter ranges that will be populated later
    
    if custom_parameter_range.any() == 0:  # checks to see if you've set a cust range of parameters for the expanded training set
        for i in range(len(STS_params[0])):   # Here we start to populate the array of parameter ranges. I call [0] because all entries should have the same number of parameters
            pr_element = [STS_params[:,i].min(),STS_params[:,i].max()]
            param_value_ranges_array[i] = pr_element
    else:
        param_value_ranges_array=custom_parameter_range

    expanded_training_set = np.ones((N,len(STS_data[0])))    # dummy array for the expanded training set
    expanded_training_set_params = np.ones((N,len(STS_params[0])))   # dummy array for the parameters of this expanded set

    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        for k in range(len(STS_params[0])):  # this will create a new set of random parameters for each instance
            p = param_value_ranges_array[k]
            new_params = np.append(new_params,np.random.random()*(p[1]-p[0])+p[0])
        expanded_training_set_params[n] = new_params  
    for f in tqdm(range(len(STS_data[0]))):    # loops through all frequencies
        values = STS_data[:,f]
        new_data=scipy.interpolate.griddata(STS_params,values,expanded_training_set_params)
        expanded_training_set[:,f] = new_data 
    if show_parameter_ranges:
        print(param_value_ranges_array)

    return expanded_training_set, expanded_training_set_params

def simulation_run (weighted_foreground,signal_model,dnu,dt):
    """Creates a simulated data curve.
    
    NOTE: Not general right now. Only works in the range of 1-50 MHz. Easy fix, but don't want to do that right now, since it's not needed.
    NOTE: Also not general because the radiometer noise is built into the function (which is fine for anything I'll be doing). 

    Parameters:
    ====================================================================
    training_set_curves: The full list of curves that Pylinex used to fit its model.  Shape (number of curves, time stamps, frequency bins)
    training_set_parameters: The associated parameters with the training set curves. Shape (number of curves, number of parameters)
    beam_file = The file that contains the beam_arrays
    foreground_array: the healpy array that is the galactic foreground, but already rotated. Should be (time steps,NPIX) shape.
                      NOTE: This could be calculated within this function, but it saves time to do it outside if your calculating 
                            multiple beams at the same timestep, then you can apply this to each of them instead of calculating each time.
    signal_model: The signal model to use to add the signal into the simulation. Will be a curve of shape (frequencies).
    time_array: Array of times you wish to evaluate this at. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    dnu: The bin size of the frequency bins. For the noise function.
    dt: Integration time. For the noise function.
   
    
    Returns
    ====================================================================
    simulated_data = An array of the simulated data as Temperature vs Frequency"""

    # Noise function
    sigT = lambda T_b, dnu, dt: T_b/(np.sqrt(dnu*dt))

    simulation_no_noise = weighted_foreground + signal_model
    simulation = np.zeros_like(simulation_no_noise)
    # Now we add radiometer noise

    for i in range(len(weighted_foreground)):
        simulation[i] = np.random.normal(simulation_no_noise[i],sigT(simulation_no_noise[i],dnu,dt))

    signal_only = signal_model
    foreground_only = weighted_foreground
    noise_only = simulation - signal_only - foreground_only

    return simulation, signal_only, foreground_only, noise_only, simulation_no_noise

def calculate_rms(curves,reference_curve,rms_mean,rms_std,curve_parameters=None):
    """Calculates the rms, z_score, and p_value for a specific curves vs a reference curve
    
    Parameters
    =============================================
    curves: An array of shape (number of curves, curve_array)
    curve_parameters: An array of the parameters associated with each curve. Not always necessary, so I've defaulted them to None.
    reference_curve: The curve you're comparing all the other curves to
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    rms_std: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms. 

    Returns
    =============================================
    rms_array: List of rms values per curve
    sorted_rms: List of rms values sorted from lowest to highest
    p_value_array: List of p_values for each curve.
    """
    
    rms_array = np.array([])
    z_score_array = np.array([])
    p_value_array = np.array([])
    for n in tqdm(range(len(curves))):    
        rms = np.sqrt(np.mean(curves[n]-reference_curve)**2)
        rms_array = np.append(rms_array,rms)
        z_score = np.abs(rms-rms_mean)/rms_std
        z_score_array = np.append(z_score_array,z_score)
        p_value = scipy.stats.norm.sf(z_score)
        p_value_array = np.append(p_value_array,p_value)

    return rms_array, z_score_array, p_value_array, curve_parameters

def narrowed_training_set(data,rms_mean,one_sigma_rms,training_set,training_set_parameters,sigma_tolerance = 5):
    """This uses the rms of the residuals to narrow the training set so that the included curves are only the curves within some 
     defined sigma of the rms of the noise.  This is useful for hammering down the wildly inaccurate curves from the training set
      so that PYLINEX doesn't lose its mind over them. PYLINEX does not do well with too large of a parameter space. Note that due
      to some issues with arrays, this only handles one time stamp at a time. You will need to loop through this function to 
      do all time stamps.
       
    Parameters
    ============================================================
    data: The simulated or real data that our training set will attempt to fit. Should be an array of the shape (number of time steps, frequency bins)
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    one_sigma_rms: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms.
    training_set: The first return of the signal_trainin_set function or expanded_training_set. The array that contains all of the curves that
                  your are attempting to fit the data to. Should be an array of the shape (number of curves, number of time steps, frequency bins)
    training_set_parameters: the second return of the signal_training_set function or expanded_training_set. The array should contain all of the
                            parameters associated with each individual training set curve. Should be size (number of curves, number of parameters)
    sigma_tolerance: The number of sigma from the data fit residual that you would like to include in the new, narrowed training set. 
                     Interger or float.
                  
    Returns
    =============================================================
    narrowed_set: A new, narrowed training set that contains the curves within the sigma_tolerance range.
    narrowed_parameters: The collection of parameters that correspond to the narrowed training set curves
    rms_array: An array of all the rms values of each curve. Important for some other functions.
    training_set: Just returns the same input parameter above. Important for some other functions.
    training_set_parameters: Just returns the same input parameter above. Important for some other functions."""

    # First step is to create the bootstrapped sigma from the data_fit_residual.
    # In more plain words: We take the signal we should get if the data was a perfect fit and calculate the rms. Then do this many times.
    
    rms_array= np.zeros((len(training_set)))  # creates dummy array for our rms values for each curve
    sigma_array = np.zeros_like(rms_array)    # create a dummy array for the distance in sigma that each curve is from the noise
    for i in tqdm(range(len(training_set))):      # loops through all the curves, subtracts the data from them, and then calculates the rms for each.
        differencing_element = data - training_set[i]
        rms_element = np.sqrt(np.mean(differencing_element**2))
        rms_array[i] = rms_element    # creates an array of the rms values for each training set curve
        sigma_array[i] = np.abs(rms_element-rms_mean)/one_sigma_rms   # creates an array of the distance from the noise is sigma of each training set curve.
    narrowed_set=training_set[np.where(sigma_array<sigma_tolerance)]
    narrowed_set_parameters=training_set_parameters[np.where(sigma_array<sigma_tolerance)]

    return narrowed_set, narrowed_set_parameters, rms_array, training_set, training_set_parameters
