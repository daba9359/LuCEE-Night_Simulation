
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