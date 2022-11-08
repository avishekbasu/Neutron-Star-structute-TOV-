import numpy as np
#from prep_data import *
from numba import jit, njit

pi = np.pi

@njit
def dp_dr(p,m,r, pressure_array, energy_array):
	#e=eos(p)
	e = np.interp(p, pressure_array, energy_array)
	#e = interpolate(p)
	dp_dr=(-1.0)*(e+p)*(m+(4.*pi*(r**3.)*p))/(r*(r-(2.*m)))
	return dp_dr
@njit
def dm_dr(p,r, pressure_array, energy_array):
	#e=eos(p)
	e = np.interp(p, pressure_array, energy_array)
	#e = interpolate(p)
	dm_dr=4.*pi*(r**2.)*e
	return dm_dr
#########################################################
@jit
def TOV_integrals(initial_pressure, pressure_array, energy_array):
	#r_initial=.00001
	dr=.01
	#initial_mass=4*pi*dr**3*eos(initial_pressure)
	initial_mass=4*pi*dr**3*np.interp(initial_pressure, pressure_array, energy_array)
	#initial_mass=4*pi*dr**3*interpolate(initial_pressure)
	
	initial_nu=0.0
	nu=[initial_nu]
	mass=[initial_mass]
	pres=[initial_pressure]
	radius=[dr]
	p_min=np.min(pressure_array)
	p_min = 1.0e-7
	ene = [np.interp(initial_pressure, pressure_array, energy_array)]
	#ene = interpolate(initial_pressure)


	while (pres[-1]>p_min):
		e = np.interp(pres[-1], pressure_array, energy_array)
		dr = 0.05*((1/mass[-1])*dm_dr(pres[-1],radius[-1], pressure_array, energy_array) - (1/pres[-1])*dp_dr(pres[-1],mass[-1],radius[-1], pressure_array, energy_array))**-1.0
		k10 = dr*dp_dr(pres[-1],mass[-1],radius[-1], pressure_array, energy_array)
		k11 = dr*dm_dr(pres[-1],radius[-1], pressure_array, energy_array)
		k20 = dr*dp_dr(pres[-1]+(k10/2.0),mass[-1]+(k11/2.0),radius[-1]+(dr/2.0), pressure_array, energy_array)
		k21 = dr*dm_dr(pres[-1]+(k10/2.0),radius[-1]+(dr/2.0), pressure_array, energy_array)
		k30 = dr*dp_dr(pres[-1]+(k20/2.0),mass[-1]+(k21/2.0),radius[-1]+(dr/2.0), pressure_array, energy_array)
		k31 = dr*dm_dr(pres[-1]+(k20/2.0),radius[-1]+(dr/2.0), pressure_array, energy_array)
		k40 = dr*dp_dr(pres[-1]+k30,mass[-1]+k31,radius[-1]+dr, pressure_array, energy_array)
		k41 = dr*dm_dr(pres[-1]+k30,radius[-1]+dr, pressure_array, energy_array)
		new_pres = pres[-1] + (1.0/6.0)*(k10+(2.0*k20)+(2.0*k30)+k40)
		new_mass = mass[-1]+ (1.0/6.0)*(k11+(2.0*k21)+(2.0*k31)+k41)
		new_radius = radius[-1] +dr
        	#new_nu=nu[-1]+(-1./(e+pres[-1]))*dr*dp_dr(pres[-1],mass[-1],radius[-1])
		new_radius=radius[-1]+dr
        	#nu.append(new_nu)
		pres.append(new_pres)
		mass.append(new_mass)
		radius.append(new_radius)
		ene.append(e)
	return radius[-1],mass[-1],pres
