import sys
import numpy as np
## declaring constants
dkm=(1.3234)*10**(-6.)


def load(fls = None):
    if fls == None:
    	print ('Data file not suppiled')
    	sys.exit()
    else:
        data=np.loadtxt(fls)
        
        energy=data[:,2]*dkm ## Converting from MeV/fm3 to Km-2
        pressure=data[:,3]*dkm
        normalized_density=data[:,0]
    return energy, pressure, normalized_density
    
