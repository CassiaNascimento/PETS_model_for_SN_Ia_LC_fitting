import sncosmo
import pandas as pd
from sncosmo.constants import HC_ERG_AA
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

_SCALE_FACTOR=1e-12

phase=np.arange(-10,51,1)
wave=np.arange(3400,8410,10)

M0_fa=np.loadtxt("sk_no_rot_M0.txt")

X, Y = np.meshgrid(phase,wave)
ab=sncosmo.get_magsystem("ab")

spec=sncosmo.Spectrum(Y.T[10],M0_fa.T[10]*_SCALE_FACTOR)
print(spec.bandmag("standard::b","ab"))

def integral(z):
	return -2.5*np.log10(np.sum(M0_fa.T[10]*_SCALE_FACTOR*wave*(1+z)/HC_ERG_AA*sncosmo.get_bandpass("standard::b")(wave*(1+z))*10))+2.5*np.log10(ab.zpbandflux("standard::b"))

print(integral(z=0.))

w=np.arange(3300,8510,10)
f=np.interp(w,wave,M0_fa.T[10],left=0.,right=0.)

# 7.8895