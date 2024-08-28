from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA,FactorAnalysis
import glob
from sncosmo.salt2utils import BicubicInterpolator
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
import sncosmo
from scipy.interpolate import interp1d
from numpy import random
import pickle as pk
from astropy.table import Table
from sncosmo.constants import HC_ERG_AA
from sklearn import preprocessing
from scipy.linalg import block_diag
from pathlib import Path

SCALE_FACTOR = 1e-12

plt.rcParams['font.size'] = 20.

class ScalarFormatterClass1(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

p_grid=np.arange(-10,51,1)
w_grid_reg=np.arange(3310,8585,10) 

X, Y=np.meshgrid(p_grid,w_grid_reg)

## Retrieve Data and Define Train/Validation samples

snnames=[]
for file in glob.glob("../Applying_GPy/SEDs_reg_normalized/*.dat"):
    snnames.append(file.split("/")[3].split(".")[0])

len(snnames)

val_sn, train_sn=train_test_split(snnames, test_size=0.9, train_size=0.1, random_state=0)

train_names=[]
train_data=[]
train_data_err=[]

val_names=[]
val_data=[]
val_data_err=[]

for sn in snnames:
    if sn in val_sn:
        val_data.append(pd.read_csv(f"../Applying_GPy/SEDs_reg_normalized/{sn}.dat")["flux"].values)
        val_data_err.append(pd.read_csv(f"../Applying_GPy/SEDs_reg_normalized/{sn}.dat")["fluxerr"].values)
        val_names.append(sn)
    else:
        train_data.append(pd.read_csv(f"../Applying_GPy/SEDs_reg_normalized/{sn}.dat")["flux"].values)
        train_data_err.append(pd.read_csv(f"../Applying_GPy/SEDs_reg_normalized/{sn}.dat")["fluxerr"].values)
        train_names.append(sn)

#print(val_names)
#print(train_names)

noise_var_init=np.mean(np.array(train_data_err)**2.,axis=0)

## Apply FA

def Apply_FA(n_components):
    fa=FactorAnalysis(n_components,noise_variance_init=noise_var_init,iterated_power=10) 
    fa.fit(train_data)
    
    X_train_fa=fa.transform(train_data)
    X_val_fa=fa.transform(val_data)
    
    return fa.mean_, fa.components_, X_train_fa, X_val_fa, fa.noise_variance_

for ncomp in range(1,len(train_names)+1):
    print(ncomp)
    mean, components, train_trans, val_trans, noise_var = Apply_FA(ncomp)
    Path(f"./fa_components/fa{ncomp}").mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(mean).to_csv(f"./fa_components/fa{ncomp}/M0_fa{ncomp}.txt",index=None,header=None)
    pd.DataFrame(train_trans).to_csv(f"./fa_components/fa{ncomp}/X_train_fa{ncomp}.txt",index=None,header=None)
    pd.DataFrame(val_trans).to_csv(f"./fa_components/fa{ncomp}/X_val_fa{ncomp}.txt",index=None,header=None)
    pd.DataFrame(noise_var).to_csv(f"./fa_components/fa{ncomp}/noise_variance_fa{ncomp}.txt",index=None,header=None)

    for i in range(ncomp):
        df2=pd.DataFrame(components[i])
        df2.to_csv(f"./fa_components/fa{ncomp}/M{i+1}_fa{ncomp}.txt",index=None,header=None)






















