import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sncosmo
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, RationalQuadratic, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
from tqdm import tqdm
import GPy
import pickle
import bz2
import emcee
from multiprocessing import Pool
import corner
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set_style("whitegrid", {'axes.grid':'False','xtick.bottom': True,'ytick.left': True})

params = {"text.usetex" : True,'font.size': 20, 'font.family':'serif', 'font.serif':'Computer Modern'}
plt.rcParams.update(params)

p_grid=np.arange(-10,51,1)
w_grid1=np.arange(3350,8710,10)
w_grid2=np.arange(3400,8410,10)

class ScalarFormatterClass1(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

# Reading filtered spectra data and Hsiao template

filtered_reg_data={}

for file in glob.glob("./filtered_spectras/*.pkl"):
    sn=file.split("/")[2].split(".")[0]
    f=bz2.BZ2File(file,'rb')
    filtered_reg_data[sn]=pickle.load(f)
    f.close()

temp_hsiao=pd.read_csv("/home/cassia/SNANA/snroot/snsed/Hsiao07.dat",header=None,sep="\s+")
temp_hsiao.columns=["phase","wave","flux"]  # unit of erg/s/cm^2/A

temp_hsiao["flux"]=temp_hsiao["flux"]/1e-8  # só para ficar numa escala mais parecida com nossos valores, eles estão na mesma unidade a parte de offsets mutiplicando

template=temp_hsiao

# Regularizing the SEDs using gaussian process on monochromatic light curves

def gaus_noise(model,wave,overfit=False):
    if wave>5000 or overfit:
        model.Gaussian_noise.variance.fix(0.0005)
        return None
    else:
        model.Gaussian_noise.variance.fix(0.)
        return None

def prior_ls(wave):
    if 5200<wave<5510:
        prior1 = GPy.priors.Gaussian(mu=30.,sigma=3.)
        return prior1
    else:
        prior1 = GPy.priors.Gaussian(mu=15.,sigma=3.)
        return prior1

def gp_w_prior(sn,wave,p,flux,fluxerr):
    kern1=GPy.kern.Matern52(1,variance=0.1, lengthscale=15.0)+GPy.kern.Fixed(1,np.diag(fluxerr**2))
        
    temp_hsiao_flux=template[template["wave"]==wave]["flux"].values/max(template[template["wave"]==wave]["flux"].values)
    temp_hsiao_phase=template[template["wave"]==wave]["phase"].values
    temp=interp1d(temp_hsiao_phase,temp_hsiao_flux)       # interpolando o template
       
    if sn in SN_wrong_max:
        def chi2_1(theta):
            a,b,c =theta
            return np.sum((flux-(a*temp(p+b)+c))**2/fluxerr**2)

        res=minimize(chi2_1,[1,0,0],method="Nelder-Mead")
        result=[res.x[0],res.x[1],res.x[2]]

    else:
        def chi2_2(theta):
            a,c =theta
            return np.sum((flux-(a*temp(p)+c))**2/fluxerr**2)

        res=minimize(chi2_2,[1,0],method="Nelder-Mead")
        result=[res.x[0],0.,res.x[1]]

    def mean(x):
        return result[0]*temp(x+result[1])+result[2]


    mf = GPy.core.Mapping(1,1)
    mf.f = mean
    mf.update_gradients = lambda a,b: None
            
    model=GPy.models.GPRegression(p.reshape(-1, 1),flux.reshape(-1, 1),kernel=kern1,mean_function=mf)
    
    gaus_noise(model,wave)
    prior1=prior_ls(wave)

    prior2 = GPy.priors.Gaussian(mu=0.1,sigma=0.1)#0.005)
    prior1.domain="positive"
    prior2.domain="positive"
    model.kern.Mat52.lengthscale.set_prior(prior1,warning=False)
    model.kern.Mat52.variance.set_prior(prior2,warning=False)
    model.kern.fixed.variance.fix()
    
    model.optimize()
    #print(model)
    if model.param_array[1]<9:
        gaus_noise(model,wave,overfit=True)
        model.optimize()
    
    kern3 = GPy.kern.Matern52(1,variance= model.param_array[0], lengthscale=model.param_array[1])
    
    y_mean, y_var=model.predict(p_grid.reshape(-1, 1),kern=kern3)

    return y_mean, y_var, temp, result

def gp_wo_prior(p,flux,fluxerr):

    kern1=GPy.kern.Matern52(1,variance=0.1, lengthscale=15.0)#+GPy.kern.Fixed(1,np.diag(fluxerr**2))
        
    model2=GPy.models.GPHeteroscedasticRegression(p.reshape(-1, 1),flux.reshape(-1, 1),kernel=kern1)
    #model2.Gaussian_noise.variance.fix(0.)
    #model2.kern.fixed.variance.fix()

    model2['.*het_Gauss.variance'] = fluxerr[:,None]**2
    model2.het_Gauss.variance.fix() 
    
    model2.optimize()

    #kern3 = GPy.kern.Matern52(1,variance= model2.param_array[0], lengthscale=model2.param_array[1])
    y_mean2, y_var2=model2._raw_predict(p_grid.reshape(-1, 1))#,kern=kern3)

    return y_mean2, y_var2

SN_wrong_max=[] #at low wavelengths, some with wrong max info

failed=[]

for sn in tqdm(filtered_reg_data.keys()):
    pp = PdfPages(f"./plots_gp/{sn}.pdf")
    ppc = PdfPages(f"./plots_gp/comparisons/{sn}.pdf")

    flux_reg_completed=[]
    eflux_reg_completed=[]

    for wave in w_grid2:
        flux=[]
        fluxerr=[]
        p=[]
        for phase in filtered_reg_data[sn].keys():
            data_phase=filtered_reg_data[sn][phase]
            if len(data_phase[data_phase["wave"]==wave]["flux"])>0:
                flux.append(np.array(data_phase[data_phase["wave"]==wave]["flux"])[0])
                fluxerr.append(np.array(data_phase[data_phase["wave"]==wave]["err_flux"])[0]) 
                p.append(phase)
            else:
                continue
                
        norm=max(flux)
        p=np.array(p)[np.array(flux)>0]
        fluxerr=np.array(fluxerr)[np.array(flux)>0]/norm
        flux=np.array(flux)[np.array(flux)>0]/norm
        
        try:
            y_mean, y_var, temp, result = gp_w_prior(sn,wave,p,flux,fluxerr)
            y_mean2, y_var2 = gp_wo_prior(p,flux,fluxerr)
        except:
            print(f"{sn} failed")
            failed.append(sn)
            break

        flux_reg_completed.append(y_mean*norm)
        eflux_reg_completed.append(np.sqrt(y_var.reshape(-1))*norm)

        if wave in np.arange(3450,8450,100):
            
            fig, ax=plt.subplots(figsize=(10,8))
            ax.plot(p_grid,y_mean*norm,lw=2,color="#ff7f0e",label="GPR w/ template prior",zorder=0)
            ax.fill_between(p_grid,y_mean.reshape(-1)*norm+2.*np.sqrt(y_var.reshape(-1))*norm,y_mean.reshape(-1)*norm-2.*np.sqrt(y_var.reshape(-1))*norm,alpha=0.6,color="#ff7f0e",label=r"95\% confidence",zorder=1)
            ax.plot(p_grid,y_mean2*norm,lw=2,color="#1f77b4",label="GPR w/o template prior",zorder=2)
            ax.fill_between(p_grid,y_mean2.reshape(-1)*norm+2.*np.sqrt(y_var2.reshape(-1))*norm,y_mean2.reshape(-1)*norm-2.*np.sqrt(y_var2.reshape(-1))*norm,alpha=0.6,color="#1f77b4",label=r"95\% confidence",zorder=3)
            ax.set_xlabel(r"Phase (days)",zorder=4)
            ax.set_ylabel(r"Flux (erg/s/cm$^2$/$\AA \times$ offset)")
            ax.plot(np.linspace(-10,50,100),(result[0]*temp(np.linspace(-10,50,100)+result[1])+result[2])*norm,lw=2,color="black",label=f"Hsiao07 template ({wave}$\AA$)",zorder=5)
            ax.errorbar(p,flux*norm,yerr=fluxerr*norm,color="black",ls="none",fmt=".",markersize=8,label="Filtered data",zorder=6)
            formatter11=ScalarFormatterClass1()
            formatter11.set_scientific(True)
            formatter11.set_powerlimits((0,0))
            ax.yaxis.set_major_formatter(formatter11)
            ax.legend()
            #plt.show()
            ppc.savefig()
            plt.close(fig)
            
            fig, ax=plt.subplots(figsize=(10,8))
            ax.plot(p_grid,y_mean*norm,lw=2,color="#1f77b4",label="GPR w/ template prior")
            ax.fill_between(p_grid,y_mean.reshape(-1)*norm+2.*np.sqrt(y_var.reshape(-1))*norm,y_mean.reshape(-1)*norm-2.*np.sqrt(y_var.reshape(-1))*norm,alpha=0.6,color="#1f77b4",label="95% confidence")
            ax.set_xlabel(r"Phase (days)")
            ax.set_ylabel(r"Flux (erg/s/cm$^2$/$\AA \times$ offset)")
            ax.plot(np.linspace(-10,50,100),(result[0]*temp(np.linspace(-10,50,100))+result[2])*norm,lw=2,color="black",label=f"Hsiao07 template ({wave}$\AA$)")
            ax.errorbar(p,flux*norm,yerr=fluxerr*norm,color="black",ls="none",fmt=".",markersize=8,label="Filtered data")
            formatter11=ScalarFormatterClass1()
            formatter11.set_scientific(True)
            formatter11.set_powerlimits((0,0))
            ax.yaxis.set_major_formatter(formatter11)
            ax.legend()
            #plt.show()
            pp.savefig()
            plt.close(fig)
            
    if sn not in failed:
        pp.close()
        ppc.close()
        
        df=pd.DataFrame((np.array(flux_reg_completed).flatten(),np.array(eflux_reg_completed).flatten())).T
        df.columns=["flux","fluxerr"]
        df.to_csv(f"./final_SEDs/{sn}_SED.dat",index=None)
