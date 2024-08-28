from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import GPy
import pickle
import bz2
import emcee
from multiprocessing import Pool
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams['font.size'] = 20.

p_grid=np.arange(-10,51,1)
w_grid_reg=np.arange(3310,8585,10) 
w_grid_plot=np.arange(3350,8585,50)

class ScalarFormatterClass1(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

# Reading filtered spectra data and Hsiao template

interpolated_reg_data={}

for file in glob.glob("../Interpolating_the_spectra/interpolated_spectras/*.pkl"):
    sn=file.split("/")[3].split(".")[0]
    f=bz2.BZ2File(file,'rb')
    interpolated_reg_data[sn]=pickle.load(f)
    f.close()

sne_list=[]

for sname in interpolated_reg_data.keys():
    if len(glob.glob(f"./plots_gp/{sname}.pdf"))==1:
        continue
    else:
        sne_list.append(sname)

Hsiao_temp=pd.read_csv("/home/cassia/SNANA/snroot/snsed/Hsiao07.dat",header=None,sep="\\s+")
Hsiao_temp.columns=["phase","wave","flux"]  # unit of erg/s/cm^2/A

Hsiao_temp["flux"]=Hsiao_temp["flux"]/np.max(Hsiao_temp["flux"])
#Hsiao_temp_reduced=Hsiao_temp[(Hsiao_temp["phase"]>=-10.) & (Hsiao_temp["phase"]<=50.) & (Hsiao_temp["wave"]>=3000) & (Hsiao_temp["wave"]<=9000)].reset_index(drop=True)
#Hsiao_interp = RegularGridInterpolator((Hsiao_temp_reduced["phase"].unique(), Hsiao_temp_reduced["wave"].unique()), Hsiao_temp_reduced["flux"].values.reshape(61,601))

def gp_w_prior(sn,wave,p,flux,fluxerr):
    kern1=GPy.kern.Matern52(1,variance=0.1, lengthscale=15.0)+GPy.kern.Fixed(1,np.diag(fluxerr**2))

    temp_hsiao_flux=Hsiao_temp[Hsiao_temp["wave"]==wave]["flux"].values
    temp_hsiao_phase=Hsiao_temp[Hsiao_temp["wave"]==wave]["phase"].values
    temp=interp1d(temp_hsiao_phase,temp_hsiao_flux)  
       
    def chi2(a):
        return np.sum((flux-(a*temp(p)))**2/fluxerr**2)

    res=minimize(chi2,[1],method="Nelder-Mead")
    result=[res.x[0]]

    def mean(x):
        return result[0]*temp(x)

    var_res=np.std(mean(p)-flux)**2

    mf = GPy.core.Mapping(1,1)
    mf.f = mean
    mf.update_gradients = lambda a,b: None

    prior1 = GPy.priors.Gaussian(mu=12.,sigma=0.2*12.)
    prior1.domain="positive"
    prior2 = GPy.priors.Gaussian(mu=var_res,sigma=0.1*var_res)
    prior2.domain="positive"
    prior3 = GPy.priors.Gaussian(mu=0.2*var_res,sigma=0.1*0.2*var_res)
    prior3.domain="positive"

    model=GPy.models.GPRegression(p.reshape(-1, 1),flux.reshape(-1, 1),kernel=kern1,mean_function=mf)
    model.kern.Mat52.lengthscale.set_prior(prior1,warning=False)
    model.kern.Mat52.variance.set_prior(prior2,warning=False)
    model.Gaussian_noise.variance.set_prior(prior3,warning=False)
    model.kern.fixed.variance.fix()
        
    model.optimize_restarts(num_restarts = 10, optimizer='bfgs', verbose=False)
    #print(model)
    
    kern2 = GPy.kern.Matern52(1,variance= model.param_array[0], lengthscale=model.param_array[1])
    y_mean, y_var=model.predict(p_grid.reshape(-1, 1),kern=kern2)
        
    return y_mean, y_var, mean(np.arange(-10,51,1)), model.param_array

for sn in tqdm(sne_list):
    pp = PdfPages(f"./plots_gp/{sn}.pdf")

    phase_reg_completed=[]
    wave_reg_completed=[]
    flux_reg_completed=[]
    eflux_reg_completed=[]

    chisqndof_sn=[]
    delta_l_sn=[]
    sigma_m_sn=[]
    sigma_n_sn=[]

    for wave in w_grid_reg:
        flux=[]
        fluxerr=[]
        p=[]
        for phase in interpolated_reg_data[sn].keys():
            if (phase>=-20) and (phase<=70):
                data_phase=interpolated_reg_data[sn][phase]
                if len(data_phase[data_phase["wave"]==wave]["flux"])>0:
                    flux.append(np.array(data_phase[data_phase["wave"]==wave]["flux"])[0])
                    fluxerr.append(np.array(data_phase[data_phase["wave"]==wave]["eflux"])[0]) 
                    p.append(phase)
                else:
                    continue
            else:
                continue
                
        norm=max(flux)
        p=np.array(p)[np.array(flux)>0]
        fluxerr=np.array(fluxerr)[np.array(flux)>0]/norm
        flux=np.array(flux)[np.array(flux)>0]/norm
        
        y_mean, y_var, mean_temp, modelpar = gp_w_prior(sn,wave,p,flux,fluxerr)

        phase_reg_completed.append(p_grid)
        wave_reg_completed.append(np.array([wave]*len(p_grid)))
        flux_reg_completed.append(y_mean*norm)
        eflux_reg_completed.append(np.sqrt(y_var.reshape(-1))*norm)

        delta_l_sn.append(modelpar[1])
        sigma_m_sn.append(np.sqrt(modelpar[0]))
        sigma_n_sn.append(np.sqrt(modelpar[3]))

        p_reduced=p[(p>=-10) & (p<=50)]
        flux_reduced=flux[(p>=-10) & (p<=50)]
        fluxerr_reduced=fluxerr[(p>=-10) & (p<=50)]

        y_mean_interp=interp1d(p_grid,y_mean.reshape(-1)*norm)(p_reduced)
        y_mean_err_interp=interp1d(p_grid,np.sqrt(y_var.reshape(-1))*norm)(p_reduced)

        chisqndof=np.sum((flux_reduced*norm-y_mean_interp)**2/((fluxerr_reduced*norm)**2+(y_mean_err_interp)**2))/(len(flux)-3)

        if np.isfinite(chisqndof):
            chisqndof_sn.append(chisqndof)
        else:
            print(f"$\\chi^2$ not finite  for {sn} at $\\lambda$={wave}$\\AA$")

        if (wave in np.arange(3310,3450,10)) or (wave in w_grid_plot):
            
            fig, ax=plt.subplots(2,1,figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
            ax[0].set_title(f"$\\sigma_M$={np.round(np.sqrt(modelpar[0]),4)}, $\\Delta l$={np.round(modelpar[1],2)},  $\\sigma_n$={np.round(np.sqrt(modelpar[3]),4)}")
            ax[0].plot(p_grid,y_mean*norm,lw=2,color="#1f77b4",label="GPR")
            ax[0].fill_between(p_grid,y_mean.reshape(-1)*norm+2.*np.sqrt(y_var.reshape(-1))*norm,y_mean.reshape(-1)*norm-2.*np.sqrt(y_var.reshape(-1))*norm,alpha=0.6,color="#1f77b4",label=r"95\% cl")
            ax[0].set_ylabel(r"Flux (erg/s/cm$^2$/$\AA \times$ offset)")
            ax[0].plot(p_grid,mean_temp*norm,lw=2,color="black",label=f"Hsiao07 ({wave}$\\AA$)")
            ax[0].errorbar(p_reduced,flux_reduced*norm,yerr=fluxerr_reduced*norm,color="black",ls="none",fmt=".",markersize=8,label="Data")
            formatter11=ScalarFormatterClass1()
            formatter11.set_scientific(True)
            formatter11.set_powerlimits((0,0))
            ax[0].yaxis.set_major_formatter(formatter11)
            ax[0].legend()

            ax[1].set_title(f"$\\chi^2$/ndof={np.round(chisqndof,2)}")
            ax[1].errorbar(p_reduced,(flux_reduced*norm-y_mean_interp)/np.sqrt((fluxerr_reduced*norm)**2+(y_mean_err_interp)**2),color="black",marker='.',markersize=8,lw=0)
            ax[1].plot(p_grid,np.array([0.]*len(p_grid)),color="#1f77b4")
            ax[1].set_xlabel(r"Phase (days)")
            ax[1].set_ylabel(r"residue")

            plt.tight_layout()

            #plt.show()
            pp.savefig()
            #plt.savefig(f"./test/{sn}.png",dpi=300,facecolor='white', transparent=False,bbox_inches='tight')
            plt.close(fig)
       
    pp.close()

    fig, ax=plt.subplots(1,4,figsize=(16,4))
    ax[0].hist(chisqndof_sn)
    ax[0].set_title(f"{np.round(np.mean(chisqndof_sn),2)} $\\pm${np.round(np.std(chisqndof_sn),2)}")
    ax[0].set_xlabel(r"$\\chi^2$/ndof")
    ax[0].set_ylabel("Counts")
    ax[1].hist(delta_l_sn)
    ax[1].set_title(f"{np.round(np.mean(delta_l_sn),2)} $\\pm${np.round(np.std(delta_l_sn),2)}")
    ax[1].set_xlabel(r"$\Delta l$")
    ax[2].hist(sigma_m_sn)
    ax[2].set_title(f"{np.round(np.mean(sigma_m_sn),2)} $\\pm${np.round(np.std(sigma_m_sn),2)}")
    ax[2].set_xlabel(r"$\sigma_M$")
    ax[3].hist(sigma_n_sn)
    ax[3].set_title(f"{np.round(np.mean(sigma_n_sn),2)} $\\pm${np.round(np.std(sigma_n_sn),2)}")
    ax[3].set_xlabel(r"$\sigma_n$")
    plt.tight_layout()
    plt.savefig(f"./plots_gp/gpr_par_plots/{sn}.png")
    #plt.show()   
    plt.close(fig)
      
    df=pd.DataFrame((np.array(phase_reg_completed).flatten(),np.array(wave_reg_completed).flatten(),np.array(flux_reg_completed).flatten(),np.array(eflux_reg_completed).flatten())).T
    df.columns=["phase","wave","flux","fluxerr"]
    df.to_csv(f"./SEDs_reg/{sn}_SED.dat",index=None)

    df2=pd.DataFrame([delta_l_sn,sigma_m_sn,sigma_n_sn,chisqndof_sn]).T
    df2.columns=["Delta_l_Matern","Sigma_Matern","Sigma_Noise","Chis2red"]
    df2.to_csv(f"./plots_gp/gpr_chi2ndof_files/{sn}_fit_info.dat",index=None)
    


'''
fig = plt.figure()
plt.hist(chisqndof_all)
plt.title(f"{np.round(np.mean(chisqndof_all),2)} $\\pm${np.round(np.std(chisqndof_all),2)}")
plt.xlabel(r"$\\chi^2$/ndof")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig(f"./chi2ndof_gpr.png")
#plt.show()   
plt.close(fig)
'''