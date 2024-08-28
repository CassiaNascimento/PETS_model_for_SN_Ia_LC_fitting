from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.size'] = 20.

class ScalarFormatterClass1(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

chisqndof_all=[]
delta_matern_all=[]
sigma_matern_all=[]
sigma_noise_all=[]


for file in glob.glob("./plots_gp/gpr_chi2ndof_files/*.dat"):
    df= pd.read_csv(file)
    df.dropna(inplace=True)
    sn_name=file.split("/")[3].split(".")[0][:-9]
    for i in range(len(df)):
        chisqndof_all.append(df["Chis2red"][i])
        delta_matern_all.append(df["Delta_l_Matern"][i])
        sigma_matern_all.append(df["Sigma_Matern"][i])
        sigma_noise_all.append(df["Sigma_Noise"][i])
print(len(chisqndof_all))
fig = plt.figure()
plt.hist(chisqndof_all,density=True)
plt.title(f"{np.round(np.mean(chisqndof_all),1)} $\\pm${np.round(np.std(chisqndof_all),1)}")
plt.xlabel(r"$\\chi^2$/ndof")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(f"./chi2ndof_gpr.png",facecolor="white")
#plt.show()   
plt.close(fig)

fig, ax=plt.subplots(1,4,figsize=(16,4))
ax[0].hist(chisqndof_all,density=True)
ax[0].set_title(f"{np.round(np.mean(chisqndof_all),1)} $\\pm${np.round(np.std(chisqndof_all),1)}")
ax[0].set_xlabel(r"$\chi^2$/ndof")
ax[0].set_ylabel("Density")
ax[1].hist(delta_matern_all,density=True)
ax[1].set_title(f"{np.round(np.mean(delta_matern_all),1)} $\\pm${np.round(np.std(delta_matern_all),1)}")
ax[1].set_xlabel(r"$\Delta l$")
ax[2].hist(sigma_matern_all,density=True)
ax[2].set_title(f"{np.round(np.mean(sigma_matern_all),2)} $\\pm${np.round(np.std(sigma_matern_all),2)}")
ax[2].set_xlabel(r"$\sigma_M$")
ax[3].hist(sigma_noise_all,density=True)
ax[3].set_title(f"{np.round(np.mean(sigma_noise_all),2)} $\\pm${np.round(np.std(sigma_noise_all),2)}")
ax[3].set_xlabel(r"$\sigma_n$")
plt.tight_layout()
plt.savefig(f"./par_plots.png",facecolor="white")
#plt.show()   
plt.close(fig)