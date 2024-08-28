import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.backends.backend_pdf import PdfPages

#sns.set_style("whitegrid", {'axes.grid':'False','xtick.bottom': True,'ytick.left': True})

params = {"text.usetex" : True,'font.size': 20, 'font.family':'serif', 'font.serif':'Computer Modern'}
plt.rcParams.update(params)

p_grid=np.arange(-10,51,1)
w_grid_reg=np.arange(3310,8585,10) 
w_grid_plot=np.arange(3350,8585,50)

fname=[]

for file in glob.glob("./SEDs_reg/*.dat"):
    fname.append(file.split("/")[2].split(".")[0])

for sn in fname:
    pp = PdfPages(f"./plots_3d_SEDs_reg/{sn}.pdf")
    fig = plt.figure(figsize=(15,12))
    ax = plt.axes(projection='3d')
    data=pd.read_csv(f"./SEDs_reg/{sn}.dat")
    X=data["phase"].values.reshape(len(w_grid_reg),len(p_grid))
    Y=data["wave"].values.reshape(len(w_grid_reg),len(p_grid))
    Z=data["flux"].values.reshape(len(w_grid_reg),len(p_grid))
    ax.plot_surface(X,Y,Z,cmap='viridis', edgecolor='none')
    ax.set_xlabel(r"Phase (days)")
    ax.set_ylabel(r"Wavelength ($\AA$)")
    ax.set_zlabel(r"Flux (erg/s/cm$^2$/$\AA \times$ offset)")
    ax.yaxis.labelpad=20
    ax.xaxis.labelpad=15
    ax.zaxis.labelpad=15
    pp.savefig()
    pp.close()
    #plt.show()
    plt.close(fig)
