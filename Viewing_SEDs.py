import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.backends.backend_pdf import PdfPages

sns.set_style("whitegrid", {'axes.grid':'False','xtick.bottom': True,'ytick.left': True})

params = {"text.usetex" : True,'font.size': 20, 'font.family':'serif', 'font.serif':'Computer Modern'}
plt.rcParams.update(params)

p_grid=np.arange(-10,51,1)
w_grid1=np.arange(3350,8710,10)
w_grid2=np.arange(3400,8410,10)

fname=[]
for file in glob.glob("./final_SEDs/*.dat"):
    fname.append(file.split("/")[2].split(".")[0])
X, Y = np.meshgrid(p_grid,w_grid2)

for sn in fname:
    pp = PdfPages(f"./plots_3d_final_SEDs/{sn}.pdf")
    fig = plt.figure(figsize=(15,12))
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(p_grid,w_grid2)
    Z=pd.read_csv(f"./final_SEDs/{sn}.dat")["flux"].values.reshape(Y.shape)
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