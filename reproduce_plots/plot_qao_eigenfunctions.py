import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import blue, orange, gray, c2, c3, c4, create_things_for_plot

C1, C2, C3 = c2, c3, c4
fontsize = 10

Nx = 120
position, V = create_things_for_plot(Nx, m=1, x_min=-3, x_max=3, omega=1, l=0.25)
eigenvalues_qao  = np.load('measurements/qao_eigenvalues_n0to2.npy')
eigenvectors_qao = np.load('measurements/qao_eigenvectors_n0to2_grid120.npy')

width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

linewidth = 1.75
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2(np.linspace(0, 1, 8)))
signs_q = [-1, 1, -1]

plt.plot(position, signs_q[2]*eigenvectors_qao[:, 2] + eigenvalues_qao[2], linewidth=linewidth, color=C1)
plt.plot(position, signs_q[1]*eigenvectors_qao[:, 1] + eigenvalues_qao[1], linewidth=linewidth, color=C2)
plt.plot(position, signs_q[0]*eigenvectors_qao[:, 0] + eigenvalues_qao[0], linewidth=linewidth, color=C3)
plt.hlines(eigenvalues_qao, -3, 3, color='black', linestyle='--',
           label=r' $\mathcal{E}_n$', linewidth=linewidth*0.6, alpha=0.75)
plt.plot(position, V, linewidth=linewidth*0.8, color='black')

plt.ylim(-0.1, 4.75)
plt.xlim(-3, 3)
plt.xlabel(r'$y$',           fontsize=fontsize)
plt.ylabel(r'$\mathcal{E}$', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize*0.8, loc='lower left', handlelength=0.75)
plt.savefig('plots/qao_eigenfunction.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/qao_eigenfunction.svg")
