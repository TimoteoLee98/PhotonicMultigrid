import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
import scipy
from plot_utils import blue, orange, gray

fontsize = 10
u  = np.load('measurements/pc_field_solution_64x64.npy')
Nx = Ny = 64

def derivative_matrix_sparse_1d(N):
    A = np.zeros((N, N))
    for i in range(N):
        if i == 0:
            A[i, i] = -3; A[i, i+1] = 4; A[i, i+2] = -1
        elif i == N-1:
            A[i, i-2] = 1; A[i, i-1] = -4; A[i, i] = 3
        else:
            A[i, i-1] = -1; A[i, i+1] = 1
    return scipy.sparse.csr_matrix(A)

Dy = np.kron(derivative_matrix_sparse_1d(Nx).toarray(), np.identity(Ny))
Dx = np.kron(np.identity(Ny), derivative_matrix_sparse_1d(Nx).toarray())

x = np.arange(Nx)
y = np.arange(Ny)

width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

plt.contourf(x, y, np.sqrt((0.5*Dx.dot(u))**2 + (0.5*Dy.dot(u))**2).reshape(Ny, Nx),
             41, cmap='inferno')
plt.streamplot(x, y, -0.5*Dx.dot(u).reshape(Ny, Nx), -0.5*Dy.dot(u).reshape(Ny, Nx),
               color='white', density=0.9, linewidth=0.01, arrowsize=0.2)
plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
ax.set_xticks([0, 20, 40, 60])
ax.set_yticks([0, 20, 40, 60])
plt.savefig('plots/pc_field.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/pc_field.svg")
