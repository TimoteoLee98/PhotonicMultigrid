"""Build the multigrid hierarchies the reproduction scripts load.

Constructs the QAO, QHO and parallel-capacitors hierarchies from their problem
definitions and writes them to mg_load_files_GENERATED/, in the same format as the
shipped mg_load_files/. The finest-level operator is fully determined by the problem
and comes out identical; coarser levels match in shape and sparsity pattern. pyamg's
setup draws its candidate vectors and spectral-radius starting vector from numpy's
global RNG, so each build_* function seeds it (SEED = 0) before calling
smoothed_aggregation_solver, making every hierarchy reproducible on its own.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))

import numpy as np
import scipy.sparse
import pyamg

from phoamg.multigrid.multigrid import MultiGrid

OUT_DIR = os.path.join(_ROOT, 'mg_load_files_GENERATED')
SEED = 0


def _as_dir(path):
    return path if path.endswith(os.sep) else path + os.sep


def generate_hamiltonian_quartic_anharmonic_oscillator(
        Nx, m=1, x_min=-6, x_max=6, omega=1, l=0, minus_sign=False):
    A = pyamg.gallery.poisson((Nx,), format='csr', type='FD')
    L = x_max - x_min
    a = L / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    if minus_sign:
        V = -(0.5 * m**2 * omega**2 * x**2) + (l * x**4)
    else:
        V = (0.5 * m**2 * omega**2 * x**2) + (l * x**4)
    H = (1 / (2 * m * a**2)) * A.toarray() + np.diag(V)
    return scipy.sparse.csr_matrix(H), x, V


def build_qao(out_dir):
    Nx = 120
    l = 0.25
    omega = 1
    x_min, x_max = -3.5, 3.5

    H, x, V = generate_hamiltonian_quartic_anharmonic_oscillator(
        Nx, omega=omega, l=l, x_min=x_min, x_max=x_max
    )
    np.random.seed(SEED)
    ml = pyamg.smoothed_aggregation_solver(H, max_levels=3)
    mg = MultiGrid(ml)
    os.makedirs(out_dir, exist_ok=True)
    mg.save_grids(_as_dir(out_dir))
    return mg


def build_qho(out_dir):
    """Quantum harmonic oscillator: the same Hamiltonian without the quartic term."""
    Nx = 120
    l = 0
    omega = 1
    x_min, x_max = -8, 8

    H, x, V = generate_hamiltonian_quartic_anharmonic_oscillator(
        Nx, omega=omega, l=l, x_min=x_min, x_max=x_max
    )
    np.random.seed(SEED)
    ml = pyamg.smoothed_aggregation_solver(H, max_levels=4)
    mg = MultiGrid(ml)
    os.makedirs(out_dir, exist_ok=True)
    mg.save_grids(_as_dir(out_dir))
    return mg


def set_dirichlet_condition(x, y, Ny, A, b, value):
    index = x + y * Ny
    A[index, :] = 0
    A[index, index] = 1
    b[index] = value
    return A, b


def build_parallel_capacitors(out_dir):
    Ny = Nx = 64
    A = pyamg.gallery.poisson((Ny, Nx), format='csr', type='FD')
    A_new = A.toarray().copy()
    b = np.zeros(A.shape[0])

    start_x, end_x = Nx // 4, (Nx // 4) * 3
    start_y, step_y = Ny // 2, 1
    end_y = start_y + step_y
    steps_away_from_center_y, charge = 8, 2

    for xi in range(start_x, end_x):
        for yi in range(start_y, end_y):
            A_new, b = set_dirichlet_condition(
                xi, yi - step_y * steps_away_from_center_y, Ny, A_new, b, charge)
    for xi in range(start_x, end_x):
        for yi in range(start_y, end_y):
            A_new, b = set_dirichlet_condition(
                xi, yi + step_y * steps_away_from_center_y, Ny, A_new, b, -charge)

    A_new_sparse = scipy.sparse.csr_matrix(A_new)
    np.random.seed(SEED)
    ml = pyamg.smoothed_aggregation_solver(A_new_sparse, max_levels=2)
    mg = MultiGrid(ml)
    os.makedirs(out_dir, exist_ok=True)
    mg.save_grids(_as_dir(out_dir))
    return mg


if __name__ == '__main__':
    print('Building QAO multigrid hierarchy...')
    build_qao(os.path.join(OUT_DIR, 'qao'))

    print('Building QHO multigrid hierarchy...')
    build_qho(os.path.join(OUT_DIR, 'qho_plots'))

    print('Building parallel-capacitors multigrid hierarchy...')
    build_parallel_capacitors(os.path.join(OUT_DIR, 'parallel_capacitors'))

    print(f'Done -> {OUT_DIR}')
