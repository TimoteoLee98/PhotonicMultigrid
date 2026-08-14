"""Multilevel multigrid solver, used for the QAO, QHO and parallel-capacitors problems.

The lattice-QCD problem uses the two-level TwoGridQCD in two_grid.py instead.
"""

import os

import numpy as np
from pyamg.util.linalg import approximate_spectral_radius
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from phoamg.linear_solver.cg import cg, get_atol_rtol
from phoamg.low_precision.dot import dot_low_vector
from phoamg.low_precision.scale import scale_factor

from .relaxation import polynomial_dot


class MultiGrid:
    """A multigrid hierarchy that can run its smoother in reduced precision.

    Built either from a pyamg hierarchy (`update_ml`) or loaded from disk (`load`).
    Which precision each level's matrix-vector product uses is decided by
    `change_relax_method` and held in `self.dots`, one callable per level; swapping an
    entry of that list is how the photonic-hardware emulation is substituted.

    `solve` runs one V- or W-cycle, `aspreconditioner` wraps that as a LinearOperator
    for an outer Krylov solver, and `solve_mixed` is the iterative-refinement loop the
    parallel-capacitors results use.
    """

    def __init__(self, ml=None, low_precision=False, bit_precision=5, rtol=1e-11, noise_strength=None, bit_precision_adc=8) -> None:
        self.max_value = 2**(bit_precision-1) - 1
        self.bit_precision = bit_precision
        self.rtol = rtol
        self.noise_strength = noise_strength
        self.bit_precision_adc = bit_precision_adc

        if ml is not None:
            self.update_ml(ml)
        if low_precision:
            self.change_relax_method(low_precision, bit_precision)


    def update_ml(self, ml):
        self.ml = ml
        self.As = []
        self.Ps = []
        self.Rs = []

        self.coefficients = []
        self.dots = []
        self.residuals = []
        self.ops = []
        for i in range(len(ml.levels)):
            A = ml.levels[i].A.tocsr()
            self.ops.append(len(A.data)*2 - A.shape[0])
            self.As.append(A)
            self.coefficients.append(1.0/approximate_spectral_radius(ml.levels[i].A))
            if i == len(ml.levels)-1:
                continue
            self.Ps.append(ml.levels[i].P)
            self.Rs.append(ml.levels[i].R)
            self.dots.append(self.dot_high_precision)


    def polynomial(self,lvl,x,b, iterations):
        for _ in range(iterations):
            polynomial_dot(lvl, x, b, iterations=1,
                            coefficient=self.coefficients[lvl], dot_product=self.dots[lvl])


    def dot_high_precision(self, lvl, x):
        return self.As[lvl].dot(x)

    def dot_low_precision(self, lvl, x):
        return dot_low_vector(self.As_low[lvl], x, self.max_value, self.A_factors[lvl],
                              bit_precision_adc=self.bit_precision_adc, noise_strength=self.noise_strength)


    def change_relax_method(self, low_precision, bit_precision=None, noise_strength=None, lvl_start=None, bit_precision_adc=8):
        """Choose the precision of each level's matrix-vector product.

        With low_precision, levels below `lvl_start` get the quantized (optionally
        noisy) product and the rest stay in fp64.
        """
        if low_precision:
            if lvl_start is None:
                lvl_start = len(self.As)
            else:
                lvl_start = np.min([lvl_start, len(self.As)])
            if bit_precision is not None:
                self.bit_precision = bit_precision
                self.max_value = 2**(bit_precision-1) - 1


            self.noise_strength = noise_strength
            self.bit_precision_adc = bit_precision_adc


            self.dots = []
            self.As_low = []
            self.A_factors = []

            for i in range(lvl_start):
                A_factor = scale_factor(self.As[i], max_value=self.max_value)
                self.A_factors.append(A_factor)
                A_low = np.round(self.As[i]*A_factor)/A_factor
                self.dots.append(self.dot_low_precision)
                self.As_low.append(A_low)
            for i in range(lvl_start, len(self.As)):
                self.dots.append(self.dot_high_precision)
                self.As_low.append(self.As[i])


        else:
            self.dots = []
            for i in range(len(self.As)):
                self.dots.append(self.dot_high_precision)


    def solve(self, lvl, b, x=None,
              numPre=2, numPost=2, cycle='V',
             print_results=False, rtol=1e-2, low_precision_residual=False, coarse_solver='cg'):
        """One multigrid cycle from level `lvl`: smooth, correct on the coarse grid, smooth again.

        Recurses down to the coarsest level, which is solved by CG -- `coarse_solver='cg'`
        is the only value implemented. `cycle` is 'V' or 'W'. Anything else raises.

        `low_precision_residual` selects the product used for the residual that feeds the
        coarse grid: True takes the level's configured product (`self.dots[lvl]` -- the
        quantized or photonic one wherever low precision is enabled), False forces the
        exact fp64 operator. Mirrors `residual_low` in TwoGridQCD.
        """
        if x is None:
            x = np.zeros_like(b)

        self.polynomial(lvl, x, b, iterations=numPre)

        if low_precision_residual:
            result = self.dots[lvl](lvl, x)
        else:
            result = self.dot_high_precision(lvl, x)
        residual = b - result
        if print_results:
            print("residual before, level: {}".format(lvl))
            print(np.linalg.norm(residual))

        coarse_b = self.Rs[lvl]@(residual)

        if lvl == len(self.As) - 2:
            if coarse_solver != 'cg':
                raise ValueError(f"unknown coarse_solver {coarse_solver!r}; expected 'cg'")
            coarse_x,_ = cg(self.As[-1], coarse_b, rtol=rtol)
        else:
            if cycle == 'V':
                coarse_x = self.solve(lvl + 1, coarse_b, numPost=numPost, cycle='V',
                         print_results=print_results, low_precision_residual=low_precision_residual,
                           coarse_solver=coarse_solver)

            elif cycle == 'W':
                coarse_x = self.solve(lvl + 1, coarse_b,
                                  numPre=numPre, numPost=numPost, cycle=cycle,
                                 print_results=print_results, low_precision_residual=low_precision_residual,
                           coarse_solver=coarse_solver)
                coarse_x = self.solve(lvl + 1, coarse_b, x=coarse_x,
                              numPre=numPre, numPost=numPost, cycle=cycle,
                             print_results=print_results, low_precision_residual=low_precision_residual,
                           coarse_solver=coarse_solver)
            else:
                raise ValueError(f"unknown cycle {cycle!r}; expected 'V' or 'W'")
        x += (self.Ps[lvl]@(coarse_x)).reshape(x.shape)   # coarse grid correction


        if print_results:
            residual2 = b - self.As[lvl]@(x)
            print("residual after coarse-grid correction, level: {}".format(lvl))
            print(np.linalg.norm(residual2))
            print("reduction factor")
            print(np.linalg.norm(residual)/np.linalg.norm(residual2))

        self.polynomial(lvl, x, b, iterations=numPost)

        return x


    def aspreconditioner(self, numPre=2, numPost=2, cycle='V',
                        print_results=False, rtol=1e-11, low_precision_residual=False,
                          coarse_solver='cg'):
        shape = self.As[0].shape
        dtype = self.As[0].dtype

        def matvec(b):
            return self.solve(0, b, numPre=numPre, numPost=numPost,
                                cycle=cycle, print_results=print_results, rtol=rtol, low_precision_residual=low_precision_residual,
                                coarse_solver=coarse_solver)

        return LinearOperator(shape, matvec, dtype=dtype)


    def solve_mixed(self, A, b, M, rtol=1e-5, atol=0., maxiter=32, rtol_internal=1e-3,
                    max_outer=200):
        """Iterative refinement: repeated inner CG solves against the fp64 residual.

        Returns (x, [inner_iterations, outer_iterations], final_residual_norm).
        `max_outer` caps the refinement loop, well clear of what the shipped problems
        need (the 100-trial parallel-capacitors sweep tops out at 12 outer iterations),
        but a cap reached silently would return a residual above tolerance and report
        a cost that is only a floor.
        """

        bnrm2 = np.linalg.norm(b)

        atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

        counter = 0
        x,info = cg(A, b, rtol=rtol_internal, maxiter=maxiter, M=M)
        counter+=info

        r = b - A.dot(x)
        for i in range(max_outer):
            y,info = cg(A, r, rtol=rtol_internal, maxiter=maxiter, M=M)
            counter+=info
            x+=y
            r = b - A.dot(x)
            rS = np.linalg.norm(r)
            if rS <atol:
                break
        return x,[counter,i + 1],rS



    def save_grids(self, directory=''):
        number_of_levels = len(self.As)
        data = []
        data.append(number_of_levels)
        for i in range(number_of_levels):
            sparse.save_npz(os.path.join(directory, f"A_{i}.npz"), self.As[i])
            data.append(self.coefficients[i])
            if i == number_of_levels-1:
                continue
            sparse.save_npz(os.path.join(directory, f"P_{i}.npz"), self.Ps[i])
            sparse.save_npz(os.path.join(directory, f"R_{i}.npz"), self.Rs[i])

        np.save(os.path.join(directory, "data.npy"), data)

    def load(self, directory):

        self.As = []
        self.Ps = []
        self.Rs = []

        self.dots = []
        self.ops = []
        data = np.load(os.path.join(directory, "data.npy"))
        self.number_of_levels = int(data[0])
        self.coefficients = data[1:]

        for i in range(self.number_of_levels):
            A = sparse.load_npz(os.path.join(directory, f"A_{i}.npz"))
            self.ops.append(len(A.data)*2 - A.shape[0])
            self.As.append(A)
            if i == self.number_of_levels-1:
                continue
            self.Ps.append(sparse.load_npz(os.path.join(directory, f"P_{i}.npz")))
            self.Rs.append(sparse.load_npz(os.path.join(directory, f"R_{i}.npz")))
            self.dots.append(self.dot_high_precision)