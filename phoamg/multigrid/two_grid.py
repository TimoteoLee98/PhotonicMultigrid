"""Two-level multigrid for lattice QCD. See the MultiGrid class in multigrid.py for
the multilevel solver used by the other problems.
"""

import numpy as np
from pyamg.aggregation.tentative import fit_candidates
from pyamg.util.linalg import approximate_spectral_radius
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from phoamg.dirac.D_sparse import CalculatorD
from phoamg.linear_solver.cg import cg, get_atol_rtol
from phoamg.low_precision.dot import dot_low_vector
from phoamg.low_precision.scale import scale_factor

from .relaxation import polynomial_dot, richardson_prolongation_smoother
from .setup import run_setup


class TwoGridQCD:
    """Two-level multigrid for the lattice-QCD Dirac operator A = D D^dagger.

    The operator is held in three precisions at once: `A_high` (fp64, exact),
    `A_single` (fp32 round-trip) and `A_low` (quantized, optionally noisy, standing
    in for the photonic hardware). `change_relax_method` selects which one the
    smoother's matrix-vector product runs on.
    """

    def __init__(self, gauge_links, Nt, Nx, Nc, Ns, m=None, size_t=4, size_x=4, bit_precision=5, rtol=1e-3,
                  numCandidates=8, noise_strength=None, bit_precision_adc=None) -> None:
        self.calc = CalculatorD(Nt,Nx,Nc,Ns,0)

        self.max_value = 2**(bit_precision-1) - 1

        self.bit_precision = bit_precision
        self.bit_precision_adc= bit_precision_adc
        self.noise_strength = noise_strength
        self.rtol = rtol
        self.numCandidates = numCandidates
        self.B, self.P, self.R, self.Ac = [None, None, None, None]

        self.dot = self.dot_high_precision

        self.start_D_original(gauge_links)
        if m is not None:
            self.update_m(m)

        self.Nt = Nt
        self.Nx = Nx
        self.Nc = Nc
        self.Ns = Ns
        self.lattice_volume = Nt*Nx

        self.size_t = size_t
        self.size_x = size_x

        self.size_vector = Nc*Ns
        self.AggOp = self.calculate_aggregate(self.size_t, self.size_x, self.size_vector,
                                               self.Nt, self.Nx, self.lattice_volume)

    def indices_finder(self, t,x,s, Nx, size_vector=2):
        return s + x*size_vector + t*Nx*size_vector

    def calculate_aggregate(self, size_t, size_x, size_vector, Nt, Nx, lattice_volume):
        size = (size_t*size_x*size_vector)

        nodes_t = np.arange(int(Nt/size_t))*size_t
        nodes_x = np.arange(int(Nx/size_x))*size_x
        row = np.arange(int(lattice_volume*size_vector/size) + 1)*size
        col = []

        for t in nodes_t:
            for x in nodes_x:
                for i in range(size_x):
                    for j in range(size_t):
                        for s in range(size_vector):
                            col.append(self.indices_finder(t+j, x+i,s, Nx, size_vector))

        values = np.ones(len(col))

        return sparse.csr_matrix((values, col, row)).T.tocsr()

    def start_D_original(self, gauge_links):
        gauge_links_t = gauge_links[:,0,:,:].flatten()
        gauge_links_x = gauge_links[:,1,:,:].flatten()
        self.calc.UpdateD(gauge_links_t, gauge_links_x)

        self.D_slash_original = self.calc.D_slash
        self.D_slash_dagger_original = self.calc.D_slash_dagger

    def update_A_low(self):
        self.D_factor = scale_factor(self.D_slash_original, max_value=self.max_value)
        self.D_dagger_factor = scale_factor(self.D_slash_dagger_original, max_value=self.max_value)
        self.D_slash_low = np.round(self.D_slash_original*self.D_factor)/self.D_factor + self.diag
        self.D_slash_dagger_low = np.round(self.D_slash_dagger_original*self.D_dagger_factor)/self.D_dagger_factor + self.diag

        self.A_low = self.D_slash_low.tocsr().dot(self.D_slash_dagger_low.tocsr())

    def update_A_single(self):
        D_slash_low = self.D_slash_original.astype("complex64")+ self.diag
        D_slash_dagger_low = self.D_slash_dagger_original.astype("complex64") + self.diag

        self.A_single = D_slash_low.tocsr().dot(D_slash_dagger_low.tocsr()).astype("complex128")


    def update_m(self,m):
        sparse_identity = sparse.identity(self.D_slash_original.shape[0])
        self.diag = sparse_identity*m
        self.D_slash = self.D_slash_original + self.diag
        self.D_slash_dagger = self.D_slash_dagger_original + self.diag

        A_high = self.D_slash.tocsr().dot(self.D_slash_dagger.tocsr())
        self.A_high = A_high

        self.coefficient = 1.0/approximate_spectral_radius(self.A_high)

        self.A_low = self.update_A_low()

        self.A = self.A_high.copy()
        self.update_A_single()

    def relax(self, A, x):
        polynomial_dot(A, x, np.zeros_like(x), iterations=1,
                    coefficient=self.coefficient, dot_product=self.dot)

    def polynomial(self,A,x,b):
        polynomial_dot(A, x, b, iterations=1,
                        coefficient=self.coefficient, dot_product=self.dot)

    def dot_high_precision(self, A, x):
        # A is ignored: the product is applied as D (D^dagger x), one factor at a time.
        return self.D_slash.dot(self.D_slash_dagger.dot(x))

    def dot_low_precision(self, A, x):
        result = dot_low_vector(self.D_slash_dagger_low, x, self.max_value, self.D_dagger_factor,
                                bit_precision_adc=self.bit_precision_adc, noise_strength=self.noise_strength)
        result = dot_low_vector(self.D_slash_low, result, self.max_value, self.D_factor,
                                 bit_precision_adc=self.bit_precision_adc, noise_strength=self.noise_strength)
        return result

    def change_relax_method(self, low_precision, bit_precision=None, noise_strength=None, bit_precision_adc=None):

        self.A_dtype = self.A.dtype
        if low_precision:

            if bit_precision is not None:
                self.bit_precision = bit_precision
                self.max_value = 2**(bit_precision-1) - 1
                self.update_A_low()


            self.noise_strength = noise_strength

            self.bit_precision_adc = bit_precision_adc

            self.dot = self.dot_low_precision
            self.A = self.A_low.copy().astype(self.A_dtype)


        else:
            self.dot = self.dot_high_precision
            self.A = self.A_high.copy().astype(self.A_dtype)

    def run_setup_phase(self, setup_name=None, **kwargs):
        run_setup(self, setup_name=setup_name, numCandidates=self.numCandidates, **kwargs)


    def create_operators(self, B):
        self.B = B
        T_l, _ = fit_candidates(self.AggOp, B)


        self.P = richardson_prolongation_smoother(self.A_high, T_l)
        self.R = np.conjugate(self.P.T).asformat(self.P.format)
        self.Ac = self.R.dot(self.A_high.dot(self.P))


    def _solve(self, A, A_high, Ac, P, R, b, numPre=2, numPost=2, M=None, x=None,
               residual_low=False):

        if x is None:
            x = np.zeros_like(b)

        for _ in range(numPre):
            self.polynomial(A, x, b)


        if not residual_low:
            residual = b- A_high.dot(x)
        else:
            result = self.dot(A,x)
            residual = b- result
        coarse_b = R.dot(residual)

        solution, _ = cg(Ac, coarse_b, rtol=self.rtol, maxiter=10000, M=M)

        x2 = P.dot(solution)

        x += x2

        for _ in range(numPost):
            self.polynomial(A, x, b)

        return x

    def solve(self, b, numPre=2, numPost=2, M=None, x=None, single_precision=True,
              residual_low=False):
        if single_precision:
            if self.A.dtype.name.startswith('complex'):
                precision = "complex64"
                double_precision = "complex128"
            else:
                precision = "float32"
                double_precision = "float64"
            return self._solve(self.A, self.A_single, self.Ac,
                        self.P, self.R, b.astype(precision).astype(double_precision),
                        numPre=numPre, numPost=numPost, M=M, x=x,
                        residual_low=residual_low)
        else:
            return self._solve(self.A, self.A_high, self.Ac, self.P, self.R, b,
                        numPre=numPre, numPost=numPost, M=M, x=x,
                        residual_low=residual_low)


    def aspreconditioner(self, numPre=2, numPost=2, M=None, single_precision=True,
                         residual_low=False):
        shape = self.A.shape
        dtype = self.A.dtype

        def matvec(b):
            return self.solve(b, numPre=numPre, numPost=numPost, M=M, single_precision=single_precision,
                                residual_low=residual_low)

        return LinearOperator(shape, matvec, dtype=dtype)


    def mgcg(self, b, atol=0., rtol=9e-12, max_inner=8, max_outer=100,
                              numPre=4, numPost=0, single_precision=True,
                              residual_low=False):


        if self.A.dtype.name.startswith('complex'):
            double_precision = "complex128"
            if single_precision:
                precision = "complex64"
            else:
                precision = double_precision
        else:
            double_precision = "float64"
            if single_precision:
                precision = "float32"
            else:
                precision = double_precision

        M = self.aspreconditioner(numPre=numPre, numPost=numPost,
                                   single_precision=single_precision,
                                   residual_low=residual_low)

        bnrm2 = np.linalg.norm(b)
        atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)


        counter = 0
        x,info = cg(self.A_single, b.astype(precision).astype(double_precision), rtol=1e-1, maxiter=max_inner, M=M)
        x= x.astype(double_precision)
        counter+=info

        r = b - self.A_high.dot(x)

        for i in range(max_outer):
            y,info = cg(self.A_single, r.astype(precision).astype(double_precision), rtol=1e-1, maxiter=max_inner, M=M)
            counter+=info
            x+=y
            r = b - self.A_high.dot(x)
            rS = np.linalg.norm(r)
            if rS <atol:
                break
        return x,[counter,i + 1],rS
