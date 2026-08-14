"""Assembly of the 2-D Wilson-Dirac operator as a sparse matrix.

CalculatorD builds D from a gauge-link configuration: a nearest-neighbour hopping
term, one gamma-matrix projector per direction, plus a diagonal mass term, with
antiperiodic boundary conditions in time. Most of the file is index bookkeeping --
the tables that say which lattice sites and spinor components each matrix entry
connects -- precomputed once so that rebuilding D for a new configuration or mass
is cheap.
"""

from scipy import sparse
import numpy as np


def roll_x_indices(Nt,Nx,size_matrix,roll):
    size0 = Nx*size_matrix

    indices_t = np.arange(Nt) *size0

    indices_x = np.arange(Nx)*size_matrix
    rolled_indices_x = np.roll(indices_x,roll)

    indices = np.arange(size_matrix)

    return ((indices_t[:,None] + rolled_indices_x[None,:])[:,:,None] + indices[None,:]).flatten()


def roll_t_indices(Nt,Nx,size_matrix,roll):
    size0 = Nx*size_matrix

    indices_t = np.arange(Nt) *size0
    rolled_indices_t = np.roll(indices_t,roll)

    indices_x = np.arange(Nx)*size_matrix


    indices = np.arange(size_matrix)

    roll_indices = ((rolled_indices_t[:,None] + indices_x[None,:])[:,:,None] + indices[None,:])

    return roll_indices.flatten(), roll_indices[int(roll*0.5 - 0.5)].flatten()

class IndicesCreator:
    def roll_x_indices(self,Nt,Nx,size_matrix,roll):
        size0 = Nx*size_matrix

        indices_t = np.arange(Nt) *size0

        indices_x = np.arange(Nx)*size_matrix
        rolled_indices_x = np.roll(indices_x,roll)

        indices = np.arange(size_matrix)

        return ((indices_t[:,None] + rolled_indices_x[None,:])[:,:,None] + indices[None,:]).flatten()


    def roll_t_indices(self,Nt,Nx,size_matrix,roll):
        size0 = Nx*size_matrix

        indices_t = np.arange(Nt) *size0
        rolled_indices_t = np.roll(indices_t,roll)

        indices_x = np.arange(Nx)*size_matrix


        indices = np.arange(size_matrix)

        roll_indices = ((rolled_indices_t[:,None] + indices_x[None,:])[:,:,None] + indices[None,:])

        return roll_indices.flatten(), roll_indices[int(roll*0.5 - 0.5)].flatten()

    def Transposed_indices(self,Nt,Nx,nDim):
        indices = np.arange(nDim)
        repeated_indices = indices*nDim
        size_matrix = nDim**2


        indices_lattice = np.arange(Nt*Nx)*size_matrix
        return ((indices_lattice[:,None] + indices[None,:])[:,:,None] + repeated_indices[None,:]).flatten()

    def Kron_indices(self,Nt,Nx, Nc,Ns):
        lattice_volume = Nt*Nx
        gauge_links_indices = np.reshape(np.arange(lattice_volume*Nc*Nc),(lattice_volume,Nc,Nc))
        repeated_gauge_links_indices =  np.repeat(np.repeat(gauge_links_indices,Ns,2),Ns,1)

        gamma_indices = np.reshape(np.arange(Ns*Ns), (Ns,Ns))
        repeated_gamma_indices = np.repeat(np.repeat(gamma_indices,Nc,0).reshape(1,Ns*Ns*Nc),lattice_volume*Nc,0)
        return repeated_gauge_links_indices.flatten() , repeated_gamma_indices.flatten()

    def indices_flatten(self,indices,size_matrix): # Add self

        indices_helper = np.arange(size_matrix)

        indices_unflatten = ((indices*size_matrix)[:,None] + indices_helper[None,:])

        return indices_unflatten.flatten()


class CalculatorD(IndicesCreator):
    def __init__(self, Nt,Nx,Nc,Ns, m):
        self.Nt = Nt
        self.Nx = Nx
        self.Nc = Nc
        self.Ns = Ns
        self.m = m
        self.lattice_volume = Nt*Nx

        self.Pauli = []
        self.Pauli.append(np.array([[0, 1], [1, 0]]))
        self.Pauli.append(np.array([[0,-1j], [1j, 0]]))
        self.Pauli.append(np.array([[1, 0], [0, -1]]))
        self.Id = np.array([[1, 0], [0, 1]])

        self.gamma5 = np.kron(np.identity(Nc), 1j*np.matmul(self.Pauli[2], self.Pauli[0]))
        self.Gamma5 = self.CalculateGamma5()

        self.down_t_gauge, _ = self.roll_t_indices(Nt,Nx,Nc*Nc,1)
        self.down_x_gauge    = self.roll_x_indices(Nt,Nx,Nc*Nc,1)

        self.transposed_indices = self.Transposed_indices(Nt,Nx,Nc)

        self.gauge_links_indices, self.gamma_indices = self.Kron_indices(Nt,Nx, Nc,Ns)

        self.row_index, self.col_index = self.CalculateD_indices()

        self.D_slash = None
        self.values_indices = self.start_values_indices()
    def start_values_indices(self):

        nDim = 2
        numCol = (2**nDim)

        indices1 = np.arange(self.lattice_volume * (self.Nc*self.Ns))*((self.Nc*self.Ns)) 
        indices2 = np.arange(numCol)*self.lattice_volume*(self.Nc*self.Ns)**2 
        indices3 = np.arange((self.Nc*self.Ns)) 
        indices = indices3[None,:] + indices2[:,None]
        indices = indices.flatten()[None,:] + indices1[:,None]

        return indices

    def CalculateDiag(self, m):
        row_index = np.arange(0,(1)*self.lattice_volume+1, 1)


        col_index = np.arange(0,(1)*self.lattice_volume, 1)
        value = np.identity(self.Nc*self.Ns)*(m+2)
        values = []
        for _ in range(self.lattice_volume):
            values.append(value)

        return sparse.bsr_array(((values, col_index, row_index)), shape=(self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))

    def CalculateGamma5(self):
        row_index = np.arange(0,(1)*self.lattice_volume+1, 1)


        col_index = np.arange(0,(1)*self.lattice_volume, 1)

        values = []
        for _ in range(self.lattice_volume):
            values.append(self.gamma5)

        return sparse.bsr_array(((values, col_index, row_index)), shape=(self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))

    def Roll(self,matrix, indices):
        return matrix[indices]

    def ConjugateTranspose(self,matrix,transposed_indices):
        return np.conjugate(matrix[transposed_indices])

    def Kron(self,matrix1,matrix2,indices1,indices2):
        return matrix1[indices1]*matrix2[indices2]

    def prepareD(self,gauge_links_t, gauge_links_x):

        offdiagonal_spinor_x_plus = -0.5*(self.Id - self.Pauli[0]).flatten()
        offdiagonal_spinor_x_minus = -0.5*(self.Id + self.Pauli[0]).flatten()
        offdiagonal_spinor_t_plus = -0.5*(self.Id - self.Pauli[2]).flatten()
        offdiagonal_spinor_t_minus = -0.5*(self.Id + self.Pauli[2]).flatten()

        gauge_links_shifted_t = self.Roll(gauge_links_t.copy(),self.down_t_gauge)
        gauge_links_shifted_x = self.Roll(gauge_links_x.copy(),self.down_x_gauge)

        offdiag_x_plus = self.Kron(gauge_links_x,offdiagonal_spinor_x_plus, self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                                                                  self.Nc*self.Ns,
                                                                                                                                  self.Nc*self.Ns) #Careful with U dimension t,x
        offdiag_x_minus = self.Kron(self.ConjugateTranspose(gauge_links_shifted_x,self.transposed_indices),offdiagonal_spinor_x_minus,
                                     self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                        self.Nc*self.Ns,
                                                                                        self.Nc*self.Ns)

        offdiag_t_plus = self.Kron(gauge_links_t, offdiagonal_spinor_t_plus, self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                                                                  self.Nc*self.Ns,
                                                                                                                                  self.Nc*self.Ns)
        offdiag_t_minus = self.Kron(self.ConjugateTranspose(gauge_links_shifted_t, self.transposed_indices),offdiagonal_spinor_t_minus, self.gauge_links_indices,
                                     self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                self.Nc*self.Ns,
                                                                self.Nc*self.Ns)

        return offdiag_x_plus, offdiag_x_minus, offdiag_t_plus, offdiag_t_minus

    def CalculateD(self, gauge_links_t, gauge_links_x):

        nDim = 2
        numCol = (2**nDim)

        offdiag_x_plus, offdiag_x_minus, offdiag_t_plus, offdiag_t_minus = self.prepareD(gauge_links_t, gauge_links_x)

        values = np.append(offdiag_t_minus, (offdiag_x_minus, offdiag_x_plus, offdiag_t_plus))
        values = values[self.values_indices.flatten()]


        antiperiodic_m_t = np.arange(self.Nx*self.Nc*self.Ns)*numCol*self.Nc*self.Ns
        indices_helper = np.arange(self.Nc*self.Ns)
        antiperiodic_m_t = (antiperiodic_m_t[:,np.newaxis] + indices_helper).flatten()

        indices_helper = np.arange(self.Nc*self.Ns)*(-1)
        antiperiodic_p_t = (np.arange(self.Nx*self.Nc*self.Ns)*-numCol*self.Nc*self.Ns)-1
        antiperiodic_p_t = (antiperiodic_p_t[:,np.newaxis] + indices_helper).flatten()

        values[antiperiodic_m_t] = -values[antiperiodic_m_t]
        values[antiperiodic_p_t] = -values[antiperiodic_p_t]

        values = values.flatten()

        D = sparse.csr_array(((values, self.col_index, self.row_index)), shape=(self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))
        D += self.CalculateDiag(self.m)
        return D

    def UpdateD(self, gauge_links_t, gauge_links_x):
        self.D_slash = self.CalculateD(gauge_links_t, gauge_links_x)
        self.D_slash_dagger = self.Gamma5.dot(self.D_slash.dot(self.Gamma5))


    def neighbour_site_columns(self):
        """Column indices of each site's 2^d nearest neighbours, one entry per site.

        Ordered (t-, x-, x+, t+). CalculateD_indices expands these to spinor level.
        """
        roll_x_m = roll_x_indices(self.Nt,self.Nx,1,1)
        roll_x_p = roll_x_indices(self.Nt,self.Nx,1,-1)
        roll_t_m, _ = roll_t_indices(self.Nt,self.Nx,1,1)
        roll_t_p, _ = roll_t_indices(self.Nt,self.Nx,1,-1)

        col_index = np.stack((roll_t_m, roll_x_m, roll_x_p, roll_t_p)).T.flatten()

        return col_index.astype(int)

    def CalculateD_indices(self):

        nDim = 2
        row_index = np.arange((self.lattice_volume)*(self.Nc*self.Ns) + 1)*((self.Nc*self.Ns)*2**nDim)


        col_old = self.neighbour_site_columns()
        col_old = self.indices_flatten(col_old, self.Nc*self.Ns).reshape(self.lattice_volume, self.Nc*self.Ns*(2**nDim))
        col_index = np.repeat(col_old, self.Nc*self.Ns, axis=0).flatten()


        return row_index.astype(int), col_index.astype(int)

