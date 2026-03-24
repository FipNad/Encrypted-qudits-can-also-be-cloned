import numpy as np


class GeneralizedPauliMatrices:
    def __init__(self, dim: int):
        self.dimension = dim
        self.omega = np.exp(2*np.pi*1j/self.dimension)
        self.pauli_matrices = []

    def ket_to_vector(self, value:int):
        """ Converts a ket value to a vector representation. 
        For example, for dimension 3, the ket |1> would be represented as [0, 1, 0]^T.
        """
        vector = np.zeros((self.dimension,1))
        vector[value][0] = 1

        return vector
    
    def get_generalized_X(self):
        """
        Returns the generalized X gate for the given dimension.
        """
        X_gate = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        X_gate[0][self.dimension-1] = 1
        for idx in range(1,self.dimension):
            X_gate[idx][idx-1] = 1
        return X_gate
    
    def get_generalized_Z(self):
        """
        Returns the generalized Z gate for the given dimension.
        """
        Z_gate = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for idx in range(self.dimension):
            Z_gate[idx][idx] = self.omega**idx
        return Z_gate


    def generate_gates(self):
        """
        Generates all generalized Pauli matrices for the given dimension.
        """
        for k in range(self.dimension):
            for j in range(self.dimension):
                # sigma_ij = np.zeros((self.dimension,self.dimension))
                sigma_ij = np.linalg.matrix_power(self.get_generalized_X(), k) @ np.linalg.matrix_power(self.get_generalized_Z(), j)   
                self.pauli_matrices.append(sigma_ij)

    def return_gates(self):
        self.generate_gates()
        return self.pauli_matrices




