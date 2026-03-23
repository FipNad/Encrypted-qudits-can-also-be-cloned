import numpy as np


class GeneralizedPauliMatrices:
    def __init__(self, dim: int):
        self.dimension = dim
        self.sigma_matrices = []
        self.omega = np.exp(2*np.pi*1j/self.dimension)

    def ket_to_vector(self, value:int):
        vector = np.zeros((self.dimension,1))
        vector[value][0] = 1

        return vector
    
    def get_generalized_X(self):
        X_gate = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        X_gate[0][self.dimension-1] = 1
        for idx in range(1,self.dimension):
            X_gate[idx][idx-1] = 1
        return X_gate
    
    def get_generalized_Z(self):
        Z_gate = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for idx in range(self.dimension):
            Z_gate[idx][idx] = self.omega**idx
        return Z_gate
    


    def create_matrix(self, k:int, j:int):
        sigma = np.zeros((self.dimension, self.dimension), dtype=np.complex128)


        for m in range(self.dimension):
            ket = self.ket_to_vector((m+k)%self.dimension)
            bra = self.ket_to_vector(m).T
            sigma += self.omega**(j*m) * (ket @ bra)
        return sigma
            



    def generate_gates(self):
        for k in range(self.dimension):
            for j in range(self.dimension):
                # sigma_ij = np.zeros((self.dimension,self.dimension))
                sigma_ij = self.create_matrix(k, j)


                self.sigma_matrices.append(sigma_ij)

    def return_gates(self):
        self.generate_gates()
        return self.sigma_matrices

    def print_matrices(self):
        self.generate_gates()
        for k in range(self.dimension):
            for j in range(self.dimension):

                print(f"sigma_{k}{j}:\n {self.sigma_matrices[k*self.dimension+j]}")


    def calculate_generalized_g_function(self, k, theta):
        value = 0
        for m in range(self.dimension):
            value += self.omega**(-k*m)*np.exp((self.omega**m)*(-1j)*theta)
        return value/self.dimension
    
    def calculate_generalized_hyperbolic_functions(self, theta):
        values = []
        for k in range(self.dimension):
            value = self.calculate_generalized_g_function(k, theta)
            values.append(value)

        return values
    
    def calculate_generalized_alternative_functions_odd(self):
        values = []
        for k in range(self.dimension):
            g_k = self.omega**(-k**2/2) * np.sqrt((1/self.dimension))
            values.append(g_k)
        
        return values
    
    def calculate_generalized_alternative_functions_even(self):
        values = []
        for k in range(self.dimension):
            g_k = np.exp(-1j*np.pi*k**2/self.dimension) * np.sqrt((1/self.dimension))
            values.append(g_k)
        return values

