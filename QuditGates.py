import numpy as np
from Generalized_Pauli_Matrices import GeneralizedPauliMatrices
from helper_functions import *

class QuditGates:
    def __init__(self, dim):
        self.dim = dim
        self.GPM = GeneralizedPauliMatrices(self.dim)
        self.generalized_pauli_matrices = self.GPM.return_gates()

    def get_X_gate(self):
        """ Returns the generalized X gate for the given dimension. 
        """
        return self.GPM.get_generalized_X()
    
    def get_Z_gate(self):
        """ Returns the generalized Z gate for the given dimension. 
        """
        return self.GPM.get_generalized_Z()
    
    def get_I_gate(self):
        """ Returns the identity gate for the given dimension. 
        """
        return np.eye(self.dim, dtype=np.complex128)
    
    def get_Y_gate(self):
        """ Returns the Y gate for qubits (dimension 2). 
        """
        if self.dim != 2:
            raise ValueError("Y gate is only defined for qubits (dimension 2).")
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    
    def get_H_gate(self):
        """ Returns the generalized Hadamard gate (F gate) for the given dimension. 
        """

        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                H[i][j] = np.exp(2 * np.pi * 1j * i * j / self.dim) / np.sqrt(self.dim)
        return H

    def CSUM_gate(self):
        """ Returns the CSUM gate for the given dimension.
            CSUM = C(X_d)
        """
        SUM = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)

        for i in range(self.dim):
            for j in range(self.dim):
                input_index = i * self.dim + j
                output_index = i * self.dim + (j + i) % self.dim
                SUM[output_index][input_index] = 1
        return SUM
    
    def Q_gate(self):
        Q = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for k in range(self.dim):
            val = 0
            for j in range(self.dim):
                cj = np.exp(-1j*np.pi*j*(j+self.dim%2)/self.dim)
                val += 1/np.sqrt(self.dim) * cj * np.exp(2*np.pi*1j*k*j/self.dim)
            Q[k][k] = val
        return Q

    def SWAP_gate(self):
        """ Returns the SWAP gate for the given dimension.
        """
        SWAP = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)

        for i in range(self.dim):
            for j in range(self.dim):
                input_index = i * self.dim + j
                output_index = j * self.dim + i
                SWAP[output_index][input_index] = 1
        return SWAP

    
    def create_Uencryption(self, number_S_qudits):
        operators = [self.get_X_gate(), self.get_Z_gate()]
        num_qudits = number_S_qudits + 1
        unitary_matrix = np.eye(self.dim ** num_qudits, dtype=np.complex128)
        
        # we generate the P_X and P_Z operators
        all_operators = []
        for operator in operators:
            all_operator = operator
            for i in range(number_S_qudits):
                all_operator = np.kron(all_operator, operator)
            all_operators.append(all_operator)

        # We construct the U_encryption unitary by summing over the powers of the P_X and P_Z operators
        for operator in all_operators:
            temp_unitary = np.zeros((self.dim ** num_qudits, self.dim ** num_qudits), dtype=np.complex128)
            for k in range(self.dim):
                temp_unitary += 1/np.sqrt(self.dim) * np.exp(2*np.pi*1j/self.dim)**(-k*(k+self.dim%2)/2) * np.linalg.matrix_power(operator, k)
            unitary_matrix @= temp_unitary



        return unitary_matrix


    def create_Udecryption(self, number_S_qudits):
        operators = [self.get_X_gate(), self.get_Z_gate()]
        operators_labels = ['X', 'Z']

        N_qudits = number_S_qudits - 1 # number of N qudits used in the gate
        S1_qudits = 1
        N1_qudits = 1
        num_qudits = N_qudits + S1_qudits + N1_qudits

        matrix_dim = self.dim ** num_qudits # setting the dimension of the operator matrix
        zero_matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.complex128) # initializing the operator matrix to zero

        
        maximally_mixed_matrix = create_ketbra_generalizedBell(self.dim)


        operators_S1 = [op for op in operators]
        operators_labels_S1 = [label for label in operators_labels]
        operators_S1 = operators_S1[::-1]
        operators_labels_S1 = operators_labels_S1[::-1]
        

        for k in range(self.dim):
            for l in range(self.dim):
                ck = np.exp(-1j*np.pi*(k)*(k+self.dim%2)/self.dim)
                cl = np.exp(-1j*np.pi*(l)*(l+self.dim%2)/self.dim)
                ckl = (ck * cl)**(-1)

                #compute the operator O_{kl}
                op_S_left = np.eye(self.dim, dtype=np.complex128)
                for idx,operator in enumerate(operators):
                    if operators_labels[idx] == 'X':
                        op_S_left @= np.linalg.matrix_power(operator, k)
                    if operators_labels[idx] == 'Z':
                        op_S_left @= np.linalg.matrix_power(operator, l)
                op_S1_left = np.kron(op_S_left, np.eye(self.dim, dtype=np.complex128))
                op_S1_right = op_S1_left.conj().T

                #compute the operator applied to the chosen pair (S, N)
                op_S =  self.SWAP_gate() @ op_S1_left @ maximally_mixed_matrix @ op_S1_right 


                #compute the operator applied to all the remaining N qudits
                operators_N = []
                for i in range(N_qudits):
                    op_N = np.eye(self.dim, dtype=np.complex128)
                    for idx,operator in enumerate(operators):
                        if operators_labels[idx] == 'X':
                            op_N @= np.linalg.matrix_power(operator, k)
                        if operators_labels[idx] == 'Z':
                            op_N @= np.linalg.matrix_power(operator, -l)
                    operators_N.append(op_N)
                op_N = operators_N[0]
                for i in range(1, N_qudits):
                    op_N = np.kron(op_N, operators_N[i])
                
                zero_matrix += ckl * np.kron(op_S, op_N)

        return zero_matrix

    