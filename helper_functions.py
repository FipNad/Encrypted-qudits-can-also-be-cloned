import numpy as np
import string

def vector_to_ket(vector):
    """ Converts a state vector to a ket representation. 
    For example, for a 2-qubit state vector of size 4, the ket representation would be a 4x1 column vector.
    """
    return vector.reshape(-1, 1)

def print_matrix(matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            amp = matrix[row][col]
            # amp = np.round(amp, decimals=3)
            # print(amp.real, end=" ")
            if amp.real<0 and amp.imag <0:
                print(f"{amp.real:.1f}{amp.imag:.1f}j", end=" ")
            elif amp.real<0 and amp.imag>=0:
                print(f"{amp.real:.1f}+{amp.imag:.1f}j", end=" ")
            elif amp.real>=0 and amp.imag<0:
                print(f"{amp.real:.2f}{amp.imag:.1f}j", end=" ")
            elif amp.real>=0 and amp.imag>=0:
                print(f"{amp.real:.2f}+{amp.imag:.1f}j", end=" ")
            # print(f"{matrix[row][col]:.2f}", end=" ")
        print()

def create_maximally_entangled_state(dimension):
    """ Creates a maximally entangled state for the given dimension. 
    For example, for dimension 3, the state would be (|00> + |11> + |22>)/sqrt(3).
    """
    state = np.zeros((dimension**2, 1), dtype=np.complex128)
    for i in range(dimension):
        index = i * dimension + i
        state[index][0] = 1/np.sqrt(dimension)
    return state

def create_ketbra_generalizedBell(dimension):
    """ Creates the ketbra of the maximally entangled state for the given dimension. 
    For example, for dimension 3, the state would be (|00> + |11> + |22>)(<00| + <11| + <22|)/3.
    """
    bell_state = create_maximally_entangled_state(dimension)
    bell_ketbra = bell_state @ bell_state.conj().T
    return bell_ketbra

def compare_quantum_states(qc1, qc2, threshold=1e-6):
    """ Compares the states of two quantum circuits and prints the basis states with amplitudes above threshold. 
    """
    return np.allclose(qc1.state, qc2.state, atol=threshold)

def partial_trace_qudits(rho, trace_out, num_total_qudits, dim_qudit):
    """
    Computes the partial trace of a density matrix rho over the qudits specified in trace_out.
    """

    dim = dim_qudit ** num_total_qudits
    assert rho.shape == (dim, dim)

    # Reshape into tensor with 2N indices
    rho_tensor = rho.reshape([dim_qudit] * (2 * num_total_qudits))

    # Build einsum indices
    letters = string.ascii_letters
    bra = list(letters[:num_total_qudits])
    ket = list(letters[num_total_qudits:2*num_total_qudits])

    # For traced qudits, set bra and ket index equal
    for q in trace_out:
        ket[q] = bra[q]

    einsum_str = ''.join(bra + ket) + '->'

    # Remaining (non-traced) indices
    remaining = [i for i in range(num_total_qudits) if i not in trace_out]
    out_indices = [bra[i] for i in remaining] + [ket[i] for i in remaining]

    einsum_str += ''.join(out_indices)

    rho_reduced = np.einsum(einsum_str, rho_tensor)

    dim_reduced = dim_qudit ** len(remaining)
    return rho_reduced.reshape(dim_reduced, dim_reduced)