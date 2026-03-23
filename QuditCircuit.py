import numpy as np

class QuditCircuit:
    def __init__(self, n, dim=3):
        """
        Initialize an n-qudit system in |0...0> state
        """
        self.n = n
        self.dim = dim  # qudit dimension
        self.state = np.zeros(self.dim**n, dtype=complex)
        self.state[0] = 1  # |0...0>

    def initialize_state(self, state_vector):
        """
        Initialize the qudit system to a given state vector
        """
        if len(state_vector) != self.dim**self.n:
            raise ValueError("State vector length does not match number of qudits")
        norm = np.linalg.norm(state_vector)
        if not np.isclose(norm, 1):
            # pass
            raise ValueError("State vector must be normalized")
        self.state = np.array(state_vector, dtype=complex)

    def _index_to_axes(self, targets):
        """
        Convert target indices to a permutation of axes for tensor reshaping
        """
        targets = list(targets)
        axes = list(range(self.n))
        for t in targets:
            axes.remove(t)
        new_axes = axes + targets
        return new_axes

    def apply_gate(self, U, targets):
        """
        Apply an arbitrary m-qudit unitary gate U to the specified targets
        """
        m = len(targets)
        if U.shape != (self.dim**m, self.dim**m):
            raise ValueError("Gate shape does not match number of target qudits")
        
        # Reshape to tensor
        state_t = self.state.reshape([self.dim]*self.n)
        # Permute axes to move targets to the end
        new_axes = self._index_to_axes(targets)
        state_t = state_t.transpose(new_axes)
        # Reshape to 2D
        state_t = state_t.reshape(self.dim**(self.n-m), self.dim**m)
        # Apply gate
        state_t = state_t @ U.T  # Transpose due to row-major ordering
        # Reshape back
        state_t = state_t.reshape([self.dim]*self.n)
        # Invert permutation
        inv_axes = np.argsort(new_axes)
        self.state = state_t.transpose(inv_axes).reshape(self.dim**self.n)

    def apply_controlled_gate(self, U, control, target, control_level):
        """
        Apply a controlled gate U that operates on the targer qudits when the control qudits from the control list are in the specified control_level
        """

        m = len(target)
        if U.shape != (self.dim**m, self.dim**m):
            raise ValueError("Gate shape does not match number of target qudits")
        
        # Reshape to tensor
        state_t = self.state.reshape([self.dim]*self.n)
        # Permute axes to move controls and targets to the end
        new_axes = self._index_to_axes(control + target)
        state_t = state_t.transpose(new_axes)
        # Reshape to 2D
        state_t = state_t.reshape(self.dim**(self.n - len(control) - m), self.dim**len(control), self.dim**m)
        # Apply controlled gate
        for i in range(self.dim**len(control)):
            control_digits = np.base_repr(i, base=self.dim).zfill(len(control))
            if all(int(control_digits[j]) == control_level[j] for j in range(len(control))):
                state_t[:, i, :] = state_t[:, i, :] @ U.T  # Transpose due to row-major ordering
        # Reshape back
        state_t = state_t.reshape([self.dim]*self.n)
        # Invert permutation
        inv_axes = np.argsort(new_axes)
        self.state = state_t.transpose(inv_axes).reshape(self.dim**self.n)
        

    def measure(self, targets=None):
        """
        Perform a computational-basis measurement on specified targets
        If targets=None, measure all qudits
        Returns classical outcome(s) as a tuple
        """
        probs = np.abs(self.state)**2
        outcome = np.random.choice(self.dim**self.n, p=probs)
        outcome_digits = np.base_repr(outcome, base=self.dim).zfill(self.n)
        outcome_list = [int(d) for d in outcome_digits]
        if targets is None:
            return tuple(outcome_list)
        else:
            return tuple(outcome_list[t] for t in targets)

    def display_state(self, threshold=1e-6):
        """
        Print the full state vector with amplitudes above threshold
        """
        for idx, amp in enumerate(self.state):
            if np.abs(amp) > threshold:
                # Convert index to qudit digits
                digits = np.base_repr(idx, base=self.dim).zfill(self.n)
                print(f"|{digits}> : {amp}")
                # print(f"|{digits}>")
                
    def display_state_no_amplitudes(self, threshold=1e-6):
        """
        Print the basis states with amplitudes above threshold
        """
        for idx, amp in enumerate(self.state):
            if np.abs(amp) > threshold:
                # Convert index to qudit digits
                digits = np.base_repr(idx, base=self.dim).zfill(self.n)
                print(f"|{digits}>")

    def display_state_abs(self, threshold=1e-6):
        """
        Print the basis states with amplitudes above threshold, along with their magnitudes and probabilities
        """
        for idx, amp in enumerate(self.state):
            if np.abs(amp) > threshold:
                # Convert index to qudit digits
                digits = np.base_repr(idx, base=self.dim).zfill(self.n)
                if amp.real <0 and amp.imag <0:
                    print(f"|{digits}> : {amp.real:.3f}{amp.imag:.4f}j => |{digits}|={np.abs(amp):.4f} => Prob={np.abs(amp)**2:.4f}")
                elif amp.real <0 and amp.imag >=0:
                    print(f"|{digits}> : {amp.real:.3f}+{amp.imag:.4f}j => |{digits}|={np.abs(amp):.4f} => Prob={np.abs(amp)**2:.4f}")
                elif amp.real >=0 and amp.imag <0:
                    print(f"|{digits}> : {amp.real:.4f}{amp.imag:.4f}j => |{digits}|={np.abs(amp):.4f} => Prob={np.abs(amp)**2:.4f}")
                elif amp.real >=0 and amp.imag >=0:
                    print(f"|{digits}> : {amp.real:.4f}+{amp.imag:.4f}j => |{digits}|={np.abs(amp):.4f} => Prob={np.abs(amp)**2:.4f}")
                # else:
                    # print(f"|{digits}> : {amp:.4f} => |{digits}|={np.abs(amp):.4f} => Prob={np.abs(amp)**2:.4f}")
                # print(f"|{digits}>")

    def number_of_nonzero_amplitudes(self, threshold=1e-6):
        """
        Return the number of basis states with amplitude above threshold
        """
        count = 0
        for amp in self.state:
            if np.abs(amp) > threshold:
                count += 1
        return count