# Encrypted-qudits-can-also-be-cloned

This repository is supplementary to the paper [Encrypted qudits can also be cloned](https://tex.stackexchange.com/questions/385675/how-do-i-add-begindefinition-enddefinition).

The proofs have been provided in the paper, but this repository contains the implementations that could be used to verify all the assumptions given in the paper. Thus, besides the mathematical derivations, one could use the implementations provided to use the algorithm.

We have implemented a simulator that contains the all the functionalities that are needed to implement a Qudit circuit.

# Summary of the files:

- **QuditCircuit.py** -> includes the logic for the Qudit simulator, for any dimension d and number of qudits
- **Generalized_Pauli_Matrices.py** -> includes the logic that generates all the Pauli (Weyl) operators for a given dimension d 
- **QuditGates.py** -> includes the implementations for the gates that have been used along the paper and for the most common qudit gates
- **helper_functions.py** -> includes functions that are used for printing, visualization, generating, etc.

# Notebooks
We have included in notebooks the neccesary code for the verifications of the statements given and proved during the paper. Each notebook could be customized for testing by modifying the dimension of the quantum system. Also, for some cases, the number of qudits in the protocol can be adjusted as suited by the user.

- **Test_circuit.ipynb** -> Implements all the logic of the protocol. It starts with two distinct circuits, one that will be used to implement the operators as explicit operators (definitions) and the other will be used for the gate implementation of both encryption and decryption operators. We initialize the circuit as presented in the protocol (we use a random state for the data qudit for showing the correctness of the implementation and mathematical derivation). After each of the encryption or decryption steps, we compare the quantum states for both the circuits to show the implementation is correct. The used can modify the dimension of the system, the number of qudits involved in the protocol and the system which will be used to store the data qudit after the decryption operation.

- **Test_unitarity.ipynb** -> We show that for any dimension (that can be chosen by user), the encryption and decryption operators are unitary matrices.

- **Test_encryption.ipynb** -> We test that the quantum state obtained after encryption does indeed respect the requirement of the paper, namely that each individual qudit is in a maximally mixed state. We show that for any dimension and number of qudits in the protocol, taking the partial trace over the chosen subsystem, will result in the maximally mixed matrix.

- **Compare_encryption_operators.ipynb** -> For the encryption operator proposed in our paper, we show that the matrix is identical to the one proposed by authors in the qubit paper.


- **test_lemmas/Lemma_Ax.ipynb** -> In each notebook, we provide numerical verification for each of the proven lemmas from the paper. The number of qudits and the dimension of the system are given as parameters.

-**Test_theorems/Theorem_IIIx.ipynb** -> In each notebook, we provide the numerical verification for each of the proven theorems from the paper. A user can choose to modify the dimension of the quantum system for each notebook.