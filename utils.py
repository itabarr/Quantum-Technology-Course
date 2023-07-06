from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
import itertools
import numpy as np



def replace_gate(circuit, old_gate, new_gate):
    # Convert the circuit to a gate-level representation
    new_qc = circuit.copy()
    #new_qc = new_qc.decompose()

    # Iterate over each gate in the decomposed circuit
    for instruction, qargs, cargs in new_qc.data:
        # If the instruction matches the old gate, replace it with the new gate
        print(instruction, qargs, cargs)

    return new_qc

def filter_bit_histogram(counts, bit_indices):
    # Generate all possible bit configurations for the bits of interest
    bit_configs = [''.join(config) for config in itertools.product('01', repeat=len(bit_indices))]

    # Initialize a dictionary to store the counts for each bit configuration
    filtered_counts = {config: 0 for config in bit_configs}

    for bitstring, count in counts.items():
        # Construct the bit configuration for the bits of interest
        bitstring = bitstring.replace(" ", "")
        config = ''.join(bitstring[-(index + 1)] for index in bit_indices)
        
        # Update the count for the specific bit configuration
        filtered_counts[config] += count

    return filtered_counts


def print_eigenvalues_and_eigenstates(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("The matrix is not square!")
        return

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Print the eigenvalues and eigenstates
    for i in range(eigenvalues.shape[0]):
        print("Eigenvalue:", eigenvalues[i])
        print("Eigenvector:", eigenvectors[:, i])
        print()

# Test the function with a sample matrix
matrix = np.array([[1, 0], [0, 2]])
print_eigenvalues_and_eigenstates(matrix)
