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

def norm_state(initial_state):
    return initial_state / np.linalg.norm(initial_state)

def counts_to_arrays(counts, num_qubits):
    states = [format(i, '0{}b'.format(num_qubits)) for i in range(2**num_qubits)]
    num_counts = np.zeros(2**num_qubits, dtype=int)
    for state, count in counts.items():
        index = int(state, 2) 
        num_counts[index] = int(count)
    return np.array(states), num_counts

def binary_to_int(states):
    return np.array([int(state, 2) for state in states])

def relative_entropy(p, q):
    sum = 0
    for i in range(q.size):
        if q[i] == 0 and p[i] != 0 :
            return np.nan 
        if  p[i] == 0:
            continue
        else:   
            sum = sum + p[i] * np.log(p[i] / q[i])   
    return sum

def rsse(p , q):
    return np.sqrt(np.sum((p - q)**2))