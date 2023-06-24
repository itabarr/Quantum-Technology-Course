from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

import numpy as np
from math import pi
import matplotlib.pyplot as plt

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    """Swapping the qubit registers for QFT operation."""
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def create_qft_circuit(n , initial_state = None):
    """Create a Quantum Fourier Transform circuit for n qubits."""
    # Initialize a quantum circuit with n qubits
    circuit = QuantumCircuit(n)

    if initial_state is not None:
        # normalize the initial state
        initial_state = initial_state / np.linalg.norm(initial_state)

        # Check that the length of initial_state is correct
        if (initial_state.shape[0] != 2**n):
            raise ValueError("The length of initial_state must equal 2**n_qubits.")

        # Initialize the qubits
        circuit.initialize(initial_state, range(n))

    # Create the QFT rotations
    qft_rotations(circuit, n)
    # Swap the registers for QFT
    swap_registers(circuit, n)
    return circuit

def simulate_circuit(circuit , shots , output_state_vector = None , plot = False):
    """Simulates a quantum circuit with the Qiskit Aer qasm_simulator and returns the result."""
    # Get the Aer qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    circ = circuit.copy()

    # Simulate the circuit with the qasm_simulator
    result = execute(circ, backend=simulator, shots=shots).result()

    # Get the counts (results) from the result object
    counts = result.get_counts(circ)

    qbits_num = circuit.num_qubits

    # create a list of all possible output states
    states = [format(i, '0%sb' % qbits_num) for i in range(2**qbits_num)]

    for state in states:
        if state not in counts:
            counts[state] = 0  
    
    all_counts = np.array([counts.get(state, 0) for state in states])
    normalized_counts = all_counts / shots

    # Create a bar plot

    if plot:
        plt.figure()
        bars = plt.bar(states , normalized_counts)
        plt.title('Quantum Circuit Simulation Results')
        plt.xlabel('States')
        plt.ylabel('Counts')
        
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), 
        #             ha='center', va='bottom')

    
    if output_state_vector is not None:
        probabilities = np.abs(output_state_vector)**2

        if plot:
            plt.scatter(states, probabilities, color='red')

    return normalized_counts

def state_vector_simulation(circuit , initial_state = None):

    sim = Aer.get_backend("aer_simulator")
    state_vector_circuit = circuit.copy()

    if initial_state is not None:
        # normalize the initial state
        initial_state = initial_state / np.linalg.norm(initial_state)

        # Check that the length of initial_state is correct
        if (initial_state.shape[0] != 2**state_vector_circuit.num_qubits):
            raise ValueError("The length of initial_state must equal 2**n_qubits.")

        plot_state_vector(initial_state, title="Initial state vector")
    
    else:
        initial_state = np.zeros(2**state_vector_circuit.num_qubits)
        initial_state[0] = 1

        plot_state_vector(initial_state, title="Initial state vector")

    
    state_vector_circuit.save_statevector()
    states_vector = sim.run(state_vector_circuit).result().get_statevector()

    plot_state_vector(states_vector, title="Output state vector")

    # Create a list of state labels in binary
    num_qubits = int(np.log2(np.asarray(states_vector).shape[0]))
    labels = [format(i, '0' + str(num_qubits) + 'b') for i in range(2**num_qubits)]
    states = np.asarray(labels)

    states_vector = np.asarray(states_vector)

    return states , states_vector

def plot_state_vector(state_vector, title=""):
    # Calculate the probabilities from the state vector
    probabilities = np.abs(state_vector)**2

    # Create a list of state labels in binary
    num_qubits = int(np.log2(np.asarray(state_vector).shape[0]))
    labels = [format(i, '0' + str(num_qubits) + 'b') for i in range(2**num_qubits)]
    
    # Create a bar plot
    # new figure
    plt.figure()
    plt.bar(labels, probabilities)
    plt.title(title)
    plt.xlabel('States')
    plt.ylabel('Probabilities')

if __name__ == "__main__":
    n = 5  # Change this to the number of qubits you want
    
    initial_state = np.zeros(2**n)
    initial_state[0] = 0.5
    initial_state[1] = 0.5

    #normalize the initial state
    initial_state = initial_state / np.linalg.norm(initial_state)

    qft_circuit = create_qft_circuit(n , initial_state )
    states , unmeasured_states_vector  = state_vector_simulation(qft_circuit , initial_state )

    #qft_circuit.initialize(initial_state , range(n)) 
    qft_circuit.measure_all()
    

    shots = 10000
    measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector , plot = True)
    unmeasured_probabilities_vector = np.abs(unmeasured_states_vector)**2


    # calculate the fidelity between the measured and unmeasured states
    prob_diff = (measured_probabilities_vector - unmeasured_probabilities_vector)**2
    total_error = np.sum(prob_diff)

    print("Total Error: ", total_error)
    
    # create expnonential range of shots
    num_of_samples = 10
    shots_range = np.linspace(1000, 100000, num_of_samples, dtype=int)
    errors = []

    for i, shots in enumerate(shots_range):
        if i == num_of_samples - 1:
            measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector , plot=True)
        
        else:
            measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector)
        prob_mse = np.sqrt(np.sum((measured_probabilities_vector - unmeasured_probabilities_vector)**2))
        
        random_prob_diff = np.ones(len(prob_diff)) * (1/len(prob_diff))
        random_mse = np.sqrt(np.sum((unmeasured_probabilities_vector - random_prob_diff)**2))

        total_error = prob_mse / random_mse
        errors.append(total_error)

        
        print(f"i: {i} Shots: {shots} Total Error: {total_error}")
    plt.figure()
    plt.plot(shots_range , errors)
    plt.show()