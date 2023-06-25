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

def simulate_circuit(circuit , shots , output_state_vector = None , plot = False , Title = None):
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

        #rotate the xticks
        plt.xticks(rotation=90)

        if Title is not None:
            plt.title(Title)
        
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), 
        #             ha='center', va='bottom')

    
    if output_state_vector is not None:
        probabilities = np.abs(output_state_vector)**2

        if plot:
            plt.scatter(states, probabilities, color='red')

    return normalized_counts

def state_vector_simulation(circuit , initial_state = None , plot = False , Title = None):

    sim = Aer.get_backend("aer_simulator")
    state_vector_circuit = circuit.copy()

    if initial_state is not None:
        # normalize the initial state
        initial_state = initial_state / np.linalg.norm(initial_state)

        # Check that the length of initial_state is correct
        if (initial_state.shape[0] != 2**state_vector_circuit.num_qubits):
            raise ValueError("The length of initial_state must equal 2**n_qubits.")

        if plot:
            plot_state_vector(initial_state, title="Initial state vector")
    
    else:
        initial_state = np.zeros(2**state_vector_circuit.num_qubits)
        initial_state[0] = 1

        plot_state_vector(initial_state, title="Initial state vector")

    
    state_vector_circuit.save_statevector()
    states_vector = sim.run(state_vector_circuit).result().get_statevector()

    if plot:
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

def relative_entropy(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == "__main__":
    n = 6  # Change this to the number of qubits you want
    
    initial_state = np.zeros(2**n)
    # t = np.linspace(0, 1, 2**n, endpoint=False)
    # initial_state = np.cos(2 * np.pi * t) + np.cos(4 * np.pi * t)
    # initial_state[2] = 3

    initial_state[0] = 1
    initial_state[1] = 1
    initial_state[2] = 1

    #normalize the initial state
    initial_state = initial_state / np.linalg.norm(initial_state)

    #calculate the fourier transform of the initial state
    fft_initial_state = np.fft.fft(initial_state)
    abs_fft_initial_state = np.abs(fft_initial_state)**2
    normalized_abs_fft_initial_state = abs_fft_initial_state / np.sum(abs_fft_initial_state)
    # plot the initial state and its fourier transform in the same figure, in 2 subplots, using stem plots

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].stem(initial_state)
    axs[0].set_title('Initial State')
    axs[1].stem(normalized_abs_fft_initial_state)
    axs[1].set_title('Fourier Transform of Initial State')
    axs[0].grid(True)
    axs[1].grid(True)


    qft_circuit = create_qft_circuit(n , initial_state )
    states , unmeasured_states_vector  = state_vector_simulation(qft_circuit , initial_state )

    #qft_circuit.initialize(initial_state , range(n)) 
    qft_circuit.measure_all()
    
    unmeasured_probabilities_vector = np.abs(unmeasured_states_vector)**2
    unified_probabilities = np.ones(2**n) / 2**n
    
    # create expnonential range of shots
    num_of_samples = 100
    shots_range = np.linspace(100, 100000, num_of_samples, dtype=int)
    errors = []

    for i, shots in enumerate(shots_range):
        if i == num_of_samples // 4:
            measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector , plot=True , Title = f"Compared final state to ideal state, shots = {shots}")

        if i == num_of_samples - 1:
            measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector , plot=True , Title = f"Compared final state to ideal state, shots = {shots}")
        
        else:
            measured_probabilities_vector = simulate_circuit(qft_circuit , shots, unmeasured_states_vector)
        prob_mse = np.sqrt(np.sum((measured_probabilities_vector - unmeasured_probabilities_vector)**2))
        
        measured_relative_entropy = relative_entropy(measured_probabilities_vector, unmeasured_probabilities_vector)
        uniform_relative_entropy = relative_entropy(unified_probabilities, unmeasured_probabilities_vector)

        total_error = np.log(measured_relative_entropy / uniform_relative_entropy)
        errors.append(total_error)

        
        print(f"i: {i} Shots: {shots} Total Error: {total_error}")
    plt.figure()
    plt.plot(shots_range , errors)
    plt.grid(alpha=0.3)

    plt.title("Relative Entropy Error vs. Number of Shots")
    plt.xlabel("Number of Shots")
    plt.ylabel("Log of Relative Entropy Error")
    plt.show()