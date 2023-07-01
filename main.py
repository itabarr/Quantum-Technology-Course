from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel , pauli_error
from qiskit.circuit.library import QFT

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

import multiprocessing as mp

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

def qft_circuit(n , initial_state = None , add_bit_flip = False , bit_flip_prob = 0.05 , general_noise = False , general_noise_prob = 0.1):
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


    # Add bit flip
    if add_bit_flip:
        print(bit_flip_prob)
        bit_flip_prob_gate = pauli_error([('X', bit_flip_prob), ('I', 1 - bit_flip_prob)])
        circuit.append(bit_flip_prob_gate, [0])
        

    # Create the QFT rotations
    qft_rotations(circuit, n)
    # Swap the registers for QFT
    swap_registers(circuit, n)

    if general_noise:
        noise_model = NoiseModel()
        error_gate1 = pauli_error([('X', general_noise_prob), ('I', 1 - general_noise_prob)])
        noise_model.add_all_qubit_quantum_error(error_gate1, ['h'])
        return circuit , noise_model

    return circuit

def inverse_qft_circuit(n):
    qc = qft_circuit(n)
    return qc.inverse()

def simulate_circuit(circuit , shots , output_state_vector = None , plot = False , Title = None , noise_model = None):
    """Simulates a quantum circuit with the Qiskit Aer qasm_simulator and returns the result."""
    # Get the Aer qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    circ = circuit.copy()

    # Simulate the circuit with the qasm_simulator
    if noise_model == None:
        result = execute(circ, backend=simulator, shots=shots).result()
    
    else:
        result = execute(circ, backend=simulator, shots=shots , noise_model=noise_model).result()

    # Get the counts (results) from the result object
    counts = result.get_counts(circ)

    qbits_num = circuit.num_qubits

    # create a list of all possible output states
    states = [format(i, '0%sb' % qbits_num) for i in range(2**qbits_num)]

    for state in states:
        if state not in counts:
            counts[state] = 0  
    
    all_counts = np.array([counts[state] for state in states])
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

def draw_circuit(name , qbits_num = 3, scale = 0.5 , fold = 20):
    if name == 'qft':
        qc = qft_circuit(qbits_num)
    
    elif name == 'iqft':
        qc = inverse_qft_circuit(qbits_num)
    
    elif name == 'qft_iqft':
        qc = qft_circuit(qbits_num)
        qc.barrier()
        qc = qc.compose(inverse_qft_circuit(qbits_num))

    qc.measure_all()
    qc.draw(output='mpl', scale = scale , fold = fold)

def create_initial_state(name , qbits_num):
    if type(name) == np.ndarray:
        name = name / np.linalg.norm(name)

        return name
    
    if name == "zero":
        initial_state = np.zeros(2**qbits_num)
        initial_state[0] = 1
        return initial_state
    
    elif name == "cos":
        t = np.linspace(0, 1, 2**qbits_num, endpoint=False)
        initial_state = np.cos(2 * np.pi * t)

    elif name == "cos2":
        t = np.linspace(0, 1, 2**qbits_num, endpoint=False)
        initial_state = 0.1 * np.cos(2 * np.pi * t) + 0.2 * np.cos(8 * np.pi * t)
        initial_state [0] = 5
    elif name == "sin":
        t = np.linspace(0, 1, 2**qbits_num, endpoint=False)
        initial_state = np.sin(2 * np.pi * t)

    elif name == "square":
        t = np.linspace(0, 1, 2**qbits_num, endpoint=False)
        initial_state = np.zeros(2**qbits_num)
        # half of the states are 1 and the other half are 0
        initial_state[2**qbits_num // 2:] = 1
    elif name == "random":
        initial_state = np.random.rand(2**qbits_num)
        initial_state = initial_state / np.linalg.norm(initial_state)

    elif name == "gaussian":
        t = np.linspace(-qbits_num, qbits_num, 2 ** qbits_num)
        initial_state = np.exp(-(qbits_num**2)  * t ** 2 )

    elif name == "center_gaussian":
        t = np.linspace(-qbits_num, qbits_num, 2 ** qbits_num)
        shift_amount = int(2**qbits_num / 2)
        initial_state = np.exp(-4*qbits_num * t ** 2 ) * np.exp(shift_amount * np.pi * t * 1j)
        
    if initial_state is not None:
        # normalize the initial state
        initial_state = initial_state / np.linalg.norm(initial_state)
    
    return initial_state

def custom_fft(initial_state , qbits_num):
    
    try:
        if initial_state == None:
            initial_state = np.zeros(2**qbits_num)
            initial_state[0] = 1
    
    except:
        pass

    fft_initial_state = np.fft.fft(initial_state)
    abs_fft_initial_state = np.abs(fft_initial_state)**2
    normalized_abs_fft_initial_state = abs_fft_initial_state / np.sum(abs_fft_initial_state)

    return normalized_abs_fft_initial_state

def plot_fft(initial_state , fft_initial_state):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    initial_state = np.abs(initial_state) ** 2 
    initial_state = initial_state / np.sum(initial_state)

    axs[0].stem(initial_state)
    axs[0].set_title('Normalized Absolute Square of Initial State')
    axs[1].stem(fft_initial_state)
    axs[1].set_title('Normalized Absolute Square of DFT of Initial State')
    axs[0].grid(True)
    axs[1].grid(True)

    return fig , axs

def qft_accuracy_simulation(qbits_num = 3 , initial_state = "square"):
    initial_state_name = initial_state

    initial_state = create_initial_state(initial_state , qbits_num)
    qc = qft_circuit(n , initial_state)
    qc.measure_all()

    unmeasured_states_vector = np.fft.fft(initial_state)
    unmeasured_states_vector = unmeasured_states_vector / np.linalg.norm(unmeasured_states_vector)
    unmeasured_probabilities_vector = np.abs(unmeasured_states_vector)**2
    unified_probabilities = np.ones(2**n) / 2**n

    # create expnonential range of shots
    num_of_samples = 100
    shots_range = np.linspace(1000, 100000, num_of_samples, dtype=int)
    mse_error = []
    relative_error = []

    for i, shots in enumerate(shots_range):
        if i == num_of_samples // 4:
            measured_probabilities_vector = simulate_circuit(qc , shots, unmeasured_states_vector , plot=True , Title = f"Compared final state to ideal state, shots = {shots}")

        if i == num_of_samples - 1:
            measured_probabilities_vector = simulate_circuit(qc , shots, unmeasured_states_vector , plot=True , Title = f"Compared final state to ideal state, shots = {shots}")
        
        else:
            measured_probabilities_vector = simulate_circuit(qc , shots, unmeasured_states_vector)
        
        uniform_prob_mse = np.sqrt(np.sum((unified_probabilities - unmeasured_probabilities_vector)**2))
        prob_mse = np.sqrt(np.sum((measured_probabilities_vector - unmeasured_probabilities_vector)**2))
        

        #print(f"uniform_relative_entropy: {uniform_relative_entropy} measured_relative_entropy: {measured_relative_entropy}")
        _mse_error = (prob_mse)
        _relative_error = (prob_mse / uniform_prob_mse)
        mse_error.append(_mse_error)
        relative_error.append(_relative_error)
        
        print(f"shots: {shots} , mse_error: {_mse_error} , relative_error: {_relative_error}")
    
    



    plt.figure()

    #calculate moving average
    window_size = 5
    av_mse_error = np.convolve(mse_error, np.ones(window_size), 'valid') / window_size
    av_relative_error = np.convolve(relative_error, np.ones(window_size), 'valid') / window_size


    # plot 4 plots in one figure
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(shots_range, mse_error)
    # plot moving average
    axs[0].plot(shots_range[window_size - 1:], av_mse_error)
    axs[0].set_title(f'MSE for QFT of initial state {initial_state_name}')
    axs[0].grid(alpha = 0.3)
    axs[0].set_xlabel('Number of shots')
    axs[0].set_ylabel('MSE')

    axs[1].plot(shots_range, relative_error)
    # plot moving average
    axs[1].plot(shots_range[window_size - 1:], av_relative_error)
    axs[1].set_title(f'Relative MSE for QFT of initial state {initial_state_name}')
    axs[1].grid(alpha = 0.3)
    axs[1].set_xlabel('Number of shots')
    axs[1].set_ylabel('Relative MSE')

    

    # Add a title to the entire figure
    fig.suptitle(f'Divergence of QFT to ideal DFT , qbits_num = {qbits_num} , initial_state = {initial_state_name}')

    fig.subplots_adjust(hspace=0.5, wspace=0.4)

def qft_accuracy_simulation_multiple_initial_states(qbits_num=3, initial_states=["square"], animate=False):
    max_shots = 100000
    d_shots = 10000
    plt.figure()
    for initial_state in initial_states:
        initial_state_name = initial_state

        initial_state = create_initial_state(initial_state, qbits_num)
        qc = qft_circuit(qbits_num, initial_state)
        qc.measure_all()

        unmeasured_states_vector = np.fft.fft(initial_state)
        unmeasured_probabilities_vector = np.abs(unmeasured_states_vector) ** 2
        unmeasured_probabilities_vector = unmeasured_probabilities_vector / np.sum(
            unmeasured_probabilities_vector
        )

        unified_probabilities = np.ones(2 ** qbits_num) / 2 ** qbits_num

        shots_range = np.arange(d_shots, max_shots + d_shots, d_shots)

        results = np.zeros((len(shots_range), 2 ** qbits_num))
        histograms = np.zeros((len(shots_range), 2 ** qbits_num))
        mse = np.zeros((len(shots_range), 1))

        labels = [format(i, "0" + str(qbits_num) + "b") for i in range(2 ** qbits_num)]

        for i, shots in enumerate(shots_range):
            results[i] = simulate_circuit(qc, shots, unmeasured_states_vector)

        for i in range(len(shots_range)):
            if i == 0:
                continue
            histograms[i] = np.sum(results[0 : i + 1], axis=0) / (i + 1)
            error_prob = np.sqrt(np.sum((histograms[i] - unmeasured_probabilities_vector)**2)) 
            mse[i] = error_prob

        shots_arr = shots_range[2:]
        plt.plot(shots_arr, mse[2:], label=f"Initial State: {initial_state_name}")

    plt.grid(alpha=0.3)
    plt.title(f"QFT Accuracy Simulation - Qubits: {qbits_num}")
    plt.xlabel("Number of Shots")
    plt.ylabel("Squared Error-Sum")
    plt.legend()
    plt.xlim([shots_arr[0] , shots_arr[-1]])

def qft_accuracy_simulation_2(qbits_num = 3 , initial_state="square" , animate = False):
    initial_state_name = initial_state

    initial_state = create_initial_state(initial_state , qbits_num)
    qc = qft_circuit(n , initial_state)
    qc.measure_all()

    
    unmeasured_states_vector = np.fft.fft(initial_state)
    unmeasured_probabilities_vector = np.abs(unmeasured_states_vector)**2 
    unmeasured_probabilities_vector = unmeasured_probabilities_vector / np.sum(unmeasured_probabilities_vector)

    unified_probabilities = np.ones(2**n) / 2**n

    # create expnonential range of shots
    max_shots = 500000
    d_shots = 10000
    shots_range = int(max_shots / d_shots)
    
    results = np.zeros((shots_range , 2**n))

    # array of histograms
    histograms = np.zeros((shots_range , 2**n))
    mse = np.zeros((shots_range , 1))

    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    for i in range(shots_range):
        results[i] = simulate_circuit(qc , d_shots, unmeasured_states_vector)
    
    for i in range(shots_range):
        if i == 0:
            continue
        histograms[i] = np.sum(results[0:i] , axis = 0) / (i)
        prob_mse = np.sqrt(np.sum((histograms[i] - unmeasured_probabilities_vector)**2))
        mse[i] = prob_mse
        #print(histograms[i])
    
    plt.figure()
    shots_arr= np.arange(d_shots, max_shots + d_shots, d_shots)
    plt.plot(shots_arr[2:] , mse[2:])
    plt.grid(alpha = 0.3)
    plt.title(f'QFT Accuracy Simulation - Qubits: {qbits_num}, Initial State: {initial_state_name}')
    plt.xlabel('Number of Shots')
    plt.ylabel('MSE')

    if animate == True:
        fig = plt.figure()
        def update(curr):
            print(curr)
            if curr == shots_range -2:
                print("closing")
                plt.cla()
                plt.close()
                exit()
            plt.cla()
            try:
                plt.bar(labels , histograms[curr])
                plt.scatter(labels, unmeasured_probabilities_vector, color='red')
                plt.ylim([0,np.max(unmeasured_probabilities_vector)*1.1])
                plt.title(f"Histogram {curr*d_shots}")
                plt.xticks(rotation=90)
                plt.grid(alpha = 0.3)
            except:
                pass

        a = animation.FuncAnimation(fig, update, frames=shots_range-1 , interval = 10)
    
    plt.show(block=True)
    # plt.figure()

    # #calculate moving average
    # window_size = 5
    # av_mse_error = np.convolve(mse_error, np.ones(window_size), 'valid') / window_size
    # av_relative_error = np.convolve(relative_error, np.ones(window_size), 'valid') / window_size


    # # plot 4 plots in one figure
    # fig, axs = plt.subplots(2, 1)

    # axs[0].plot(shots_range, mse_error)
    # # plot moving average
    # axs[0].plot(shots_range[window_size - 1:], av_mse_error)
    # axs[0].set_title(f'MSE for QFT of initial state {initial_state_name}')
    # axs[0].grid(alpha = 0.3)
    # axs[0].set_xlabel('Number of shots')
    # axs[0].set_ylabel('MSE')

    # axs[1].plot(shots_range, relative_error)
    # # plot moving average
    # axs[1].plot(shots_range[window_size - 1:], av_relative_error)
    # axs[1].set_title(f'Relative MSE for QFT of initial state {initial_state_name}')
    # axs[1].grid(alpha = 0.3)
    # axs[1].set_xlabel('Number of shots')
    # axs[1].set_ylabel('Relative MSE')

def qft_accuracy_simulation_qbits():
    initial_states = ["square" , "cos2", "cos"]
    qbits_nums = [3 , 4 , 5 , 6 , 7 , 8]

    relative_mse_error = np.zeros((len(initial_states) , len(qbits_nums)))

    for i , _initial_state in enumerate(initial_states):
        for j , qbits_num in enumerate(qbits_nums):
            n = qbits_num

            initial_state_name = _initial_state
            initial_state = create_initial_state(_initial_state , qbits_num)

            qc = qft_circuit(qbits_num , initial_state)
            qc.measure_all()

            unmeasured_states_vector = np.fft.fft(initial_state)
            unmeasured_states_vector = unmeasured_states_vector / np.linalg.norm(unmeasured_states_vector)
            unmeasured_probabilities_vector = np.abs(unmeasured_states_vector)**2
            unified_probabilities = np.ones(2**n) / 2**n

            shots = 100000

            measured_probabilities_vector = simulate_circuit(qc , shots, unmeasured_states_vector , plot=False)
            uniform_prob_mse = np.sqrt(np.sum((unified_probabilities - unmeasured_probabilities_vector)**2))
            prob_mse = np.sqrt(np.sum((measured_probabilities_vector - unmeasured_probabilities_vector)**2))

            _log_relative_error = prob_mse
            relative_mse_error[i][j] = _log_relative_error

            print(f"qbits_num: {qbits_num} , initial_state: {initial_state_name} , log_relative_error: {_log_relative_error}")
        
    plt.figure()
    for i , initial_state in enumerate(initial_states):
        plt.plot(qbits_nums , relative_mse_error[i] , label = initial_state)

def simulate_noise(qbits_num=5, shots=10000 , initial_state_name = "square" , plot = False):
    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    d_p = 0.05
    p_noise_arr = np.arange(0, 1 + d_p, d_p)

    # Simulate circuit without noise
    initial_state = create_initial_state(initial_state_name, qbits_num)
    
    final_state = custom_fft(initial_state , qbits_num)

    p_initial_state = np.abs(initial_state) **2 
    p_initial_state = np.abs(initial_state) **2 / np.sum(p_initial_state)

    # Simulate circuit with noise
    results = []
    mse = []
    for p in p_noise_arr:
        qc_with_noise = qft_circuit(qbits_num, initial_state, add_bit_flip=True, bit_flip_prob=p)
        qc_with_noise.measure_all()
        counts_with_noise = simulate_circuit(qc_with_noise, shots)

        # Calculate the square root of the sum of errors squared
        error_squared = (counts_with_noise - final_state) ** 2
        mse.append(np.sqrt(np.sum(error_squared)))

        if plot:
            plt.bar(labels , counts_with_noise)
            plt.scatter(labels, p_initial_state, color='red')
            plt.show()

        

    # Plot the results
    plt.plot(p_noise_arr, mse , label=f"Initial State: {initial_state_name}")
    plt.grid(alpha = 0.3)

def simulate_general_noise(qbits_num=5, shots=10000 , initial_state_name = "square" , plot = False):
    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    d_p = 0.005
    p_noise_arr = np.arange(0, 0.1 + d_p, d_p)

    # Simulate circuit without noise
    initial_state = create_initial_state(initial_state_name, qbits_num)
    
    final_state = custom_fft(initial_state , qbits_num)

    p_initial_state = np.abs(initial_state) **2 
    p_initial_state = np.abs(initial_state) **2 / np.sum(p_initial_state)

    # Simulate circuit with noise
    results = []
    mse = []
    for p in p_noise_arr:
        qc_with_noise , noise_model= qft_circuit(qbits_num, initial_state, add_bit_flip=False, bit_flip_prob=p , general_noise=True , general_noise_prob=p)
        qc_with_noise.measure_all()
        counts_with_noise = simulate_circuit(qc_with_noise, shots , noise_model=noise_model)

        # Calculate the square root of the sum of errors squared
        error_squared = (counts_with_noise - final_state) ** 2
        mse.append(np.sqrt(np.sum(error_squared)))

        print(p , initial_state_name)
        if plot:
            plt.bar(labels , counts_with_noise)
            plt.scatter(labels, final_state, color='red')
            plt.show()

        

    # Plot the results
    plt.plot(p_noise_arr, mse , label=f"Initial State: {initial_state_name}")
    plt.grid(alpha = 0.3)


def simulate_general_noise_for_changing_qbits(initial_state_name="square", shots=10000, plot=False):
    qbits_arr = [3, 4, 5, 6, 7, 8]

    # Define the backend simulator
    simulator = Aer.get_backend('qasm_simulator')

    p_noise = 0.05

    mse = []
    for qbits_num in qbits_arr:
        # Simulate circuit without noise
        initial_state = create_initial_state(initial_state_name, qbits_num)
        final_state = custom_fft(initial_state, qbits_num)

        p_initial_state = np.abs(initial_state) ** 2
        p_initial_state = np.abs(initial_state) ** 2 / np.sum(p_initial_state)

        # Simulate circuit with noise
        qc_with_noise, noise_model = qft_circuit(qbits_num, initial_state, add_bit_flip=False,
                                                bit_flip_prob=p_noise, general_noise=True,
                                                general_noise_prob=p_noise)
        qc_with_noise.measure_all()
        counts_with_noise = simulate_circuit(qc_with_noise, shots, noise_model=noise_model)

        # Calculate the square root of the sum of errors squared
        error_squared = (counts_with_noise - final_state) ** 2
        mse.append(np.sqrt(np.sum(error_squared)))

        print(f"Qubits: {qbits_num}, Initial State: {initial_state_name}")
        if plot:
            labels = [format(i, '0' + str(qbits_num) + 'b') for i in range(2 ** qbits_num)]
            plt.bar(labels, counts_with_noise)
            plt.scatter(labels, final_state, color='red')
            plt.xticks(rotation=90)
            plt.show()

    # Plot the results
    plt.plot(qbits_arr, mse, label=f"Initial State: {initial_state_name}")
    plt.grid(alpha=0.3)
    plt.title("Root Sum Square Error Vs Number of Qubits")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Root Sum Square Error")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    n = qbits_num = 5  # Change this to the number of qubits you want

    #draw all the circuits
    # draw_circuit('qft' , qbits_num , scale = 0.7 , fold = 100)
    # # draw_circuit('iqft' , qbits_num)
    # # draw_circuit('qft_iqft' , qbits_num)

    initial_state = create_initial_state("gaussian" , qbits_num)
    fft_initial_state = custom_fft(initial_state , qbits_num)

    # plot_fft(initial_state , fft_initial_state)
    
    # qft_accuracy_simulation(qbits_num = qbits_num , initial_state = "square")
    # qft_accuracy_simulation(qbits_num = qbits_num , initial_state = "cos")
    #qft_accuracy_simulation_multiple_initial_states(qbits_num = 3 , initial_states = ["gaussian","cos", "square"])
    # qft_accuracy_simulation_multiple_initial_states(qbits_num = 5 , initial_states = ["gaussian","cos", "square"])
    # qft_accuracy_simulation_multiple_initial_states(qbits_num = 8 , initial_states = ["gaussian","cos", "square"])
    
    #simulation
    #qft_accuracy_simulation_2(qbits_num = 5 , initial_state="gaussian" , animate = True)
    #qft_accuracy_simulation_2(qbits_num = 5 , initial_state="square" , animate = True)
    
    plt.title(f"Root Sum Square Error Vs General Error Probability, Qubits: {qbits_num}")
    plt.xlabel("General Error Probability")
    plt.ylabel("RSSE")

    

    # simulate_noise(qbits_num=5 , initial_state_name = "zero")
    # simulate_noise(qbits_num=5 , initial_state_name = "gaussian")
    # simulate_noise(qbits_num=5 , initial_state_name = "cos")
    # simulate_noise(qbits_num=5 , initial_state_name = "square")
    # simulate_noise(qbits_num=5 , initial_state_name = "cos2")
    
    # simulate_general_noise(qbits_num=5 , initial_state_name = "zero" , plot = False)
    # simulate_general_noise(qbits_num=5 , initial_state_name = "gaussian" , plot = False)
    # simulate_general_noise(qbits_num=5 , initial_state_name = "cos" , plot = False)
    # simulate_general_noise(qbits_num=5 , initial_state_name = "square" , plot = False)
    # simulate_general_noise(qbits_num=5 , initial_state_name = "cos2" , plot = False)

    simulate_general_noise_for_changing_qbits(initial_state_name="square", shots=1000000, plot=False)
    plt.legend()
    
    plt.show()