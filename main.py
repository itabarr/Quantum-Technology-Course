from qiskit import QuantumCircuit , Aer, execute , QuantumRegister, ClassicalRegister 
from qiskit_aer.noise import NoiseModel , pauli_error , depolarizing_error
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import * 

def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft_circuit(n , initial_state = None , add_bit_flip = False , bit_flip_prob = 0.05 , general_noise = False , general_noise_prob = 0.1):
    circuit = QuantumCircuit(n)

    # Initialize State
    if initial_state is not None:
        initial_state = norm_state(initial_state)
        circuit.initialize(initial_state, range(n))

    # Add bit flip
    if add_bit_flip:
        bit_flip_prob_gate = pauli_error([('X', bit_flip_prob), ('I', 1 - bit_flip_prob)])
        circuit.append(bit_flip_prob_gate, [0])
        
    # Create the QFT 
    qft_rotations(circuit, n)
    swap_registers(circuit, n)

    if general_noise:
        noise_model = NoiseModel()
        error_gate = depolarizing_error(general_noise_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_gate, ['h'])
        return circuit , noise_model

    return circuit , None

def simulate_circuit(circuit , shots , plot = False , Title = None , noise_model = None):
    simulator = Aer.get_backend('qasm_simulator')

    if noise_model is None:
        result = execute(circuit, backend=simulator, shots=shots).result()
    else:
        result = execute(circuit, backend=simulator, shots=shots , noise_model=noise_model).result()

    counts = result.get_counts(circuit)
    qbits_num = circuit.num_qubits

    states , counts = counts_to_arrays(counts, qbits_num)
    counts = counts / shots

    if plot:
        plt.figure()
        plt.bar(states , counts)
        plt.title('Quantum Circuit Simulation Results')
        plt.xlabel('States')
        plt.ylabel('Counts')
        plt.xticks(rotation=90)

        if Title is not None:
            plt.title(Title)
        
    return counts

def draw_circuit(qbits_num = 3, scale = 0.5 , fold = 20):
    qc, _= qft_circuit(qbits_num)
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

    elif name == "mixed":
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

    normalized_abs_fft_initial_state = np.where(normalized_abs_fft_initial_state < 10**-20, 0, normalized_abs_fft_initial_state)

    return normalized_abs_fft_initial_state

def plot_fft(initial_state , fft_initial_state):
    plt.rcParams.update({'font.size': 10})  # decrease font size
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    
    # Normalize initial state
    initial_state = np.abs(initial_state) ** 2 
    initial_state = initial_state / np.sum(initial_state)

    axs[0].stem(initial_state)
    axs[0].set_title(r'$|f(x)|^2$', fontsize=12)   # Set the title in math format
    axs[0].grid(True)
    axs[1].stem(fft_initial_state)
    axs[1].set_title(r'$|DFT\{f(x)\}|^2}$', fontsize=12)  # Set the title in math format
    axs[1].grid(True)

    plt.subplots_adjust(hspace = 0.3)
    fig.suptitle('FFT Analysis', fontsize=16)  # Add main title

    return fig , axs

def plot_all_fft(qbits_num, initial_states):
    for state_name in initial_states:
        initial_state = create_initial_state(state_name, qbits_num)
        fft_initial_state = custom_fft(initial_state, qbits_num)
        plot_fft(initial_state, fft_initial_state)

def qft_accuracy_simulation_animation(qbits_num = 3 , initial_state="square" , animate = False):
    n =qbits_num
    initial_state_name = initial_state

    initial_state = create_initial_state(initial_state , qbits_num)
    qc, _= qft_circuit(qbits_num , initial_state)
    qc.measure_all()

    unmeasured_probabilities_vector = custom_fft(initial_state , qbits_num)

    # create expnonential range of shots
    max_shots = 500000
    d_shots = 1000
    shots_range = int(max_shots / d_shots)
    
    results = np.zeros((shots_range , 2**n))

    # array of histograms
    histograms = np.zeros((shots_range , 2**n))
    mse = np.zeros((shots_range , 1))
    _relative_entropy = np.zeros((shots_range , 1))

    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    for i in range(shots_range):
        results[i] = simulate_circuit(qc , d_shots)
    
    for i in range(shots_range):
        if i == 0:
            continue

        histograms[i] = np.sum(results[0:i] , axis = 0) / (i)
        mse[i] = rsse(unmeasured_probabilities_vector , histograms[i])
        _relative_entropy[i] = relative_entropy(unmeasured_probabilities_vector, histograms[i])

    # plot rsse
    plt.figure()
    shots_arr= np.arange(d_shots, max_shots + d_shots, d_shots)
    plt.plot(shots_arr[2:] , mse[2:])
    plt.grid(alpha = 0.3)
    plt.title(f'RSSE - Qubits: {qbits_num}, Initial State: {initial_state_name}')
    plt.xlabel('Shots')
    plt.ylabel('RSSE')
    
    # plot relative entropy
    plt.figure()
    plt.plot(shots_arr[2:] , _relative_entropy[2:])
    plt.title(f'Relative Entropy - Qubits: {qbits_num}, Initial State: {initial_state_name}')
    plt.xlabel('Shots')
    plt.ylabel('Relative Entropy')
    plt.grid(alpha = 0.3)

    # plot log relative entropy
    plt.figure()
    plt.plot(shots_arr[2:] , np.log(_relative_entropy[2:]))
    plt.title(f'Log Relative Entropy - Qubits: {qbits_num}, Initial State: {initial_state_name}')
    plt.xlabel('Shots')
    plt.ylabel('Log Relative Entropy')
    plt.grid(alpha = 0.3)

    # Create Animation
    if animate == True:
        fig = plt.figure()
        def update(curr):
            print(f"iteration number {curr}.")

            if curr == 2 or curr == 11 or curr == 101:
                print("Enter to continue:")
                input()
            if curr == 102:
                print("Exiting")
                plt.cla()
                plt.close()
                exit()
            plt.cla()
            try:
                plt.bar(labels , histograms[curr] , label="Measured")
                plt.scatter(labels, unmeasured_probabilities_vector, color='red' , label="Ideal")
                plt.ylim([0,np.max(unmeasured_probabilities_vector)*1.1])
                plt.title(f"Histogram Comparison, Shots: {curr*d_shots} , Initial State: {initial_state_name}")
                plt.xticks(rotation=90)
                plt.grid(alpha = 0.3)
                plt.legend(loc='upper center')
                plt.xlabel("States")
                plt.ylabel("Probabilities")
            except:
                pass

        a = animation.FuncAnimation(fig, update, frames=shots_range-1 , interval = 10)
    
    plt.show(block=True)

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

def simulate_bit_flip_noise(qbits_num=5, shots=1000 , initial_state_name = "square" , plot = False):
    n = qbits_num
    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    d_p = 0.05
    p_noise_arr = np.arange(0, 1 + d_p, d_p)

    # Simulate circuit without noise
    initial_state = create_initial_state(initial_state_name, qbits_num)
    final_state = custom_fft(initial_state , qbits_num)

    p_initial_state = np.abs(initial_state) **2 
    p_initial_state = np.abs(initial_state) **2 / np.sum(p_initial_state)

    # Simulate circuit with noise
    _rsse = []
    _relative_entropy = []
    for p in p_noise_arr:
        print(f"Initial State  = {initial_state_name} , p = {p:.3f}")
        qc_with_noise, _= qft_circuit(qbits_num, initial_state, add_bit_flip=True, bit_flip_prob=p)
        qc_with_noise.measure_all()
        counts_with_noise = simulate_circuit(qc_with_noise, shots)

        _rsse.append(rsse(final_state , counts_with_noise))
        _relative_entropy.append(relative_entropy(final_state , counts_with_noise))

        if plot:
            plt.bar(labels , counts_with_noise)
            plt.scatter(labels, final_state, color='red')
            
            plt.ylim([0,np.max(final_state)*1.1])
            plt.title(f"Histogram Comparison , p_bit_flip = {p:.2f}, Initial State: {initial_state_name}")
            plt.xticks(rotation=90)
            plt.grid(alpha = 0.3)
            plt.xlabel("States")
            plt.ylabel("Probabilities")
            plt.show()    

    return _rsse , _relative_entropy , np.log(_relative_entropy)

def plot_all_statistics(qbits_num=5, shots=10000 , err = "bit_flip"):
    # Initialize array for bit flip probabilities
    d_p = 0.05
    p_noise_arr = np.arange(0, 1 + d_p, d_p)

    # List of state names
    state_names = ["zero", "gaussian", "cos", "square", "mixed"]

    # Initialize dictionaries to store each type of statistic for each state
    rsse_dict = {}
    relative_entropy_dict = {}
    log_relative_entropy_dict = {}

    # For each state name, compute all statistics
    for state_name in state_names:
        if err == "bit_flip":
            rsse, relative_entropy, log_relative_entropy = simulate_bit_flip_noise(qbits_num=qbits_num, shots=shots, initial_state_name=state_name, plot=False)
        
        if err == "general":
            rsse, relative_entropy, log_relative_entropy = simulate_general_noise(qbits_num=qbits_num, shots=shots, initial_state_name=state_name, plot=False)

        # Store statistics in corresponding dictionary
        rsse_dict[state_name] = rsse
        relative_entropy_dict[state_name] = relative_entropy
        log_relative_entropy_dict[state_name] = log_relative_entropy

    # Create a dictionary to store all statistics dictionaries
    all_stats_dict = {'RSSE': rsse_dict, 'Relative Entropy': relative_entropy_dict, 'Log Relative Entropy': log_relative_entropy_dict}

    # For each type of statistic, create a separate plot
    for stat_name, stat_dict in all_stats_dict.items():
        plt.figure(figsize=(10, 5))

        # For each state name, plot the statistic
        for state_name, statistic in stat_dict.items():
            plt.plot(p_noise_arr, statistic, label=f"{state_name}")

        plt.title(f'{stat_name} vs Bit Flip Probability')
        if err == "bit_flip":
            plt.xlabel('Bit Flip Probability')
        
        if err == "general":
            plt.xlabel('Depolarizing Error Parameter')

        plt.ylabel(stat_name)  # Y-label is the name of the statistic
        plt.grid(alpha = 0.3)
        plt.legend()

def simulate_general_noise(qbits_num=5, shots=10000 , initial_state_name = "square" , plot = False):
    n = qbits_num
    labels = [format(i, '0' + str(n) + 'b') for i in range(2**n)]
    d_p = 0.005
    p_noise_arr = np.arange(0, 0.1 + d_p, d_p)

    # Simulate circuit without noise
    initial_state = create_initial_state(initial_state_name, qbits_num)
    final_state = custom_fft(initial_state , qbits_num)

    p_initial_state = np.abs(initial_state) **2 
    p_initial_state = np.abs(initial_state) **2 / np.sum(p_initial_state)

    # Simulate circuit with noise
    _rsse = []
    _relative_entropy = []
    _relative_entropy = []
    for p in p_noise_arr:
        print(f"Initial State  = {initial_state_name} , p = {p:.3f}")
        qc_with_noise , noise_model = qft_circuit(qbits_num, initial_state,  general_noise = True , general_noise_prob = p)
        qc_with_noise.measure_all()
        counts_with_noise = simulate_circuit(qc_with_noise, shots , noise_model = noise_model)

        # Calculate the square root of the sum of errors squared
        _rsse.append(rsse(final_state , counts_with_noise))
        _relative_entropy.append(relative_entropy(final_state , counts_with_noise))

        if plot:
            plt.bar(labels , counts_with_noise)
            plt.scatter(labels, final_state, color='red')
            
            plt.ylim([0,np.max(final_state)*1.1])
            plt.title(f"Histogram Comparison , depolarizing error parameter = {p:.3f}, Initial State: {initial_state_name}")
            plt.xticks(rotation=90)
            plt.grid(alpha = 0.3)
            plt.xlabel("States")
            plt.ylabel("Probabilities")
            plt.show()
    
    return _rsse , _relative_entropy , np.log(_relative_entropy)

def simulate_general_noise_for_changing_qbits(initial_state_name="square", shots=10000, plot=False):
    qbits_arr = [3, 4, 5, 6, 7, 8]

    # Define the backend simulator
    simulator = Aer.get_backend('qasm_simulator')

    p_noise = 0.025

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
    plt.title(f"Root Sum Square Error Vs Number of Qubits, Depolarizing Error Parameter = {p_noise}")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Root Sum Square Error")
    plt.legend()
    
def simulate_mixed_noise_for_different_general_noise(qbits_num=5, shots=100000, initial_state_name="gaussian", plot=False):
    labels = [format(i, '0' + str(qbits_num) + 'b') for i in range(2**qbits_num)]
    d_p = 0.005
    p_noise_arr = np.arange(0, 0.1 + d_p, d_p)
    general_noise_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]  # or whatever levels you want

    # Simulate circuit without noise
    initial_state = create_initial_state(initial_state_name, qbits_num)
    final_state = custom_fft(initial_state, qbits_num)

    p_initial_state = np.abs(initial_state) **2
    p_initial_state = np.abs(initial_state) **2 / np.sum(p_initial_state)

    for gen_noise in general_noise_levels:
        # Simulate circuit with noise
        results = []
        mse = []

        for p in p_noise_arr:
            print(f"Initial State = {initial_state_name} , bit_flip = {p} , dep = {gen_noise}")
            # Simulate both general noise and bit-flip noise simultaneously
            qc_with_noise, noise_model = qft_circuit(qbits_num, initial_state, 
                                                     add_bit_flip=True, bit_flip_prob=p,
                                                     general_noise=True, general_noise_prob=gen_noise)
            qc_with_noise.measure_all()
            counts_with_noise = simulate_circuit(qc_with_noise, shots, noise_model=noise_model)

            # Calculate the square root of the sum of errors squared
            error_squared = (counts_with_noise - final_state) ** 2
            mse.append(np.sqrt(np.sum(error_squared)))

            if plot:
                plt.bar(labels, counts_with_noise)
                plt.scatter(labels, final_state, color='red')
                plt.show()

        # Plot the results
        plt.plot(p_noise_arr, mse, label=f"Depolarizing Error Param: {gen_noise}")
    
    plt.grid(alpha=0.3)
    plt.title(f"RSSE Vs Bit-flip noise probability for different Depolarizing Error Parameter , initial state : {initial_state_name}")
    plt.xlabel("Bit-flip Probability")
    plt.ylabel("RSSE")
    plt.legend(fontsize='small')

    plt.show()

def qft_error_correction(n  , initial_state_name = None , bit_flip_prob = 0.5, general_noise = False , general_noise_prob = 0.1):
    
    q = QuantumRegister(n,'q')
    encod = QuantumRegister(2 , 'e')
    decod=  QuantumRegister(2 , 'd')
    c = ClassicalRegister(2,'c')

    meas = ClassicalRegister(n, 'm')

    qc = QuantumCircuit(q, encod , decod , c , meas)
    qc.barrier()
    if initial_state_name is not None:
        initial_state = create_initial_state(initial_state_name , n)

        # normalize the initial state
        initial_state = initial_state / np.linalg.norm(initial_state)

        # Check that the length of initial_state is correct
        if (initial_state.shape[0] != 2**n):
            raise ValueError("The length of initial_state must equal 2**n_qubits.")

        # Initialize the qubits
        qc.initialize(initial_state, range(n))

    qc.cx(q[0],encod[0])
    qc.cx(q[0],encod[1])
    qc.barrier()
    my_custom_gate = pauli_error([('X', bit_flip_prob), ('I', 1 - bit_flip_prob)])
    #my_custom_gate = Gate(name=f"BF", num_qubits=1, params=[])

    qc.append(my_custom_gate , [0])
    qc.append(my_custom_gate , [n])
    qc.append(my_custom_gate , [n+1])

    qc.barrier()
    
    qc.cx(q[0],decod[0])
    qc.cx(encod[0],decod[0])
    qc.cx(encod[0],decod[1])
    qc.cx(encod[1],decod[1])

    qc.barrier()

    qc.measure(decod[0] , c[0])
    qc.measure(decod[1] , c[1])

    qc.x(q[0]).c_if(c, 1)
    qc.x(encod[0]).c_if(c, 3)
    qc.x(encod[1]).c_if(c, 2)

    qc.barrier()
    qc.cx(q[0],encod[0])
    qc.cx(q[0],encod[1])

    qc.barrier()
    qftc , _= qft_circuit(n)
    qc.compose(qftc, inplace=True)
    qc.barrier()


    if general_noise:
        noise_model = NoiseModel()
        #error_gate1 = pauli_error([('X', general_noise_prob), ('I', 1 - general_noise_prob)])
        error_gate1 = depolarizing_error(general_noise_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_gate1, ['h'])

        qc.measure(q , meas)

        return qc , noise_model


    qc.measure(q , meas)

    return qc , None

def compare_circuits(n, initial_state=None, bit_flip_prob=0.5, general_noise=False, general_noise_prob=0.1 , plot = False):
    qbits_num = n
    initial_state_name = initial_state
    # Generate initial state if provided
    if initial_state is not None:
        initial_state = create_initial_state(initial_state , n)

    # Create QFT circuit with error correction
    qc_error_correction, noise_model_correction = qft_error_correction(n, initial_state, bit_flip_prob, general_noise = general_noise, general_noise_prob = general_noise_prob)
    
    # Create regular QFT circuit
    qc_no_error_correction, noise_model_no_correction = qft_circuit(n , initial_state = initial_state , add_bit_flip = True , bit_flip_prob = bit_flip_prob , general_noise = general_noise , general_noise_prob = general_noise_prob)
    qc_no_error_correction.measure_all()
    # Initialize simulator
    simulator_0 = Aer.get_backend('qasm_simulator')
    simulator_1 = Aer.get_backend('qasm_simulator')

    # Run the circuits
    job_error_correction = execute(qc_error_correction, simulator_0, shots=10000, noise_model=noise_model_correction)
    job_no_error_correction = execute(qc_no_error_correction, simulator_1, shots=10000 , noise_model=noise_model_no_correction)

    # Get the results
    result_error_correction = job_error_correction.result()
    result_no_error_correction = job_no_error_correction.result()

    # Get the counts (how many times each possible state was measured)
    counts_error_correction = result_error_correction.get_counts(qc_error_correction)
    counts_no_error_correction = result_no_error_correction.get_counts(qc_no_error_correction)

     # create a list of all possible output states
    states = [format(i, '0%sb' % qbits_num) for i in range(2**qbits_num)]

    for state in states:
        if state not in counts_no_error_correction:
            counts_no_error_correction[state] = 0  
    
    counts_no_error_correction = np.array([counts_no_error_correction[state] for state in states])
    counts_no_error_correction = counts_no_error_correction / 10000


    counts_error_correction = filter_bit_histogram(counts_error_correction, list(range(n+1,1,-1))) 
    sorted_keys = sorted(counts_error_correction.keys(), key=lambda x: int(x, 2))
    # Create the ordered array of values
    counts_error_correction = np.array([counts_error_correction[key] for key in sorted_keys])
    counts_error_correction = counts_error_correction / 10000


    final_state = custom_fft(initial_state , n)
    if plot:
        plt.figure()
        xaxis = np.arange(len(states))
        plt.bar(xaxis - 0.2, counts_error_correction , 0.4, color = 'blue' , label = "Error Correction")
        plt.bar(xaxis + 0.2, counts_no_error_correction ,0.4, color = 'c' , label = "No Error Correction")
        plt.scatter(xaxis , final_state , color = 'red' , label = "Ideal")
        plt.grid(alpha = 0.3)

        plt.xticks(xaxis, states)
        plt.xticks(rotation=90)
        plt.title(f"Histogram Comparison, p_bit_flip = {bit_flip_prob:.2f} , Initial State: {initial_state_name}")

        plt.legend()

    error_squared_correction = (counts_error_correction - final_state) ** 2
    error_squared_correction = np.sqrt(np.sum(error_squared_correction))

    error_squared_no_correction = (counts_no_error_correction - final_state) ** 2
    error_squared_no_correction = np.sqrt(np.sum(error_squared_no_correction))

    _relative_entropy_correction = relative_entropy(final_state , counts_error_correction)
    _relative_entropy_no_correction = relative_entropy(final_state , counts_no_error_correction)

    return [error_squared_correction , error_squared_no_correction , _relative_entropy_correction , _relative_entropy_no_correction , np.log(_relative_entropy_correction) , np.log(_relative_entropy_no_correction)]

def run_simulations(n, initial_state=None, prob_range=(0, 1, 0.1), general_noise=False, general_noise_prob=0.1):
    bit_flip_probs = np.arange(*prob_range)
    errors_squared_correction = []
    errors_squared_no_correction = []
    relative_entropies_correction = []
    relative_entropies_no_correction = []
    log_relative_entropies_correction = []
    log_relative_entropies_no_correction = []

    for i , bit_flip_prob in enumerate(bit_flip_probs):
        error_correction, error_no_correction, relative_entropy_correction, relative_entropy_no_correction, log_relative_entropy_correction, log_relative_entropy_no_correction = compare_circuits(n, initial_state, bit_flip_prob, general_noise, general_noise_prob, plot = bit_flip_prob in [bit_flip_probs[2] , bit_flip_probs[6] , bit_flip_probs[10]])
        errors_squared_correction.append(error_correction)
        errors_squared_no_correction.append(error_no_correction)
        relative_entropies_correction.append(relative_entropy_correction)
        relative_entropies_no_correction.append(relative_entropy_no_correction)
        log_relative_entropies_correction.append(log_relative_entropy_correction)
        log_relative_entropies_no_correction.append(log_relative_entropy_no_correction)

        print(bit_flip_prob)

    plt.figure()
    plt.plot(bit_flip_probs, errors_squared_correction, label='Error Correction')
    plt.plot(bit_flip_probs, errors_squared_no_correction, label='No Error Correction')
    plt.ylabel('RSSE')
    plt.title('RSSE for different bit flip probabilities')
    plt.xlabel('Bit flip probability')
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.figure()
    plt.plot(bit_flip_probs, relative_entropies_correction, label='Error Correction')
    plt.plot(bit_flip_probs, relative_entropies_no_correction, label='No Error Correction')
    plt.ylabel('Relative Entropy')
    plt.title('Relative Entropy for different bit flip probabilities')
    plt.xlabel('Bit flip probability')
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.figure()
    plt.plot(bit_flip_probs, log_relative_entropies_correction, label='Error Correction')
    plt.plot(bit_flip_probs, log_relative_entropies_no_correction, label='No Error Correction')
    plt.ylabel('Log Relative Entropy')
    plt.title('Log Relative Entropy for different bit flip probabilities')

    plt.xlabel('Bit flip probability')
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()


if __name__ == "__main__":
    action_list = ["draw_circuits" , "plot_all_fft" , "qft_accuracy_simulation_animation" , "qft_simulate_bit_flip_noise" , "qft_simulate_dep_noise" , "qft_simulate_mixed_noise" , "qft_simulate_changing_qubits", "qft_error_correction"]
    action = "qft_error_correction"

    #draw circuit
    if action == "draw_circuits":
        qbits_num = 5
        draw_circuit(qbits_num , scale = 0.7 , fold = 100)
        plt.show()
    
    # draw all fft
    if action == "plot_all_fft":
        initial_states = ["zero" ,  "mixed" , "gaussian" , "cos" , "square"]
        qbits_num = 5
        plot_all_fft(qbits_num, initial_states)
        plt.show()

    # convergence to number of shots with animation simulation
    if action == "qft_accuracy_simulation_animation":
        qft_accuracy_simulation_animation(qbits_num = 5 , initial_state="gaussian" , animate = True)
        #qft_accuracy_simulation_animation(qbits_num = 5 , initial_state="cos" , animate = True)
    

    if action == "qft_simulate_bit_flip_noise":
        #simulate_bit_flip_noise(qbits_num=5, shots=10000, initial_state_name="gaussian", plot=True)
        #simulate_bit_flip_noise(qbits_num=qbits_num, shots=10000, initial_state_name="cos", plot=True)
        plot_all_statistics(qbits_num=5, shots=1000, err = "bit_flip")
        plt.show()
    
    if action == "qft_simulate_dep_noise":
        #simulate_general_noise(qbits_num=5, shots=10000 , initial_state_name = "gaussian" , plot = True)
        #simulate_general_noise(qbits_num=5, shots=10000 , initial_state_name = "squacosre" , plot = True)
        plot_all_statistics(qbits_num=5, shots=1000, err = "general")
        plt.show()
    
    if action == "qft_simulate_mixed_noise":
        #simulate_mixed_noise_for_different_general_noise(initial_state_name="zero" , shots = 1000)
        simulate_mixed_noise_for_different_general_noise(initial_state_name="gaussian" , shots = 10000)
        # simulate_mixed_noise_for_different_general_noise(initial_state_name="cos" , shots = 10000)
        # simulate_mixed_noise_for_different_general_noise(initial_state_name="square" , shots = 10000)
        # simulate_mixed_noise_for_different_general_noise(initial_state_name="zero" , shots = 10000)
        # simulate_mixed_noise_for_different_general_noise(initial_state_name="mixed" , shots = 10000)
        plt.show()


    if action == "qft_simulate_changing_qubits":
        simulate_general_noise_for_changing_qbits(initial_state_name="square", shots=100000, plot=False)
        # simulate_general_noise_for_changing_qbits(initial_state_name="cos", shots=100000, plot=False)
        # simulate_general_noise_for_changing_qbits(initial_state_name="zero", shots=100000, plot=False)
        plt.show()
    
    
    if action == "qft_error_correction":
        run_simulations(5, initial_state="gaussian", prob_range=(0, 1, 0.05), general_noise=False, general_noise_prob=0.1)
        # run_simulations(n, initial_state="cos", prob_range=(0, 1, 0.05), general_noise=False, general_noise_prob=0.1)

    