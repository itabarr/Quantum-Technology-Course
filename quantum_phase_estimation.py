import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
from qiskit import Aer, execute

from utils import * 

# Create Quantum Phase Estimation Circuit, num_qubits, with Phase Gate of theta
def create_qpe_circuit(theta, num_qubits):
    first = QuantumRegister(size=num_qubits, name="x") 
    second = QuantumRegister(size=1, name="state")
    classical = ClassicalRegister(size=num_qubits, name="meas")
    qpe_circuit = QuantumCircuit(first, second, classical)
    qpe_circuit.x(second)
    qpe_circuit.barrier()
    qpe_circuit.h(first)
    qpe_circuit.barrier()
    for j in range(num_qubits):
        qpe_circuit.cp(theta * 2 * np.pi * (2**j), j, num_qubits)
    qpe_circuit.barrier()
    qpe_circuit.compose(QFT(num_qubits, inverse=True), inplace=True)
    qpe_circuit.barrier()
    qpe_circuit.measure(first, classical)
    return qpe_circuit

# Simulate qpe circuit
def simulate_qpe_circuit(theta, num_qubits, shots=1000):
    qpe_circuit = create_qpe_circuit(theta, num_qubits)
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qpe_circuit, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(qpe_circuit)
    return counts

# Get a for ideal calculation
def get_a(n , theta):
    a = theta* (2**n)
    a = round(a)
    return a 

# Get delta for ideal calculations
def get_delta(n , theta):
    a = get_a(n,theta)
    return theta - (a / (2**n))

# Calculate ideal probabillity for specific bin  
def cx(n, a, delta):
    """Calculate cx for given a and delta."""
    x_values = np.arange(2**n)
    k_values = np.arange(2**n)
    cx_values = []
    for x in x_values:
        cx = (1/2**n) * np.sum(np.exp(-2j * np.pi * k_values / 2**n * (x - a)) * np.exp(2j * np.pi * delta * k_values))
        cx_values.append(np.abs(cx)**2)  # take the absolute value as we're dealing with complex numbers
    return x_values, np.array(cx_values)

# Helper to add text to bar
def addlabels(x,y , _range = []):
    for i in range(len(x)):
        if len(_range) != 0:
            if _range[0] % len(x) <= i <=_range[1] % len(x):
                plt.text(i, y[i]+ 2, f"{y[i]}", ha = 'center', fontsize=7)
            
        else:
            plt.text(i, y[i]+ 2, f"{y[i]}", ha = 'center', fontsize=7)

# Plot ideal probabillities 
def plot_cx(n, theta , range = [] ):
    a = get_a(n , theta)
    delta = get_delta(n,theta)
    x_values, cx_values = cx(n, a, delta)

    plt.bar(x_values, cx_values)
    addlabels(x_values , cx_values , range)
    
    plt.xlabel('theta')
    plt.ylabel('prob')
    plt.title(f'Ideal Probability to Measure Theta ,  n = {n}, theta = {theta}')
    plt.grid()

    plt.xticks(x_values, x_values / (2**n), fontsize = 7, rotation = 'vertical')
    
    if len(range) != 0:
        plt.xlim([x_values[range[0]] - 0.3  ,x_values[range[1]] +0.3])

# Custom plot for counts
def plot_counts(counts, num_qubits , theta , range = []):
    states , num_counts = counts_to_arrays(counts=counts , num_qubits= num_qubits)
    integers = binary_to_int(states)

    plt.bar(integers, num_counts)
    addlabels(integers , num_counts , range)
    plt.xlabel('theta')
    plt.ylabel('counts')
    plt.title(f'Measured Counts vs Theta ,  n = {num_qubits}, theta = {theta} , shots = 10000')
    plt.grid(alpha = 0.3)
    plt.xticks(integers, integers / (2**n), fontsize = 7, rotation = 'vertical')

    if len(range) != 0:
        plt.xlim([integers[range[0]] - 0.3  ,integers[range[1]] +0.3])

if __name__ == "__main__":

    # Draw qpe circuit , chnage num_qubits and theta for diffrent circuits
    num_qubits = 5
    theta = 0.5
    qpe_circuit_fixed_phase = create_qpe_circuit(theta, num_qubits)
    qpe_circuit_fixed_phase.draw("mpl" , scale = 0.8)
    
    # Plot ideal 1
    plt.figure()
    n =  5 # Number of qubits
    theta = 0.73
    plot_cx(n, theta)
    #plot_cx(n, theta , [-75 , -65])
    
    plt.figure()
    n =  5 # Number of qubits
    theta = 0.73
    counts = simulate_qpe_circuit(theta, n , shots=1000)
    plot_counts(counts, n , theta )
    #plot_counts(counts, n , theta , range = [-75 , -65])

    plt.show()




