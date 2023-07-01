# import required libraries
import math
import matplotlib.pyplot as plt
from qiskit import execute , Aer
from qiskit import QuantumCircuit, transpile
from qiskit.tools.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error


simulator = Aer.get_backend('qasm_simulator')

pi = math.pi
p_meas = 0.1

error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
noise_meas = NoiseModel()
noise_meas.add_all_qubit_quantum_error(error_meas, "measure")

# Create Quantum Fourier Transform circuit with qbits_num qubits
def qft_circuit(qbits_num = 3):
    
    # Create the circuit
    circuit = QuantumCircuit(3,0)

    # Encode the state |101>
    circuit.x(0)
    circuit.x(2)

    # start of QFT
    circuit.h(2)
    circuit.cp(pi / 4, 2, 0)
    circuit.cp(pi / 2, 2, 1)
    circuit.h(1)
    circuit.cp(pi / 2, 1, 0)
    circuit.h(0)
    circuit.swap(2, 0)

    #end of QFT
    circuit.barrier()
    #Start of IQFT
    circuit.swap(0, 2)
    circuit.h(0)
    circuit.cp(-pi / 2, 1, 0)
    circuit.h(1)
    circuit.cp(-pi / 2, 2, 1)
    circuit.cp(-pi / 4, 2, 0)
    circuit.h(2)

    circuit.measure_all()


    return circuit

def excecute_circuit(circuit , shots = 10000):

    # Run ideal simulation
    job = execute(circuit, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)

    #Plot ideal
    plot_histogram(counts)
    circuit.draw(output='mpl')
    plt.show()


# # add error to one of the X gates
# p_gate1 = 0.05
# p_gate2 = 0.35
# error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])

# # Add errors to noise model
# noise_x = NoiseModel()
# noise_x.add_all_qubit_quantum_error(error_gate1, ["x"])

# print(noise_x)

# job = execute(circuit, simulator,
#                   basis_gates=noise_x.basis_gates,
#                   noise_model=noise_x)
# result = job.result()
# counts = result.get_counts(circuit)

# # Plot noisy output
# plot_histogram(counts)
# plt.show()



## if we want to see the statevector for debuding
# sim = Aer.get_backend("aer_simulator")
circuit_init = circuit.copy()
circuit_init.save_statevector()
# statevector = sim.run(circuit_init).result().get_statevector()
# print('State Vector:', statevector)



if __name__ == "__main__":
    

    # create quantum fourier transform circuit
    circuit = qft_circuit(qbits_num = 3)

    # excecute the circuit
    excecute_circuit(circuit)

    
