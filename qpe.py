from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.circuit.library import XGate
import matplotlib.pyplot as plt

def create_phase_estimation_circuit(unitary, eigenvector, precision):
    """
    Create a Quantum Phase Estimation circuit.
    
    Args:
        unitary (QuantumCircuit): The unitary operator whose eigenvalues we want to find.
        eigenvector (QuantumCircuit): The eigenvector of the unitary operator.
        precision (int): The number of precision qubits for the phase estimation.
    
    Returns:
        QuantumCircuit: The phase estimation circuit.
    """
    num_unitary_qubits = unitary.num_qubits
    qc = QuantumCircuit(precision + num_unitary_qubits, precision)

    # Apply Hadamard gates to the precision qubits
    qc.h(range(precision))

    # Prepare the eigenvector
    qc.compose(eigenvector, range(precision, precision + num_unitary_qubits), inplace=True)

    # Apply controlled unitary operations
    for i in range(precision):
        controlled_unitary = unitary.control()
        qc.compose(controlled_unitary, [i] + list(range(precision, precision + num_unitary_qubits)), inplace=True)
        unitary = unitary.compose(unitary)

    # Apply inverse QFT
    iqft = QFT(precision, do_swaps=True, inverse=True, name='iqft')
    qc.compose(iqft, range(precision), inplace=True)

    # Measure the precision qubits
    qc.measure(range(precision), range(precision))

    # Draw the circuit
    qc.draw('mpl')
    return qc

# Define the unitary operator as a quantum circuit
unitary = QuantumCircuit(1)
unitary.append(XGate(), [0])

# Define the eigenvector as a quantum circuit
eigenvector = QuantumCircuit(1)
eigenvector.x(0)

# Define the precision for the phase estimation
precision = 3

# Create the phase estimation circuit
pec = create_phase_estimation_circuit(unitary, eigenvector, precision)

plt.show()