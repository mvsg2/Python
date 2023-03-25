from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
from qiskit.visualization import plot_histogram
import math
def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    for j in range(n):
        circ.h(j)
        for k in range(j+1,n):
            circ.cu1(-math.pi/float(2**(k-j)),k,j)
        circ.swap(j, n-1)
    circ.reverse_bits()

def qft(circ, n):
    """n-qubit QFT on the first n qubits in circ"""
    for j in range(n):
        k = n-j-1
        circ.swap(k, n-1)
        for l in range(k, 0, -1):
            circ.cu1(math.pi/float(2**(k-l+1)),l-1,k)
        circ.h(k)

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)

# Apply QFT to the first three qubits
qft(qc, 3)

# Measure the qubits
for i in range(3):
    qc.measure(q[i], c[i])

# Print the circuit
print(qc.draw())

# Execute the circuit on a simulator
backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(qc, backend, shots=shots).result()
counts = results.get_counts()
plot_histogram(counts, title="QFT")