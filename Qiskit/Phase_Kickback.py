from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.x(1)
qc.h([0,1])
#qc.x([0,1])
qc.cnot(0,1)
#qc.x([0,1])
#qc.h([0,1])
print(qc)
print(Statevector(data=qc))