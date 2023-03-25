from QuantumCircuits_utilities import entanglement_swapping
from qiskit import *
from qiskit.visualization import *
import matplotlib.pyplot as plt

qc = entanglement_swapping()

print(qc)

b = Aer.get_backend('aer_simulator')
j = execute(qc, backend=b, shots=10000)
r = j.result()
c = r.get_counts()
print(c)

plot_histogram(c)
plt.show()
plt.close()