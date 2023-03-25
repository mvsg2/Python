import pennylane as qml
import numpy as np

def qft(n):
    for i in range(n):
        for j in range(i+1, n):
            qml.CNOT(wires=[i, j])
        qml.RX(np.pi/2, wires=i)
        qml.CRZ(np.pi/2**(i+1), wires=[i]+list(range(i+1, n)))

dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def qft_circuit(inputs):
    qft(3)
    return qml.probs(wires=range(3))

input_state = np.array([1,0,0])
print(qft_circuit(input_state))
