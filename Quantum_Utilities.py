import numpy as np
import functools as ft

X = np.array([[0,1], [1,0]])   # Applying a Pauli X gate
H = (1/np.sqrt(2))*np.array([[1,1], [1,-1]]) # Applying a Hadamard gate
Z = np.array([[1,0], [0,-1]])
CX = [[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]]  # For most entanglement circuits, control 0 is used, so I am defining it separately in the list form
CX_control1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]) # CNOT gate with control qubit as 1
CX_control0 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]]) # CNOT gate with control qubit as 0

def initialize_state(num_qubits):
    """Prepare a qubit in state |0>.
    
    Returns:
        array[float]: the vector representation of state |0>.
    """
    # PREPARE THE STATE |0>
    zero_components = [0 for _ in range(2**num_qubits-1)]   
    return np.array([1] + zero_components)

def complex_norm(z):
    re = np.real(z)
    im = np.imag(z)
    return np.sqrt(re**2 + im**2)

def normalize_state(alpha, beta):
    """Compute a normalized quantum state given arbitrary amplitudes.
    
    Args:
        alpha (complex): The amplitude associated with the |0> state.
        beta (complex): The amplitude associated with the |1> state.
        
    Returns:
        array[complex]: A vector (numpy array) with 2 elements that represents
        a normalized quantum state.
    """
    norm1 = complex_norm(alpha)
    norm2 = complex_norm(beta)
    if (norm1**2 + norm2**2)==1:
        print("The state is already normalized!")
        return
    else:
        state = np.array([alpha, beta])
        norm = np.sqrt(norm1**2 + norm2**2)
        for i in range(len(state)):
            state[i] /= norm
        return state

def complex_innerProduct(z1, z2):
    inner_product = np.dot(np.conjugate(z1),z2)
    return inner_product

def measure_state(state, num_meas):
    """Simulate a quantum measurement process.

    Args:
        state (array[complex]): A normalized qubit state vector. 
        num_meas (int): The number of measurements to take
        
    Returns:
        array[int]: A set of num_meas samples, 0 or 1, chosen according to the probability 
        distribution defined by the input state.
    """
    # COMPUTE THE MEASUREMENT OUTCOME PROBABILITIES
    norm1 = complex_norm(state[0])
    norm2 = complex_norm(state[1])
    measurements = np.random.choice([0,1], size=num_meas, p=[norm1**2, norm2**2])
    # RETURN A LIST OF SAMPLE MEASUREMENT OUTCOMES
    return measurements

def apply_U_Gate(U, state):
    """Apply a quantum operation."""
    return np.dot(U, state)

def measure_state(state, num_meas):
    """Measure a quantum state num_meas times."""
    probs = []
    for i in range(len(state)):
        prob = np.abs(state[i])**2
        probs.append(prob)
    # p_alpha = np.abs(state[0]) ** 2
    # p_beta = np.abs(state[1]) ** 2
    meas_outcome = np.random.choice([0, 1], p=probs, size=num_meas)
    return meas_outcome

def quantum_algorithm(gates_to_be_applied, no_times_to_run):
    """Use the functions above to implement the quantum algorithm described below.
    1. Prepare the qubit in the |0> state
    2. Apply the unitary gate transformation to the qubit
    3. Measure the state of the qubit 
    Returns:
        array[int]: the measurement results after running the algorithm some specified number of times
    """
    # PREPARE THE STATE, APPLY U, THEN TAKE MEASUREMENT SAMPLES
    init = initialize_state(1)
    for gate in gates_to_be_applied:
        state_after_U_applied = apply_U_Gate(gate, init)
        init = state_after_U_applied
    final_state = state_after_U_applied
    meas = measure_state(state_after_U_applied, no_times_to_run) # For simplicity, just one gate is applied, but any sequential steps in an algorithm can be applied
    return meas, final_state                                                         # for instance, the entanglement logic coded below in a function

def entanglement_circuit(num_qubits):
    """Entangles all the given qubits

    Args:
        num_qubits (int) : The number of qubits the user wants to entangle
    Returns:
        statevector (numpy.array) : The final statevector once all the qubits are entangled    
    """
    # s0 = np.tensordot(H, np.identity(n=2**(num_qubits-1)))  # This will throw a shape mis-match error
    s0 = initialize_state(num_qubits=num_qubits)
    u0 = np.kron(np.identity(n=2**(num_qubits-1)), H)
    s0_1 = apply_U_Gate(U=u0, state=s0)

    id_above = []
    for i in range(num_qubits-1):
        id_above.append([np.identity(n=2) for _ in range(i)])
    id_below = id_above[::-1]

    unitaries = []
    for j in range(num_qubits-1):
        l = id_below[j]+[CX]+id_above[j]
        uj = ft.reduce(np.kron, l)
        unitaries.append(uj)

    print(f"Number of CNOT units: {len(unitaries)}")
    print("The unitaries are:\n", unitaries)
    print()

    state_vectors = []
    for k in range(num_qubits-1):
        sv = apply_U_Gate(U=unitaries[k], state=s0_1)
        state_vectors.append(sv)
        s0_1 = sv
    print("The final statevector is:")

    return state_vectors[-1]

# z = 2 + 3j
# w = 4-5j
# print(complex_innerProduct(z, w))
# print(complex_innerProduct(w, z))

# # From these two lines below, it's obvious that np.dot() is not a reliable method for taking (z*)z for complex inner products
# # print(np.dot(z,w))
# # print(np.dot(w,z))

# a, b = np.array([1j,2]), np.array([3,4j])
# print(complex_innerProduct(a,b))
# print(complex_innerProduct(b,a))

# # Let's say the state of a qubit is given by (not normalized) [z w] where z corresponds to the amplitude of |0> state and w of |1> state
# print(normalize_state(z, w))

# x = [1,2]
# y = [6, 1+7j]
# # print(np.dot(x, y))
# # print(np.dot(y, x))

# print(apply_U_Gate(U=X, state=[1,0]))
# print(apply_U_Gate(U=H, state=[1/np.sqrt(2),1/np.sqrt(2)]))

# print(apply_U_Gate(U=CX_control0, state=[0,1,0,0]))
# print(apply_U_Gate(U=CX_control1, state=[0,0,1,0]))
# print(apply_U_Gate(U=H, state=[1,0]))

# m, f = quantum_algorithm(gates_to_be_applied=[X], no_times_to_run=10)

# print(f"The final state is: {f}")
# print("The measurement results are: ", m)

if __name__ == "__main__":
    n = int(input("How many qubits? "))
    print(entanglement_circuit(num_qubits=n))
