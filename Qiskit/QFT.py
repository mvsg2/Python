import numpy as np

def qft(qc, qr):
    """Quantum Fourier Transform on the given Quantum Register."""
    n = len(qr)
    for i in range(n):
        qc.h(i)
        for j in range(i+1, n):
            qc.cu1(np.pi/float(2**(j-i)), j, i)
    qc.barrier()