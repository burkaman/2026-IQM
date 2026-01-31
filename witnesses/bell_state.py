import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm.iqm_provider import IQMProvider

# ----------------------------
# 1) State preparation
# ----------------------------
def prepare_phi_plus():
    """|Phi+> = (|00> + |11>) / sqrt(2)"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# ----------------------------
# 2) Basis rotations for measurement
# ----------------------------
def rotate_for_basis_measurement(qc: QuantumCircuit, qubit: int, basis: str):
    """
    Qiskit measures in the computational (Z) basis by default.
    To measure in X or Y basis, rotate that basis onto Z first:

    - X basis: apply H, then measure Z  (since H Z H = X)
    - Y basis: apply Sdg then H, then measure Z  (common standard trick)  :contentReference[oaicite:1]{index=1}
    - Z basis: do nothing
    """
    b = basis.upper()
    if b == "X":
        qc.h(qubit)
    elif b == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif b == "Z":
        pass
    else:
        raise ValueError("basis must be 'X', 'Y', or 'Z'")

# ----------------------------
# 3) Correlator estimation from counts
# ----------------------------
def correlator_from_counts(counts: dict, shots: int) -> float:
    """
    Each shot gives two bits. Map bit 0->(+1), bit 1->(-1) for Z measurement.
    The product eigenvalue is +1 for even parity and -1 for odd parity:
      sign = (-1)^(#ones)
    This works regardless of bitstring endianness for 2-qubit correlators.
    """
    total = 0.0
    for bitstring, c in counts.items():
        ones = bitstring.count("1")
        sign = +1 if (ones % 2 == 0) else -1
        total += sign * c
    return total / shots

def measure_two_qubit_correlator(basisA: str, basisB: str, shots: int = 20_000, seed: int = 1234) -> float:
    """
    Estimates <basisA ⊗ basisB> on |Phi+>.
    """
    #qc = prepare_phi_plus()
    qc = QuantumCircuit(2, 2)

    # Rotate Alice (qubit 0) and Bob (qubit 1) into the requested measurement bases
    rotate_for_basis_measurement(qc, 0, basisA)
    rotate_for_basis_measurement(qc, 1, basisB)

    # Measure both qubits in Z basis
    qc.measure([0, 1], [0, 1])

    # set backend to IQM hardware
    provider = IQMProvider("https://resonance.meetiqm.com", quantum_computer="sirius",
                           token="oXhAT4UD1nBMCR3WQhf976lyDRM7DRxdPEVO2GuQ3lQBnBVHVfd7orxK23ufeZDB")
    backend = provider.get_backend()

    # set backend to simulator
    #backend = AerSimulator(seed_simulator=seed)

    # transpile and run
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    return correlator_from_counts(counts, shots)

# ----------------------------
# 4) Witness for |Phi+>
# ----------------------------
def bell_phi_plus_witness(shots: int = 20_000):
    Exx = measure_two_qubit_correlator("X", "X", shots=shots)
    Eyy = measure_two_qubit_correlator("Y", "Y", shots=shots)
    Ezz = measure_two_qubit_correlator("Z", "Z", shots=shots)

    # Witness: <W> = (1 - <XX> + <YY> - <ZZ>) / 4
    W = 0.25 * (1.0 - Exx + Eyy - Ezz)

    # Fidelity with |Phi+>: F = <Phi+|rho|Phi+> = (1 + <XX> - <YY> + <ZZ>) / 4 = 0.5 - W
    F = 0.25 * (1.0 + Exx - Eyy + Ezz)

    # Simple per-correlator standard error (each shot is ±1 RV)
    def stderr(E): 
        return math.sqrt(max(0.0, 1.0 - E*E) / shots)

    print(f"shots = {shots}")
    print(f"<XX> = {Exx:.6f}  (stderr ~ {stderr(Exx):.6f})")
    print(f"<YY> = {Eyy:.6f}  (stderr ~ {stderr(Eyy):.6f})")
    print(f"<ZZ> = {Ezz:.6f}  (stderr ~ {stderr(Ezz):.6f})")
    print(f"<W>  = {W:.6f}   (negative => entangled)")
    print(f"F    = {F:.6f}   (entangled certified if F > 0.5)")
    print("Certified entangled?", "YES" if W < 0 else "NO (not certified)")

if __name__ == "__main__":
    bell_phi_plus_witness(shots=1)