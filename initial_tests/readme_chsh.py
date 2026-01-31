import numpy as np
from qrisp import QuantumVariable, cx, h, ry, x, z
from qrisp.interface import IQMBackend
from qrisp.operators import Z


def CHSH_measurement(A, B):
    # Implements a circuit for calculating an expectation values in <S>

    # prepare a singlet state
    singlet = QuantumVariable(2)
    h(singlet[0])
    cx(singlet[0], singlet[1])
    x(singlet[1])
    z(singlet[0])

    # If Alice chooses measurement A2, measure in the X basis. Otherwise use the Z basis.
    if A == 2:
        h(singlet[0])

    # Same for Bob, but we have to rotate his measurement basis first
    ry(-np.pi / 4, singlet[1])
    if B == 2:
        h(singlet[1])

    print(singlet.qs)
    return singlet


def ZZ(circ, sim=True):
    if sim:
        results = circ.get_measurement()
    else:
        results = circ.get_measurement(backend=quantum_computer)
    return results["00"] + results["11"] - results["01"] - results["10"]


CHSH_sim = (
    ZZ(CHSH_measurement(1, 1))
    - ZZ(CHSH_measurement(1, 2))
    + ZZ(CHSH_measurement(2, 1))
    + ZZ(CHSH_measurement(2, 2))
)
print("Simulated result: ", CHSH_sim)

quantum_computer = IQMBackend(
    api_token="",
    device_instance="garnet",
)  # Change this to change which device you run on

# CHSH_qc = (
#    ZZ(CHSH_measurement(1, 1), False)
#    - ZZ(CHSH_measurement(1, 2), False)
#    + ZZ(CHSH_measurement(2, 1), False)
#    + ZZ(CHSH_measurement(2, 2), False)
# )

# print("Quantum computer result: ", CHSH_qc)
