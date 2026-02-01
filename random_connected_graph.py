import random
from qiskit import QuantumCircuit

def generate_graph_state_exact_depth(N, D, p_extra=0.3, seed=None):
    """
    Generate a random connected graph state of N qubits with circuit depth exactly D.

    Parameters
    ----------
    N : int
        Number of qubits.
    D : int
        Desired circuit depth (number of sequential layers of CZ gates).
    p_extra : float
        Probability of adding extra edges per layer.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    QuantumCircuit
        Circuit preparing the graph state.
    """
    if N <= 0 or D <= 0:
        raise ValueError("N and D must be positive integers")

    if seed is not None:
        random.seed(seed)

    # Initialize adjacency and layers
    adjacency = {i: set() for i in range(N)}
    layers = [[] for _ in range(D)]
    qubits_used = [set() for _ in range(D)]

    # Step 1: Build a random spanning tree to guarantee connectivity
    nodes = list(range(N))
    random.shuffle(nodes)
    for i in range(N - 1):
        u, v = nodes[i], nodes[i + 1]
        # assign edge to a random available layer
        available_layers = [k for k in range(D) if u not in qubits_used[k] and v not in qubits_used[k]]
        layer = random.choice(available_layers)
        layers[layer].append((u, v))
        qubits_used[layer].update([u, v])
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Step 2: Add extra edges randomly per layer
    for layer_idx in range(D):
        for i in range(N):
            for j in range(i + 1, N):
                if j not in adjacency[i] and i not in qubits_used[layer_idx] and j not in qubits_used[layer_idx]:
                    if random.random() < p_extra:
                        layers[layer_idx].append((i, j))
                        qubits_used[layer_idx].update([i, j])
                        adjacency[i].add(j)
                        adjacency[j].add(i)

    # Step 3: Build Qiskit circuit
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)

    for layer in layers:
        for u, v in layer:
            qc.cz(u, v)

    return qc
