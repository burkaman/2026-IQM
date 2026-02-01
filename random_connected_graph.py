import random
from qiskit import QuantumCircuit


def generate_connected_graph(n):
    """Generate a random connected graph with n nodes.

    Returns a dictionary mapping each node to a set of its neighbors.
    The graph is guaranteed to be connected (every node is reachable from
    every other node) but is not necessarily complete.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return {0: set()}

    adjacency = {i: set() for i in range(n)}

    # Step 1: Build a random spanning tree to guarantee connectivity.
    # Shuffle nodes and connect them in a random chain.
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Step 2: Add extra random edges (each possible edge included with ~30% probability).
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adjacency[i] and random.random() < 0.3:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def graph_to_circuit(adjacency):
    """Convert a graph adjacency dict into a Qiskit circuit with CZ gates.

    Each node becomes a qubit. Each edge becomes a CZ gate.
    All qubits are initialized in the |+> state (H gate) so the CZ gates
    create a graph state.
    """
    n = len(adjacency)
    qc = QuantumCircuit(n)

    # Prepare |+> state on every qubit
    for i in range(n):
        qc.h(i)

    # Apply CZ for each edge (iterate sorted pairs to avoid duplicates)
    for i in sorted(adjacency):
        for j in sorted(adjacency[i]):
            if j > i:
                qc.cz(i, j)

    return qc

def generate_graph_state(n):
    adj = generate_connected_graph(n)
    return graph_to_circuit(adj)