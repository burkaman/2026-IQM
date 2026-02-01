"""
Topology-Aware Linear Cluster State Witness

Uses branch-and-bound DFS to find the lowest-error path of n qubits through
the hardware connectivity graph, then runs the Toth & Guhne Theorem 2 witness
W = 3 - 2*(P_A + P_B), where W < 0 implies GME.

The circuit is transpiled onto the exact physical path found by the optimizer,
so no SWAP gates are needed and the circuit matches the hardware topology.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from iqm.qiskit_iqm.iqm_provider import IQMProvider


# ============================================================
# PATH OPTIMIZER (branch-and-bound DFS)
# ============================================================

def build_adjacency(edges):
    """Build adjacency dict from edge list [(u, v, weight), ...]."""
    adj = {}
    edge_weight = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        edge_weight[(min(u, v), max(u, v))] = w
    return adj, edge_weight


def find_best_path(edges, node_costs, k):
    """
    Find the lowest-cost simple path of k nodes via branch-and-bound DFS.

    The cost of a path is the sum of all edge weights along it plus all
    node costs. The search prunes any partial path whose cost already
    exceeds the best complete path found so far.

    Args:
        edges: List of (u, v, weight) tuples.
        node_costs: Dict of {node_id: cost}.
        k: Number of qubits in the path.

    Returns:
        (path, cost) — ordered list of physical qubit indices, and total cost.
    """
    adj, edge_weight = build_adjacency(edges)
    all_nodes = sorted(adj.keys())

    best = {"path": None, "cost": float("inf")}

    def _edge_cost(u, v):
        return edge_weight[(min(u, v), max(u, v))]

    def _dfs(path, visited, cost):
        if len(path) == k:
            if cost < best["cost"]:
                best["cost"] = cost
                best["path"] = list(path)
            return

        tail = path[-1]
        for neighbor in adj.get(tail, []):
            if neighbor in visited:
                continue
            new_cost = cost + _edge_cost(tail, neighbor) + node_costs.get(neighbor, 0.0)
            if new_cost >= best["cost"]:
                continue  # prune
            path.append(neighbor)
            visited.add(neighbor)
            _dfs(path, visited, new_cost)
            path.pop()
            visited.discard(neighbor)

    # Start DFS from every node
    for start in all_nodes:
        start_cost = node_costs.get(start, 0.0)
        if start_cost >= best["cost"]:
            continue
        _dfs([start], {start}, start_cost)

    if best["path"] is None:
        raise RuntimeError(f"No connected path of {k} nodes found in the graph.")

    return best["path"], best["cost"]


# ============================================================
# WITNESS EXPERIMENT
# ============================================================

def make_cluster_state_circuit(n_qubits):
    """Create the linear cluster state on n_qubits logical qubits."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2):
        qc.cz(i, i + 1)
    for i in range(1, n_qubits - 1, 2):
        qc.cz(i, i + 1)
    return qc


def compute_stabilizer_prob(counts, n_qubits, target_indices, shots):
    """
    Compute probability that ALL stabilizers at target_indices are satisfied.

    For stabilizer K_i = Z_{i-1} X_i Z_{i+1}, the eigenvalue is +1 when
    the parity of the measured bits at {i-1, i, i+1} is even.
    """
    success = 0
    for bitstring, count in counts.items():
        all_satisfied = True
        for i in target_indices:
            parity = 1
            if bitstring[n_qubits - 1 - i] == "1":
                parity *= -1
            if i > 0 and bitstring[n_qubits - 1 - (i - 1)] == "1":
                parity *= -1
            if i < n_qubits - 1 and bitstring[n_qubits - 1 - (i + 1)] == "1":
                parity *= -1
            if parity == -1:
                all_satisfied = False
                break
        if all_satisfied:
            success += count
    return success / shots


def run_topology_aware_witness(n_qubits, backend, edge_data, node_cost_data,
                                shots=2000, verbose=True):
    """
    Run the cluster state witness on the best physical qubit path.

    Args:
        n_qubits: Number of qubits for the cluster state.
        backend: IQM (or other Qiskit) backend.
        edge_data: List of (u, v, w) hardware edges with two-qubit error cost.
        node_cost_data: List of (node_id, cost) pairs for single-qubit error.
        shots: Measurement shots per circuit.
        verbose: Print progress information.

    Returns:
        Dictionary with witness value, probabilities, path info, and cost.
    """
    # --- Normalize inputs ---
    edges = []
    for item in edge_data:
        if len(item) == 3:
            u, v, w = item
        else:
            (u, v), w = item
            if isinstance(u, list):
                u, v = u[0], u[1]
        if u > v:
            u, v = v, u
        edges.append((u, v, w))

    node_costs = dict(node_cost_data)

    # --- Step 1: Find optimal path ---
    if verbose:
        print(f"Finding optimal {n_qubits}-qubit path...")

    ordered_path, opt_cost = find_best_path(edges, node_costs, n_qubits)

    if verbose:
        print(f"  Path: {ordered_path}")
        print(f"  Cost: {opt_cost:.4f}")

    # --- Step 2: Build coupling map and layout from path ---
    cmap_edges = []
    for i in range(len(ordered_path) - 1):
        cmap_edges.append([ordered_path[i], ordered_path[i + 1]])
        cmap_edges.append([ordered_path[i + 1], ordered_path[i]])

    # Map logical qubit i -> physical qubit ordered_path[i]
    initial_layout = ordered_path

    # --- Step 3: Build circuits ---
    qc_base = make_cluster_state_circuit(n_qubits)

    # Setting A: measure X on even-index qubits, Z on odd-index
    qc_A = qc_base.copy()
    for i in range(0, n_qubits, 2):
        qc_A.h(i)
    qc_A.measure_all()

    # Setting B: measure X on odd-index qubits, Z on even-index
    qc_B = qc_base.copy()
    for i in range(1, n_qubits, 2):
        qc_B.h(i)
    qc_B.measure_all()

    # --- Step 4: Transpile onto the chosen physical path ---
    if verbose:
        print(f"  Transpiling onto physical qubits...")

    cmap = CouplingMap(cmap_edges)
    basis_gates = backend.configuration().basis_gates
    tc_A = transpile(qc_A, basis_gates=basis_gates, coupling_map=cmap,
                     initial_layout=initial_layout, optimization_level=1)
    tc_B = transpile(qc_B, basis_gates=basis_gates, coupling_map=cmap,
                     initial_layout=initial_layout, optimization_level=1)

    if verbose:
        print(f"  Circuit A depth: {tc_A.depth()}, gates: {tc_A.count_ops()}")
        print(f"  Circuit B depth: {tc_B.depth()}, gates: {tc_B.count_ops()}")

    # --- Step 5: Execute ---
    if verbose:
        print(f"  Running witness for N={n_qubits} ({shots} shots per setting)...")

    job_A = backend.run(tc_A, shots=shots)
    job_B = backend.run(tc_B, shots=shots)
    counts_A = job_A.result().get_counts()
    counts_B = job_B.result().get_counts()

    # --- Step 6: Compute witness ---
    prob_A = compute_stabilizer_prob(
        counts_A, n_qubits, range(0, n_qubits, 2), shots
    )
    prob_B = compute_stabilizer_prob(
        counts_B, n_qubits, range(1, n_qubits, 2), shots
    )

    w_value = 3 - 2 * (prob_A + prob_B)

    if verbose:
        status = "GME DETECTED" if w_value < 0 else "no GME"
        print(f"  Result: W = {w_value:.4f} (P_even={prob_A:.3f}, P_odd={prob_B:.3f}) [{status}]")

    return {
        "n_qubits": n_qubits,
        "w_value": w_value,
        "prob_A": prob_A,
        "prob_B": prob_B,
        "gme_detected": w_value < 0,
        "physical_path": ordered_path,
        "optimizer_cost": opt_cost,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    edge_data = [
        ([52, 53], 0.9155800964991245), ([3, 4], 1.8192316747080772),
        ([8, 7], 0.8837752740216809), ([12, 11], 0.9179964548055808),
        ([0, 4], 0.4257549737402777), ([17, 18], 0.2892841978012206),
        ([15, 7], 0.9855673254395625), ([24, 23], 0.7284042308198013),
        ([19, 11], 0.564546975788438), ([21, 13], 2.307193598446955),
        ([28, 27], 0.22167952116319833), ([24, 32], 0.5802886120780393),
        ([33, 34], 1.386738188299319), ([26, 34], 1.298701162486282),
        ([37, 38], 0.4757243476306794), ([28, 36], 0.4290387536496798),
        ([40, 39], 0.3029503978040027), ([44, 43], 0.3891400040439974),
        ([30, 38], 0.15279853480204553), ([47, 48], 0.26309781865180293),
        ([47, 41], 0.4781683165119399), ([5, 4], 0.7060813815400802),
        ([8, 9], 0.4725270605374421), ([43, 49], 7.441027613791384),
        ([8, 2], 0.1789868354317603), ([12, 13], 0.3124558953408352),
        ([15, 14], 0.8528258775608033), ([10, 4], 1.2190286433494846),
        ([15, 23], 0.7590435226697223), ([19, 18], 0.23497714522078272),
        ([17, 25], 1.287840147756203), ([24, 25], 0.42324688082245876),
        ([28, 29], 0.8232818908602946), ([19, 27], 0.6036426716165555),
        ([35, 34], 0.5154203254698619), ([21, 29], 3.828431211490979),
        ([40, 32], 0.8966610275229048), ([40, 41], 0.19168903490524025),
        ([44, 45], 0.9923925197859562), ([5, 6], 0.4204247051326382),
        ([34, 42], 1.507906821331917), ([10, 9], 0.6636227556801599),
        ([44, 36], 0.7065378485403784), ([47, 51], 0.937432189311882),
        ([15, 16], 0.49705311576799804), ([19, 20], 0.31972606602314046),
        ([49, 53], 1.3614456624297167), ([5, 1], 0.7347207561420821),
        ([25, 26], 0.9224362410974796), ([30, 29], 0.6561243117543025),
        ([8, 16], 0.8686489120027563), ([10, 18], 0.1844503641643791),
        ([31, 32], 0.35412621646764286), ([35, 36], 0.31000562740644),
        ([41, 42], 0.3146466929474956), ([12, 20], 1.446927754521643),
        ([49, 50], 0.8950663667080749), ([52, 51], 1.3691388044951625),
        ([31, 23], 0.6084517668385026), ([33, 25], 0.47185893050541994),
        ([35, 27], 1.105280334830916), ([40, 46], 0.8937829418373955),
        ([44, 50], 0.0857038840365365), ([3, 9], 0.5735334860861974),
        ([5, 11], 0.46191574895697585), ([22, 14], 0.21275771631790175),
        ([24, 16], 0.6479119798293964), ([18, 26], 1.060549061816396),
        ([28, 20], 3.992889730025484), ([33, 41], 0.6790866218461589),
        ([35, 43], 0.5091354090133837), ([52, 48], 0.19723307076128238),
        ([0, 1], 0.2323351364436177), ([3, 2], 1.2249253218507228),
        ([10, 11], 0.22812821171529496), ([17, 16], 0.6612857561426821),
        ([21, 20], 1.7297983852624599), ([22, 23], 0.33762959353725863),
        ([26, 27], 3.1258999408530386), ([33, 32], 0.2460590329208978),
        ([36, 37], 0.48785240074543657), ([47, 46], 0.3287896774973653),
    ]

    node_cost_data = [
        (35, 0.18267244875723732), (37, 0.04541987667315839),
        (40, 0.03822234982263639), (42, 0.04553586473223792),
        (44, 0.23431868347043583), (47, 0.04665852358756162),
        (49, 0.07935963501167764), (52, 0.034417561645139205),
        (1, 0.0788049208323427), (2, 0.053757367084994456),
        (4, 0.04721196180086373), (6, 0.07360398141972313),
        (7, 0.15154100006348203), (9, 0.03612702969671977),
        (11, 0.04058430605967667), (13, 0.04025262727723744),
        (14, 0.06798198252165877), (16, 0.11711812426692392),
        (18, 0.041991952865583215), (20, 0.04873246537964171),
        (23, 0.0480942748866231), (25, 0.04846653920176358),
        (27, 0.2621557317233236), (29, 0.12508807110688247),
        (32, 0.05647806499021746), (34, 0.08476795654820402),
        (0, 0.09865281632336353), (36, 0.07093776176767541),
        (38, 0.04752279065339815), (3, 0.14383451657742352),
        (39, 0.03219279302524303), (5, 0.04689576637412207),
        (8, 0.17297858265346022), (41, 0.0291886643905781),
        (43, 0.39730082924456234), (10, 0.07724239810303901),
        (45, 0.09456159477804205), (12, 0.05924875957360465),
        (15, 0.15080242806093658), (46, 0.04061319272595787),
        (17, 0.042576140728378675), (19, 0.04087415711732101),
        (48, 0.04049756041075714), (50, 0.07428052861675472),
        (21, 0.05052102481531984), (51, 0.0510298998178782),
        (22, 0.03857353950770248), (53, 0.2714952499649592),
        (24, 0.057149679710577495), (26, 0.17811084900063445),
        (28, 0.08568695199644427), (30, 0.06601638446883662),
        (31, 0.05863076644652132), (33, 0.04646413145059647),
    ]

    # --- Connect to backend ---
    provider = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer="garnet",
        token="kXL7TYp+aF382y0PoH+iJ9bfYPCbhwDt8fZCu7KHoaMBnBezagx+Q5zUHT1QCtkp",
    )
    backend = provider.get_backend()

    # --- Run ---
    print("=" * 60)
    print("TOPOLOGY-AWARE LINEAR CLUSTER STATE WITNESS")
    print("=" * 60)

    for n in [15]:
        result = run_topology_aware_witness(
            n_qubits=n,
            backend=backend,
            edge_data=edge_data,
            node_cost_data=node_cost_data,
            shots=10000,
        )
        print(f"\nN={n}: W = {result['w_value']:.3f} "
              f"(P_even={result['prob_A']:.2f}, P_odd={result['prob_B']:.2f})")
        print(f"  Path: {result['physical_path']}")
        print(f"  Cost: {result['optimizer_cost']:.4f}")
        print(f"  GME: {'YES' if result['gme_detected'] else 'no'}")
