from ortools.sat.python import cp_model

def optimize_linear_cluster_path(
    num_phys,
    undirected_edges,   # list of (u, v, w) with u<v
    node_costs,         # list length num_phys
    n_qubits,           # logical qubits in your cluster line
    time_limit_sec=30,
    scale=1_000_000,
    num_workers=8,
):
    """
    Returns: (path, path_edges, objective_float)
      path: [p0, p1, ..., p_{n-1}] physical qubit indices
      path_edges: [(p0,p1,w01), (p1,p2,w12), ...] with weights from input
    """
    if n_qubits < 1:
        return [], [], 0.0
    if n_qubits > num_phys:
        raise ValueError("n_qubits cannot exceed num_phys")
    if len(node_costs) != num_phys:
        raise ValueError("node_costs length mismatch")

    # Build directed arc list from undirected couplers
    # Each arc is (u, v, w)
    arcs = []
    w_lookup = {}
    for u, v, w in undirected_edges:
        if u == v:
            continue
        if u > v:
            u, v = v, u
        w_lookup[(u, v)] = w
        arcs.append((u, v, w))
        arcs.append((v, u, w))

    if n_qubits == 1:
        # Just pick cheapest node
        best = min(range(num_phys), key=lambda p: node_costs[p])
        return [best], [], node_costs[best]

    model = cp_model.CpModel()

    # x[t,p] = 1 iff position t uses physical qubit p
    x = {}
    for t in range(n_qubits):
        for p in range(num_phys):
            x[(t, p)] = model.NewBoolVar(f"x[{t},{p}]")

    # Each position picks exactly one physical qubit
    for t in range(n_qubits):
        model.Add(sum(x[(t, p)] for p in range(num_phys)) == 1)

    # No physical qubit used twice
    for p in range(num_phys):
        model.Add(sum(x[(t, p)] for t in range(n_qubits)) <= 1)

    # For each step t -> t+1, select exactly one directed arc (u->v)
    z = {}  # z[t,a] = arc chosen at step t
    for t in range(n_qubits - 1):
        for a, (u, v, w) in enumerate(arcs):
            z[(t, a)] = model.NewBoolVar(f"z[{t},{a}]")

        model.Add(sum(z[(t, a)] for a in range(len(arcs))) == 1)

        # Link arc endpoints to chosen physical qubits at positions t and t+1
        for a, (u, v, w) in enumerate(arcs):
            model.Add(z[(t, a)] <= x[(t, u)])
            model.Add(z[(t, a)] <= x[(t + 1, v)])

    # Objective: node + edge costs (scaled to integers)
    node_cost_int = [int(round(scale * c)) for c in node_costs]
    arc_cost_int = [int(round(scale * w)) for (_, _, w) in arcs]

    obj_terms = []
    # node cost
    for t in range(n_qubits):
        for p in range(num_phys):
            obj_terms.append(node_cost_int[p] * x[(t, p)])
    # edge cost
    for t in range(n_qubits - 1):
        for a in range(len(arcs)):
            obj_terms.append(arc_cost_int[a] * z[(t, a)])

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(num_workers)
    # solver.parameters.log_search_progress = True  # uncomment to debug

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible embedding: cannot place a length-n path on this coupling graph.")

    # Extract physical path
    path = []
    for t in range(n_qubits):
        chosen_p = None
        for p in range(num_phys):
            if solver.Value(x[(t, p)]) == 1:
                chosen_p = p
                break
        path.append(chosen_p)

    # Build path edges with original weights
    path_edges = []
    for t in range(n_qubits - 1):
        u, v = path[t], path[t + 1]
        key = (u, v) if u < v else (v, u)
        w = w_lookup.get(key, None)
        path_edges.append((u, v, w))

    obj_float = solver.ObjectiveValue() / scale
    return path, path_edges, obj_float