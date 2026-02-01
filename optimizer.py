from ortools.linear_solver import pywraplp

def solve_k_node_tree(num_nodes, edges, node_costs, k, time_limit_sec=60):
    """
    num_nodes: int (54)
    node_costs: list/array of length num_nodes (c_v)
    edges: list of (u, v, w) undirected with 0<=u<v<num_nodes
    k: number of nodes to select
    Returns: (selected_nodes, selected_edges, objective_value)
    """

    if not (1 <= k <= num_nodes):
        raise ValueError("k must be between 1 and num_nodes")
    if len(node_costs) != num_nodes:
        raise ValueError("node_costs must have length num_nodes")

    # Try SCIP, fall back to CBC
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No MILP solver available (SCIP/CBC).")

    solver.SetTimeLimit(int(time_limit_sec * 1000))

    V = range(num_nodes)
    E = range(len(edges))

    # Decision vars
    x = {v: solver.IntVar(0, 1, f"x[{v}]") for v in V}      # select node
    r = {v: solver.IntVar(0, 1, f"r[{v}]") for v in V}      # root
    y = {ei: solver.IntVar(0, 1, f"y[{ei}]") for ei in E}   # select edge

    # Flow vars on directed arcs
    f = {}
    for ei, (u, v, w) in enumerate(edges):
        f[(ei, u, v)] = solver.NumVar(0.0, k - 1, f"f[{u}->{v}]")
        f[(ei, v, u)] = solver.NumVar(0.0, k - 1, f"f[{v}->{u}]")

    # 1) Exactly k nodes
    solver.Add(sum(x[v] for v in V) == k)

    # 2) Exactly one root, root must be selected
    solver.Add(sum(r[v] for v in V) == 1)
    for v in V:
        solver.Add(r[v] <= x[v])

    # 3) Edges only between chosen nodes
    for ei, (u, v, w) in enumerate(edges):
        solver.Add(y[ei] <= x[u])
        solver.Add(y[ei] <= x[v])

    # 4) Capacity on flow: only through selected edges
    for ei, (u, v, w) in enumerate(edges):
        solver.Add(f[(ei, u, v)] <= (k - 1) * y[ei])
        solver.Add(f[(ei, v, u)] <= (k - 1) * y[ei])

    # Build in/out flow lists per node
    out_arcs = {v: [] for v in V}
    in_arcs  = {v: [] for v in V}
    for ei, (u, v, w) in enumerate(edges):
        out_arcs[u].append(f[(ei, u, v)])
        in_arcs[v].append(f[(ei, u, v)])
        out_arcs[v].append(f[(ei, v, u)])
        in_arcs[u].append(f[(ei, v, u)])

    # 5) Flow conservation with variable root
    for v in V:
        solver.Add(sum(out_arcs[v]) - sum(in_arcs[v]) == (k - 1) * r[v] - x[v] + r[v])

    # 6) Tree edge count (recommended if edge weights are nonnegative)
    solver.Add(sum(y[ei] for ei in E) == k - 1)

    # 7) Force path topology: every selected node has at most 2 tree edges
    #for v in V:
    #   incident = [y[ei] for ei, (u, w_, wt) in enumerate(edges) if u == v or w_ == v]
    #   if incident:
    #       solver.Add(sum(incident) <= 2)

    # Objective: node costs + edge costs
    solver.Minimize(
        sum(node_costs[v] * x[v] for v in V) +
        sum(edges[ei][2] * y[ei] for ei in E)
    )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL):
        raise RuntimeError("No feasible solution (graph may not have a connected k-node subgraph).")

    selected_nodes = [v for v in V if x[v].solution_value() > 0.5]
    selected_edges = []
    for ei, (u, v, w) in enumerate(edges):
        if y[ei].solution_value() > 0.5:
            selected_edges.append((u, v, w))

    return selected_nodes, selected_edges, solver.Objective().Value()

edge_data = [([52, 53], 0.9155800964991245), ([3, 4], 1.8192316747080772), ([8, 7], 0.8837752740216809), ([12, 11], 0.9179964548055808), ([0, 4], 0.4257549737402777), ([17, 18], 0.2892841978012206), ([15, 7], 0.9855673254395625), ([24, 23], 0.7284042308198013), ([19, 11], 0.564546975788438), ([21, 13], 2.307193598446955), ([28, 27], 0.22167952116319833), ([24, 32], 0.5802886120780393), ([33, 34], 1.386738188299319), ([26, 34], 1.298701162486282), ([37, 38], 0.4757243476306794), ([28, 36], 0.4290387536496798), ([40, 39], 0.3029503978040027), ([44, 43], 0.3891400040439974), ([30, 38], 0.15279853480204553), ([47, 48], 0.26309781865180293), ([47, 41], 0.4781683165119399), ([5, 4], 0.7060813815400802), ([8, 9], 0.4725270605374421), ([43, 49], 7.441027613791384), ([8, 2], 0.1789868354317603), ([12, 13], 0.3124558953408352), ([15, 14], 0.8528258775608033), ([10, 4], 1.2190286433494846), ([15, 23], 0.7590435226697223), ([19, 18], 0.23497714522078272), ([17, 25], 1.287840147756203), ([24, 25], 0.42324688082245876), ([28, 29], 0.8232818908602946), ([19, 27], 0.6036426716165555), ([35, 34], 0.5154203254698619), ([21, 29], 3.828431211490979), ([40, 32], 0.8966610275229048), ([40, 41], 0.19168903490524025), ([44, 45], 0.9923925197859562), ([5, 6], 0.4204247051326382), ([34, 42], 1.507906821331917), ([10, 9], 0.6636227556801599), ([44, 36], 0.7065378485403784), ([47, 51], 0.937432189311882), ([15, 16], 0.49705311576799804), ([19, 20], 0.31972606602314046), ([49, 53], 1.3614456624297167), ([5, 1], 0.7347207561420821), ([25, 26], 0.9224362410974796), ([30, 29], 0.6561243117543025), ([8, 16], 0.8686489120027563), ([10, 18], 0.1844503641643791), ([31, 32], 0.35412621646764286), ([35, 36], 0.31000562740644), ([41, 42], 0.3146466929474956), ([12, 20], 1.446927754521643), ([49, 50], 0.8950663667080749), ([52, 51], 1.3691388044951625), ([31, 23], 0.6084517668385026), ([33, 25], 0.47185893050541994), ([35, 27], 1.105280334830916), ([40, 46], 0.8937829418373955), ([44, 50], 0.0857038840365365), ([3, 9], 0.5735334860861974), ([5, 11], 0.46191574895697585), ([22, 14], 0.21275771631790175), ([24, 16], 0.6479119798293964), ([18, 26], 1.060549061816396), ([28, 20], 3.992889730025484), ([33, 41], 0.6790866218461589), ([35, 43], 0.5091354090133837), ([52, 48], 0.19723307076128238), ([0, 1], 0.2323351364436177), ([3, 2], 1.2249253218507228), ([10, 11], 0.22812821171529496), ([17, 16], 0.6612857561426821), ([21, 20], 1.7297983852624599), ([22, 23], 0.33762959353725863), ([26, 27], 3.1258999408530386), ([33, 32], 0.2460590329208978), ([36, 37], 0.48785240074543657), ([47, 46], 0.3287896774973653)]
edge_data = [(u, v, w) for ([u, v], w) in edge_data]
edge_data = [(u, v, w) if u < v else (v, u, w) for (u, v, w) in edge_data]

node_cost = [(35, 0.18267244875723732), (37, 0.04541987667315839), (40, 0.03822234982263639), (42, 0.04553586473223792), (44, 0.23431868347043583), (47, 0.04665852358756162), (49, 0.07935963501167764), (52, 0.034417561645139205), (1, 0.0788049208323427), (2, 0.053757367084994456), (4, 0.04721196180086373), (6, 0.07360398141972313), (7, 0.15154100006348203), (9, 0.03612702969671977), (11, 0.04058430605967667), (13, 0.04025262727723744), (14, 0.06798198252165877), (16, 0.11711812426692392), (18, 0.041991952865583215), (20, 0.04873246537964171), (23, 0.0480942748866231), (25, 0.04846653920176358), (27, 0.2621557317233236), (29, 0.12508807110688247), (32, 0.05647806499021746), (34, 0.08476795654820402), (0, 0.09865281632336353), (36, 0.07093776176767541), (38, 0.04752279065339815), (3, 0.14383451657742352), (39, 0.03219279302524303), (5, 0.04689576637412207), (8, 0.17297858265346022), (41, 0.0291886643905781), (43, 0.39730082924456234), (10, 0.07724239810303901), (45, 0.09456159477804205), (12, 0.05924875957360465), (15, 0.15080242806093658), (46, 0.04061319272595787), (17, 0.042576140728378675), (19, 0.04087415711732101), (48, 0.04049756041075714), (50, 0.07428052861675472), (21, 0.05052102481531984), (51, 0.0510298998178782), (22, 0.03857353950770248), (53, 0.2714952499649592), (24, 0.057149679710577495), (26, 0.17811084900063445), (28, 0.08568695199644427), (30, 0.06601638446883662), (31, 0.05863076644652132), (33, 0.04646413145059647)]
node_cost = [i for (_, i) in sorted(node_cost)]

_, edges, cost = solve_k_node_tree(54, edge_data, node_cost, 15, 60)
edges = [[u, v] for (u, v, _) in edges]
print(edges)
print(len(edges))
print(cost)