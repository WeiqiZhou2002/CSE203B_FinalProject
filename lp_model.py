import pulp
from data_parser import parse_instance

def build_lp_model(filepath, num_vehicles=5):
    """
    Builds the standard LP Relaxation of the E-VRPTW model.
    """
    data = parse_instance(filepath)
    nodes = data['nodes']
    dist = data['dist_matrix']
    ev_params = data['ev_params']
    
    n = len(nodes)
    
    # Identify sets
    # Assuming node 0 is the depot
    # we duplicate depot for start and end, but for simplicity, let's treat it as node 0 (start) and node n (end, same coords).
    # Since nodes are given in the file, we can simply enforce flows starting and ending at node 0.
    C = [i for i, node in enumerate(nodes) if node['type'] == 'c']
    F = [i for i, node in enumerate(nodes) if node['type'] == 'f']
    D = [i for i, node in enumerate(nodes) if node['type'] == 'd']
    depot = D[0]
    
    # Big-M constants
    M_time = max(node['due_date'] for node in nodes) + max(max(r) for r in dist)
    M_Q = ev_params['Q']
    M_C = ev_params['C']
    
    # Initialise PuLP problem
    # Minimizing the distance
    prob = pulp.LpProblem("EVRPTW_Relaxation", pulp.LpMinimize)
    
    # Decisions Variables
    # x[i][j] is the flow from node i to node j. Relaxed to Continuous [0, 1]
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n)), lowBound=0.0, upBound=1.0, cat='Continuous')
    
    # Time variables: arrival time at node i
    tau = pulp.LpVariable.dicts("tau", (i for i in range(n)), lowBound=0.0, cat='Continuous')
    
    # Load variables: remaining load upon departure from node i
    u = pulp.LpVariable.dicts("u", (i for i in range(n)), lowBound=0.0, upBound=ev_params['C'], cat='Continuous')
    
    # Battery variables: remaining battery upon departure from node i
    y = pulp.LpVariable.dicts("y", (i for i in range(n)), lowBound=0.0, upBound=ev_params['Q'], cat='Continuous')
    
    # Objective: Minimize total traveled distance
    prob += pulp.lpSum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
    
    # Constraints
    # 1. Each customer is visited exactly once
    for j in C:
        prob += pulp.lpSum(x[i, j] for i in range(n) if i != j) == 1, f"Visit_C_{j}"
        prob += pulp.lpSum(x[j, i] for i in range(n) if i != j) == 1, f"Leave_C_{j}"
        
    # 2. Flow conservation for charging stations (can be visited 0 or multiple times, but let's restrict to bounded flow)
    for j in F:
        prob += pulp.lpSum(x[i, j] for i in range(n) if i != j) == pulp.lpSum(x[j, i] for i in range(n) if i != j), f"Flow_F_{j}"
        
    # 3. Limit vehicles returning/leaving depot
    prob += pulp.lpSum(x[depot, j] for j in range(n) if j != depot) <= num_vehicles, "Max_Vehicles_Leave"
    prob += pulp.lpSum(x[j, depot] for j in range(n) if j != depot) <= num_vehicles, "Max_Vehicles_Return"
    prob += pulp.lpSum(x[depot, j] for j in range(n) if j != depot) == pulp.lpSum(x[j, depot] for j in range(n) if j != depot), "Vehicles_Flow_Depot"
    
    # No self loops
    for i in range(n):
        prob += x[i, i] == 0, f"No_Self_{i}"
        
    # 4. Time windows & Service time compatibility (Simplified formulation)
    # tau_j >= tau_i + service_i + travel_ij - M(1 - x_ij)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Travel time = distance / velocity
                t_ij = dist[i][j] / ev_params['v']
                # for charging stations we just use the fixed service time defined in instance if it's 0 then 0, 
                # a proper formulation adds charge amount * g. To keep linear relaxation simple we ignore variable recharge time.
                s_i = nodes[i]['service_time']
                prob += tau[j] >= tau[i] + s_i + t_ij - M_time * (1 - x[i, j]), f"Time_{i}_{j}"
                
    # Time window limits
    for i in range(n):
        prob += tau[i] >= nodes[i]['ready_time'], f"TW_lower_{i}"
        prob += tau[i] <= nodes[i]['due_date'], f"TW_upper_{i}"
        
    # 5. Load considerations
    for i in range(n):
        for j in range(n):
            if i != j:
                prob += u[j] <= u[i] - nodes[i]['demand'] + M_C * (1 - x[i, j]), f"Load_{i}_{j}"
                
    # 6. Battery state-of-charge considerations
    for i in range(n):
        for j in range(n):
            if i != j:
                # battery drop = distance * r
                energy_ij = dist[i][j] * ev_params['r']
                if j in F or j == depot:
                    # if arriving at charging station or depot, it gets recharged to Q upon departure
                    pass
                else:
                    prob += y[j] <= y[i] - energy_ij + M_Q * (1 - x[i, j]), f"Battery_{i}_{j}"
                    
    for i in range(n):
        if i in F or i == depot:
            prob += y[i] == ev_params['Q'], f"Battery_Full_{i}"
            
    return prob

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "evrptw_instances/c101C5.txt"
    model = build_lp_model(filepath)
    print("Model built with:")
    print(f"  {len(model.variables())} variables")
    print(f"  {len(model.constraints)} constraints")
    
    print("Solving relaxation...")
    model.solve()
    print(f"Status: {pulp.LpStatus[model.status]}")
    print(f"Objective (Relaxed Lower Bound): {pulp.value(model.objective):.2f}")
