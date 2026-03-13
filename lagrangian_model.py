import pulp
from data_parser import parse_instance

def solve_lagrangian_subproblem(filepath, multipliers, num_vehicles=5):
    """
    Solves the Lagrangian subproblem where time window and capacity constraints 
    are relaxed and penalized in the objective function.
    
    multipliers: tuple of (lambda_T, lambda_L, lambda_B), each a 2D list of shape n x n.
    """
    data = parse_instance(filepath)
    nodes = data['nodes']
    dist = data['dist_matrix']
    ev_params = data['ev_params']
    
    lam_T, lam_L, lam_B = multipliers
    n = len(nodes)
    
    C = [i for i, node in enumerate(nodes) if node['type'] == 'c']
    F = [i for i, node in enumerate(nodes) if node['type'] == 'f']
    D = [i for i, node in enumerate(nodes) if node['type'] == 'd']
    depot = D[0]
    
    M_time = max(node['due_date'] for node in nodes) + max(max(r) for r in dist)
    M_Q = ev_params['Q']
    M_C = ev_params['C']
    
    prob = pulp.LpProblem("EVRPTW_Lagrangian_Subproblem", pulp.LpMinimize)
    
    # We can explicitly solve this as an LP because the remaining constraints 
    # (flow/assignment) form a totally unimodular matrix, meaning x_ij will naturally be {0,1}
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n)), lowBound=0.0, upBound=1.0, cat='Continuous')
    tau = pulp.LpVariable.dicts("tau", (i for i in range(n)), lowBound=0.0, cat='Continuous')
    u = pulp.LpVariable.dicts("u", (i for i in range(n)), lowBound=0.0, upBound=ev_params['C'], cat='Continuous')
    y = pulp.LpVariable.dicts("y", (i for i in range(n)), lowBound=0.0, upBound=ev_params['Q'], cat='Continuous')
    
    # Subproblem Objective
    obj_terms = []
    
    # 1. Original travel distance
    for i in range(n):
        for j in range(n):
            if i != j:
                obj_terms.append(dist[i][j] * x[i, j])
                
    # 2. Relaxed Time Windows: tau_j >= tau_i + s_i + t_ij - M(1 - x_ij)
    # => tau_i - tau_j + s_i + t_ij - M + M x_ij <= 0
    for i in range(n):
        for j in range(n):
            if i != j:
                t_ij = dist[i][j] / ev_params['v']
                s_i = nodes[i]['service_time']
                viol_T = tau[i] - tau[j] + s_i + t_ij - M_time * (1 - x[i, j])
                obj_terms.append(lam_T[i][j] * viol_T)
                
    # 3. Relaxed Load constraints: u_j <= u_i - d_i + M(1 - x_ij)
    # => u_j - u_i + d_i - M + M x_ij <= 0
    for i in range(n):
        for j in range(n):
            if i != j:
                viol_L = u[j] - u[i] + nodes[i]['demand'] - M_C * (1 - x[i, j])
                obj_terms.append(lam_L[i][j] * viol_L)
                
    # 4. Relaxed Battery constraints: y_j <= y_i - energy_ij + M(1 - x_ij)
    # => y_j - y_i + energy_ij - M + M x_ij <= 0 (skip if j is charging station/depot)
    for i in range(n):
        for j in range(n):
            if i != j:
                energy_ij = dist[i][j] * ev_params['r']
                if j not in F and j != depot:
                    viol_B = y[j] - y[i] + energy_ij - M_Q * (1 - x[i, j])
                    obj_terms.append(lam_B[i][j] * viol_B)
                    
    prob += pulp.lpSum(obj_terms)
    
    # Rest of constraints (Hard Linking)
    for j in C:
        prob += pulp.lpSum(x[i, j] for i in range(n) if i != j) == 1
        prob += pulp.lpSum(x[j, i] for i in range(n) if i != j) == 1
        
    for j in F:
        prob += pulp.lpSum(x[i, j] for i in range(n) if i != j) == pulp.lpSum(x[j, i] for i in range(n) if i != j)
        
    prob += pulp.lpSum(x[depot, j] for j in range(n) if j != depot) <= num_vehicles
    prob += pulp.lpSum(x[j, depot] for j in range(n) if j != depot) <= num_vehicles
    prob += pulp.lpSum(x[depot, j] for j in range(n) if j != depot) == pulp.lpSum(x[j, depot] for j in range(n) if j != depot)
    
    for i in range(n):
        prob += x[i, i] == 0
        
    # Bounds for decoupled variables
    for i in range(n):
        prob += tau[i] >= nodes[i]['ready_time']
        prob += tau[i] <= nodes[i]['due_date']
        
    for i in range(n):
        if i in F or i == depot:
            prob += y[i] == ev_params['Q']
            
    # Suppress output and solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Need to return: objective value (lower bound), variable values
    def safe_val(var, default):
        v = pulp.value(var)
        return v if v is not None else default

    x_vals = [[safe_val(x[i, j], 0.0) for j in range(n)] for i in range(n)]
    tau_vals = [safe_val(tau[i], nodes[i]['ready_time']) for i in range(n)]
    u_vals = [safe_val(u[i], 0.0) for i in range(n)]
    y_vals = [safe_val(y[i], ev_params['Q']) for i in range(n)]
    
    return pulp.value(prob.objective), x_vals, tau_vals, u_vals, y_vals

if __name__ == "__main__":
    filepath = "evrptw_instances/c101C5.txt"
    data = parse_instance(filepath)
    n = len(data['nodes'])
    # zero multipliers
    zeros = [[0.0]*n for _ in range(n)]
    obj, x_v, t_v, u_v, y_v = solve_lagrangian_subproblem(filepath, (zeros, zeros, zeros))
    print(f"Lagrangian Subproblem Bound (all lam=0): {obj:.2f}")
