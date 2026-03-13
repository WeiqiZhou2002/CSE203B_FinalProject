from lagrangian_model import solve_lagrangian_subproblem
from data_parser import parse_instance

def subgradient_optimization(filepath, num_vehicles=5, max_iter=200, UB=2000.0, log=True):
    data = parse_instance(filepath)
    nodes = data['nodes']
    dist = data['dist_matrix']
    ev_params = data['ev_params']
    n = len(nodes)
    
    F = [i for i, node in enumerate(nodes) if node['type'] == 'f']
    D = [i for i, node in enumerate(nodes) if node['type'] == 'd']
    depot = D[0]
    
    M_time = max(node['due_date'] for node in nodes) + max(max(r) for r in dist)
    M_Q = ev_params['Q']
    M_C = ev_params['C']
    
    lam_T = [[0.0]*n for _ in range(n)]
    lam_L = [[0.0]*n for _ in range(n)]
    lam_B = [[0.0]*n for _ in range(n)]
    
    best_LB = -float('inf')
    alpha = 2.0
    no_improve_count = 0
    
    for iteration in range(max_iter):
        LB, x, tau, u, y = solve_lagrangian_subproblem(filepath, (lam_T, lam_L, lam_B), num_vehicles)
        
        if LB > best_LB + 1e-4:
            best_LB = LB
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 10:
                alpha /= 2.0
                no_improve_count = 0
                
        if log and iteration % 10 == 0:
            print(f"Iter {iteration}: LB = {LB:.2f}, Best LB = {best_LB:.2f}, Alpha = {alpha:.4f}")
            
        norm_sq = 0.0
        viol_T = [[0.0]*n for _ in range(n)]
        viol_L = [[0.0]*n for _ in range(n)]
        viol_B = [[0.0]*n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    t_ij = dist[i][j] / ev_params['v']
                    s_i = nodes[i]['service_time']
                    vT = tau[i] - tau[j] + s_i + t_ij - M_time * (1 - x[i][j])
                    viol_T[i][j] = vT
                    if lam_T[i][j] > 0 or vT > 0: norm_sq += vT**2
                    
                    vL = u[j] - u[i] + nodes[i]['demand'] - M_C * (1 - x[i][j])
                    viol_L[i][j] = vL
                    if lam_L[i][j] > 0 or vL > 0: norm_sq += vL**2
                    
                    if j not in F and j != depot:
                        energy_ij = dist[i][j] * ev_params['r']
                        vB = y[j] - y[i] + energy_ij - M_Q * (1 - x[i][j])
                        viol_B[i][j] = vB
                        if lam_B[i][j] > 0 or vB > 0: norm_sq += vB**2

        if norm_sq < 1e-6:
            if log:
                print("Optimal Lagrangian bound reached (subgradient zero).")
            break
            
        step_size = alpha * max(0.0, UB - LB) / norm_sq
        
        # Update and project multipliers
        for i in range(n):
            for j in range(n):
                lam_T[i][j] = max(0.0, lam_T[i][j] + step_size * viol_T[i][j])
                lam_L[i][j] = max(0.0, lam_L[i][j] + step_size * viol_L[i][j])
                lam_B[i][j] = max(0.0, lam_B[i][j] + step_size * viol_B[i][j])
                
        if alpha < 1e-5:
            if log:
                print("Alpha too small, terminating.")
            break

    if log:
        print(f"Subgradient Optimization finished. Best LB = {best_LB:.2f}")
    return best_LB

if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'evrptw_instances/c101C5.txt'
    subgradient_optimization(filepath, max_iter=100)
