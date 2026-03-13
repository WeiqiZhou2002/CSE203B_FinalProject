import re
import math

def parse_instance(filepath):
    """
    Parses an E-VRPTW instance file and returns a dictionary with the data.
    """
    nodes = []
    ev_params = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    parsing_nodes = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("StringID"):
            parsing_nodes = True
            continue
            
        if parsing_nodes:
            parts = line.split()
            if len(parts) >= 8 and parts[1] in ['d', 'f', 'c']:
                node = {
                    'id': parts[0],
                    'type': parts[1],
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'demand': float(parts[4]),
                    'ready_time': float(parts[5]),
                    'due_date': float(parts[6]),
                    'service_time': float(parts[7])
                }
                nodes.append(node)
            else:
                parsing_nodes = False # Reached the end of nodes block
                
        # Parse EV params using regex to find content between slashes
        if not parsing_nodes:
            match = re.search(r'/(.*?)/', line)
            if match:
                val = float(match.group(1))
                if line.startswith("Q"):
                    ev_params['Q'] = val
                elif line.startswith("C"):
                    ev_params['C'] = val
                elif line.startswith("r"):
                    ev_params['r'] = val
                elif line.startswith("g"):
                    ev_params['g'] = val
                elif line.startswith("v"):
                    ev_params['v'] = val

    # Calculate distance matrix (Euclidean)
    n = len(nodes)
    dist_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            dist_matrix[i][j] = math.sqrt(dx**2 + dy**2)
            
    return {
        'nodes': nodes,
        'ev_params': ev_params,
        'dist_matrix': dist_matrix
    }

if __name__ == "__main__":
    # Test on a small instance
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "evrptw_instances/c101C5.txt"
    
    data = parse_instance(filepath)
    print(f"Parsed {len(data['nodes'])} nodes.")
    print(f"EV Params: {data['ev_params']}")
    print(f"Distance between node 0 and 1: {data['dist_matrix'][0][1]:.2f}")
