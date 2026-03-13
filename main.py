import os
import glob
import csv
import time
import matplotlib.pyplot as plt
from data_parser import parse_instance
from lp_model import build_lp_model
from subgradient_solver import subgradient_optimization
import pulp

def analyze_gap(instance_path, max_iter=150):
    print(f"--- Analyzing {os.path.basename(instance_path)} ---")
    num_vehicles = 5
    
    # 1. LP Relaxation
    start = time.time()
    lp_model = build_lp_model(instance_path, num_vehicles=num_vehicles)
    lp_model.solve(pulp.PULP_CBC_CMD(msg=False))
    lp_time = time.time() - start
    lp_bound = pulp.value(lp_model.objective) if lp_model.status == pulp.LpStatusOptimal else None
    print(f"LP Bound: {lp_bound:.2f} (Time: {lp_time:.2f}s)")
    
    # 2. Lagrangian Relaxation via Subgradient Optimization
    start = time.time()
    lr_bound = subgradient_optimization(instance_path, num_vehicles=num_vehicles, max_iter=max_iter, log=False)
    lr_time = time.time() - start
    print(f"LR Bound: {lr_bound:.2f} (Time: {lr_time:.2f}s)")
    
    # Return metrics
    return {
        'Instance': os.path.basename(instance_path),
        'LP_Bound': round(lp_bound, 2) if lp_bound else None,
        'LR_Bound': round(lr_bound, 2) if lr_bound else None,
        'LP_Time_s': round(lp_time, 2),
        'LR_Time_s': round(lr_time, 2)
    }

def run_analysis():
    # Gather all small instances to test
    instances = glob.glob('evrptw_instances/*C5.txt') + glob.glob('evrptw_instances/*C10.txt')
    instances = sorted(instances)
    
    results = []
    for inst in instances:
        try:
            res = analyze_gap(inst, max_iter=100)
            results.append(res)
        except Exception as e:
            print(f"Failed on {inst}: {e}")
            
    if results:
        # Calculate duality gap between LR and LP
        # Gap = (LR - LP) / LP * 100
        for r in results:
            if r['LP_Bound'] and r['LR_Bound']:
                r['Gap_%'] = round((r['LR_Bound'] - r['LP_Bound']) / r['LP_Bound'] * 100, 2)
            else:
                r['Gap_%'] = None
                
        keys = results[0].keys()
        with open('duality_gap_analysis.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            
        print("\n--- Analysis Complete ---")
        for r in results:
            print(f"{r['Instance']}: LP={r['LP_Bound']}, LR={r['LR_Bound']}, Gap={r['Gap_%']}%")
            
        # Generate plot
        valid_results = [r for r in results if r['LP_Bound'] is not None and r['LR_Bound'] is not None]
        if valid_results:
            labels = [r['Instance'].replace('.txt', '') for r in valid_results]
            lp_vals = [r['LP_Bound'] for r in valid_results]
            lr_vals = [r['LR_Bound'] for r in valid_results]
            
            x = range(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar([pos - width/2 for pos in x], lp_vals, width, label='LP Bound', color='skyblue')
            rects2 = ax.bar([pos + width/2 for pos in x], lr_vals, width, label='LR Bound', color='salmon')
            
            ax.set_ylabel('Objective Value (Travel Time/Distance)')
            ax.set_title('Comparison of LP vs LR Bounding across Small Instances')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            
            fig.tight_layout()
            plt.savefig("duality_gap_plot.png", dpi=300)
            print("Generated duality_gap_plot.png")

    
if __name__ == "__main__":
    run_analysis()
