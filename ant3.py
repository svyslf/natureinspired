

import cProfile
from io import StringIO
import os
import pprint
import pstats

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import digraph
import random
import matplotlib.font_manager


plt.rcParams["font.family"] = "monospace"

bp1_items = [i for i in range(1, 501)]
bp2_items = [(i * i) / 2 for i in range(1, 501)]

bp1_configurations = {
    "BP1, 100 Paths, 0.6 evaporation": [bp1_items, 100, 0.6],
    "BP1, 100 Paths, 0.9 evaporation": [bp1_items, 100, 0.9],
    "BP1, 10 Paths, 0.6 evaporation": [bp1_items, 10, 0.6],
    "BP1, 10 Paths, 0.9 evaporation": [bp1_items, 10, 0.9],
}
bp2_configurations = {
    "BP2, 100 Paths, 0.6 evaporation": [bp2_items, 100, 0.6],
    "BP2, 100 Paths, 0.9 evaporation": [bp2_items, 100, 0.9],
    "BP2, 10 Paths, 0.6 evaporation": [bp2_items, 10, 0.6],
    "BP2, 10 Paths, 0.9 evaporation": [bp2_items, 10, 0.9],
}



class ACO:
    def __init__(self, graph, p, e, config):
        self.graph = graph
        self.paths = p
        self.evaporation_rate = e
        self.best_fitness = float("inf")
        self.config = config

    def pick_next(self, current_node):
        #THIS function is really bad - but i'm sure something im doing here is stupid. 

        # Get all edges from the current node with their weights
        edges_data = self.graph.graph[current_node]  # Access outgoing edges from the current node

        # Extract weights of edges
        edges = list(edges_data.items())  # Convert to a list of (next_node, attributes_dict)
        weights = [data['weight'] for _, data in edges]  # Extract weights from edge attributes

        # Calculate the total weight
        total_weight = sum(weights)
        
        # Generate a random threshold
        random_weight_threshold = random.uniform(0, total_weight)
        
        # Iterate through edges to find the next node based on the random threshold
        current_cumulative_weight = 0
        for next_node, data in edges:
            current_cumulative_weight += data['weight']
            if current_cumulative_weight >= random_weight_threshold:
                return next_node

        # # In case of rounding errors or if no node is picked, return a fallback node
        # return edges[-1][0] if edges else None
    
    def fitness(self, path):
        bin_sums = {}
        for bin_i in range(1, self.graph.number_of_bins+1):
            bin_sums[bin_i] = 0
        # Traverse the path and sum items in each bin
        for node in path:
            if isinstance(node, tuple) and len(node) == 2: # not S or E
                bin_num, item = node
                bin_sums[bin_num] += item

        # Calculate the fitness as the difference between the largest and smallest bin sum
        largest_bin = max(bin_sums.values())
        smallest_bin = min(bin_sums.values())
        fitness = largest_bin - smallest_bin
        # print(bin_sums, fitness)
        return fitness
    
    def generate_paths(self):
        path_list = []
     
        node = self.graph.start_node
        path_set = set()
        while node not in path_set:
            path_list.append(node)
            path_set.add(node)
            if node != "E":
                node = self.pick_next(node)
        fitness = self.fitness(path_list)
        return path_list, fitness
    
    def generate_all(self):
        all_paths_and_fitnesses = []
        number_of_paths = self.paths
        while number_of_paths != 0:
            all_paths_and_fitnesses.append(self.generate_paths())
            number_of_paths -= 1
        
        self.graph.update_paths(all_paths_and_fitnesses, self.evaporation_rate)
        return all_paths_and_fitnesses
    
# number_of_items = 500
# number_of_bins = 10
# bp1_items = [i for i in range(1, 501)]
# # number_of_items = 5
# # number_of_bins = 2
# # bp1_items = [1, 2,3, 4, 5]
# graph = digraph.OptimizedGraph(number_of_items, number_of_bins)
# graph.create_graph(bp1_items)


# aco = ACO(graph, 10, 0.9, "BP1")

# p = aco.generate_all()
# print("")

def trials(t, config_type):
    bins = None
    if config_type == "BP1":
        configurations = bp1_configurations
        bins = 10
    elif config_type == "BP2":
        configurations = bp2_configurations
        bins = 50
    else:
        raise ValueError("config_type must be 'BP1' or 'BP2'")

    all_results = {}
    for label, config in configurations.items():
        items, paths, evaporation = config

        i = None

        # Store each trial's fitness evaluations in a list
        trials_results = {}
        for trial in range(1, t + 1):
            paths_and_fitnesses = []
            seed = random.seed(trial)
            bp_graph = digraph.Graph(items, number_of_bins=bins, seed=seed)
            bp_graph.create_graph()
            ant = ACO(bp_graph, paths, evaporation, config=config_type)

            if paths == 100:
                i = 100
            if paths == 10:
                i = 1000

            for iteration in tqdm(range(i)):
                paths_and_fitnesses.extend(ant.generate_all())

            fitnesses = [fitness for _, fitness in paths_and_fitnesses]
            trials_results[f"Trial {trial}"] = fitnesses

        all_results[label] = trials_results
    return all_results




def performance_linear_graphs_by_trial_grid(results, folder_name, poly_degree=3):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Iterate through each configuration and its corresponding trials
    for config_name, trials in results.items():
        # Create a subfolder for each configuration inside the main folder
        config_folder = os.path.join(folder_name, config_name)
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
        
        for trial_name, fitness_values in trials.items():
            # Generate a figure for each trial
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Plot the actual fitness values
            steps = np.arange(len(fitness_values))
            ax.plot(steps, fitness_values, label='Fitness Values', linewidth=1.5, color='blue')
            
            # Fit a polynomial regression line of specified degree
            poly_coeffs = np.polyfit(steps, fitness_values, poly_degree)
            poly_eq = np.poly1d(poly_coeffs)
            poly_fit_values = poly_eq(steps)
            
            # Plot the polynomial regression line
            ax.plot(steps, poly_fit_values, label=f'Polynomial Fit (Degree {poly_degree})', color='red', linestyle='--')
            
            # Add plot titles and labels
            ax.set_title(f"{config_name} - {trial_name} Fitness with Polynomial Fit", fontsize=12)
            ax.set_xlabel("Step")
            ax.set_ylabel("Fitness")
            ax.legend()
            
            # Save the plot to the appropriate folder
            plot_filename = os.path.join(config_folder, f"{trial_name}_Fitness_Plot.png")
            plt.savefig(plot_filename)
            plt.close(fig)

def run_profiled_trials():
    results = trials(t=5, config_type="BP2")
    # print(results)
    performance_linear_graphs_by_trial_grid(results, folder_name="BP2", poly_degree=3)

profiler = cProfile.Profile()
profiler.enable()
run_profiled_trials()
profiler.dump_stats("profile_1results.prof")
profiler.disable()
