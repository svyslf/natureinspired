import cProfile
import json
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import Graph
import random

from matplotlib import font_manager

font = "monospace"

# Check if the monospace font exists on system.
available_fonts = set(f.name.lower() for f in font_manager.fontManager.ttflist)

if font.lower() in available_fonts:
    plt.rcParams["font.family"] = font


bp1_items = [i for i in range(1, 501)]
bp2_items = [(i * i) / 2 for i in range(1, 501)]

further_experiment_bp1 = {
    "BP1, 100 Paths, 0.9 evaporation": [bp1_items, 10, 100, 0.9],
}
further_experiment_bp2 = {
    "BP2, 100 Paths, 0.9 evaporation": [bp2_items, 50, 100, 0.9],
}
bp1_configurations = {
    "BP1, 100 Paths, 0.9 evaporation": [bp1_items, 10, 100, 0.9],
    "BP1, 100 Paths, 0.6 evaporation": [bp1_items, 10, 100, 0.6],
    "BP1, 10 Paths, 0.9 evaporation": [bp1_items, 10, 10, 0.9],
    "BP1, 10 Paths, 0.6 evaporation": [bp1_items, 10, 10, 0.6],
}
bp2_configurations = {
    "BP2, 100 Paths, 0.9 evaporation": [bp2_items, 50, 100, 0.9],
    "BP2, 100 Paths, 0.6 evaporation": [bp2_items, 50, 100, 0.6],
    "BP2, 10 Paths, 0.9 evaporation": [bp2_items, 50, 10, 0.9],
    "BP2, 10 Paths, 0.6 evaporation": [bp2_items, 50, 10, 0.6],
}

all_configs = {
    "BP1, 100 Paths, 0.9 evaporation": [bp1_items, 10, 100, 0.9],
    "BP1, 100 Paths, 0.6 evaporation": [bp1_items, 10, 100, 0.6],
    "BP1, 10 Paths, 0.9 evaporation": [bp1_items, 10, 10, 0.9],
    "BP1, 10 Paths, 0.6 evaporation": [bp1_items, 10, 10, 0.6],
    "BP2, 100 Paths, 0.9 evaporation": [bp2_items, 50, 100, 0.9],
    "BP2, 100 Paths, 0.6 evaporation": [bp2_items, 50, 100, 0.6],
    "BP2, 10 Paths, 0.9 evaporation": [bp2_items, 50, 10, 0.9],
    "BP2, 10 Paths, 0.6 evaporation": [bp2_items, 50, 10, 0.6],
}


class Aco_IBPS:
    def __init__(self, graph, p, e):
        self.graph = graph
        self.paths = p
        self.evaporation_rate = e
        self.best_fitness = float("inf")
        self.isconverged = False

    def pick_next(self, node):
        edges = self.graph.edges(node)
        weights = edges.values()
        total_weight = sum(weights)

        random_weight_threshold = random.uniform(0, total_weight)
        current_cumulative_weight = 0
        for node in edges:
            current_cumulative_weight += edges[node]
            if current_cumulative_weight >= random_weight_threshold:
                return node

    def fitness(self, path):
        bin_sums = [0] * (self.graph.number_of_bins + 1)

        for node in path:
            if not isinstance(node, str):  # not S or E
                bin_num, item = node
                bin_sums[bin_num] += item

        largest_bin = max(bin_sums[1:])
        smallest_bin = min(bin_sums[1:])
        fitness = largest_bin - smallest_bin

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

    def itbms(self, iteration):
        if (iteration != 0 and iteration <= 600) or self.isconverged:
            if (iteration % 300) == 0:
                evaporation_step = 0.4
                paths_step = 45

                self.evaporation_rate = max(
                    0.1, self.evaporation_rate - evaporation_step
                )
                self.paths = max(10, int(self.paths - paths_step))

    def converged(self, solutions, similarity_threshold=0.7):
        solution_counts = Counter(solutions)
        most_common_count = solution_counts.most_common(1)[0][1]
        similarity_ratio = most_common_count / len(solutions)
        return similarity_ratio >= similarity_threshold

    def generate_all(self, iteration):
        self.itbms(iteration)
        all_paths_and_fitnesses = []
        number_of_paths = self.paths
        while number_of_paths != 0:
            all_paths_and_fitnesses.append(self.generate_paths())
            number_of_paths -= 1
        fitnesses_list = [fitness for path, fitness in all_paths_and_fitnesses]

        if self.converged(fitnesses_list):
            self.isconverged = True
            self.paths = 100
            self.evaporation_rate = 0.9
        self.graph.update_paths(all_paths_and_fitnesses, self.evaporation_rate)
        return all_paths_and_fitnesses


def trials(t, configurations):

    all_results = {}
    for label, config in configurations.items():
        items, bins, paths, evaporation = config

        i = None
        if paths != 100:
            raise ValueError("aco_IBPS() does not accept anything other than 100 paths")
        if evaporation != 0.9:
            raise ValueError("aco_IBPS() does not accept anything other than 0.9 e")

        trials_results = {}
        for trial in range(1, t + 1):
            paths_and_fitnesses = []
            seed = random.seed(trial)
            print(trial, label)
            bp_graph = Graph.Graph(items, number_of_bins=bins, seed=seed)
            ant = Aco_IBPS(bp_graph, paths, evaporation)

            for iteration in tqdm(range(1000)):
                paths_and_fitnesses.extend(ant.generate_all(iteration))
            fitnesses = [fitness for _, fitness in paths_and_fitnesses]
            trials_results[f"Trial {trial}"] = fitnesses

        all_results[label] = trials_results
    return all_results


def split_by_bp(results):
    # New dictionary to store the transformed structure
    transformed_data = {}

    # Iterate over the original dictionary to transform it
    for key, value in results.items():
        # Split the original key into BP type, paths, and evaporation rate
        bp_type, paths, evaporation = key.split(", ", 2)

        # Initialize nested dictionaries if they do not exist
        if bp_type not in transformed_data:
            transformed_data[bp_type] = {}
        if paths + ", " + evaporation not in transformed_data[bp_type]:
            transformed_data[bp_type][paths + ", " + evaporation] = {}

        # Assign the trials data to the new nested structure
        transformed_data[bp_type][paths + ", " + evaporation] = value
    return transformed_data


def run_profiled_trials():

    results = trials(t=5, configurations=further_experiment_bp1)

    with open("bp2_conv.json", "w") as file:
        json.dump(results, file, indent=4)

    data_by_bp = split_by_bp(results)
    with open("bp2_conv_split.json", "w") as file:
        json.dump(data_by_bp, file, indent=4)


profiler = cProfile.Profile()
profiler.enable()
run_profiled_trials()
profiler.dump_stats("profile_results_itbms_conv_bp2.prof")
profiler.disable()
