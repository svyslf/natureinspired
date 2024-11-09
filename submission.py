from collections import Counter
import json
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

bp1_items = [i for i in range(1, 501)]
bp2_items = [(i * i) / 2 for i in range(1, 501)]

further_experiment = {
    "BP1, 100 Paths, 0.9 evaporation": [bp1_items, 10, 100, 0.9],
    "BP2, 100 Paths, 0.9 evaporation": [bp2_items, 50, 100, 0.9],
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


class Graph:
    def __init__(self, items, number_of_bins, seed=None):
        """
        Initializes a "construction graph" structure from the given Bin Packing problem set.
        Each node in the graph is a tuple like (bin, item)
        Parameters:
        items (list): A list of items to be placed in bins.
        number_of_bins (int): The number of bins in which the items must be packed.
        seed (int, optional): A seed for the random number generator for reproducibility across trials.
        """
        self.items = items
        self.number_of_items = len(items)
        self.number_of_bins = number_of_bins
        self.graph = {}
        self.start_node = "S"
        self.end_node = "E"
        self.graph[self.start_node] = {}
        self.graph[self.end_node] = {}

        if seed is not None:
            np.random.seed(seed)

        for item_index, item in enumerate(items):
            for bin in range(1, number_of_bins + 1):
                keyName = (bin, item)
                self.graph[keyName] = {}

                if item_index == self.number_of_items - 1:
                    self.graph[keyName][self.end_node] = np.random.random()

                if item_index == 0:
                    self.graph[self.start_node][keyName] = np.random.random()

                if item_index != self.number_of_items - 1:
                    for next_bin in range(1, number_of_bins + 1):
                        nextKey = (next_bin, items[item_index + 1])
                        self.graph[keyName][nextKey] = np.random.random()

    def edges(self, node):
        """
        Returns the edges (neighbors) of a given node in the graph.

        Parameters:
        node (tuple): The node whose edges are being queried.

        Returns:
        dict: A dictionary of neighboring nodes and edge weights.
        """
        return self.graph[node]

    def update_paths(self, path_and_fitnesses, e):
        """
        Updates edge weights based on fitness scores for a set of p paths, evaporates all the links in the graph.

        Parameters:
        path_and_fitnesses (list of tuples): A list of (path, fitness) tuples, where each path is a list of nodes.
        e (float): The evaporation factor; edges are multiplied by this factor to simulate evaporation.
        """
        epsilon = 1e-10
        for path, fitness in path_and_fitnesses:
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                self.graph[current_node][next_node] += 100 / (fitness + epsilon)

        for node, neighbors in self.graph.items():
            for next_node in neighbors:
                neighbors[next_node] *= e

    def update_path(self, path, fitness):
        """
        Updates edge weights based on the fitness score for a single path.
        (this function is only used for further experimentation)

        Parameters:
        path (list): A list of nodes representing a path.
        fitness (float): The fitness score of the path, where a lower score is better.
        """
        epsilon = 1e-10
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            self.graph[current_node][next_node] += 100 / (fitness + epsilon)

    def update_evaporation(self, e):
        """
        Applies an evaporation factor to all edges in the graph, reducing their weights.
        (this function is only used for further experimentation)
        Parameters:
        e (float): The evaporation rate, where each edge weight is multiplied by e.
        """
        for node, neighbors in self.graph.items():
            for next_node in neighbors:
                neighbors[next_node] *= e


class Aco:
    def __init__(self, graph, p, e):
        """
        Initializes the aco algorithm with the given graph, number of paths, and evaporation rate.
        Adds a check for whether the algorithm has converged.

        Args:
            graph: The graph object that contains bins and items.
            p (int): The number of paths to generate.
            e (float): The evaporation rate.
        """
        self.graph = graph
        self.paths = p
        self.evaporation_rate = e
        self.best_fitness = float("inf")
        self.isconverged = False

    def pick_next(self, node):
        """
        Picks the next node to move to, based on the edges and their weights (edge pheromone).

        Args:
            node (tuple): The current node (bin, item) from which to pick the next node.

        Returns:
            The next node to visit based on weighted probability.
        """
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
        """
        Calculates the fitness of a given path, based on the sum of the items in each bin.

        The fitness is defined as the difference between the heaviest and lightest bin.

        Args:
            path (list): The path of bin packs taken from S to E.

        Returns:
            float: The fitness score of the path.
        """
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
        """
        Generates a path starting from the start node to the end node,
        and then calculates fitness.

        The path is generated based on the results of the pick next function.

        Returns:
            tuple: A tuple containing the generated path (list) and its fitness (float).
        """
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

    def ibps(self, iteration):
        """
        The ibps function adjusts the number of paths explored and the evaporation rate
        every 300 iterations. (Further exploration function)

        Allows the alogrithm to converge to a good solution while exploring many paths.

        Args:
            itertation (int): the current iteration of the algorithm
        """
        if (iteration != 0 and iteration <= 600) or self.isconverged:
            if (iteration % 300) == 0:
                evaporation_step = 0.4
                paths_step = 45

                self.evaporation_rate = max(
                    0.1, self.evaporation_rate - evaporation_step
                )
                self.paths = max(10, int(self.paths - paths_step))

    def converged(self, solutions, similarity_threshold=0.7):
        """
        Determines if the algorithm has converged based on the similarity of solutions.

        The algorithm is considered to have converged if the most common solution
        appears with a frequency above a certain threshold (similarity ratio).

        Args:
            solutions (list): A list of solutions generated by the algorithm.
            similarity_threshold (float, optional): The threshold for the similarity ratio
                                                    to consider the algorithm converged.
                                                    Default is 0.7 (70%).

        Returns:
            bool: True if the algorithm has converged (i.e., the most common solution
                appears with a frequency greater than or equal to the similarity threshold),
                False otherwise.
        """
        solution_counts = Counter(solutions)
        most_common_count = solution_counts.most_common(1)[0][1]
        similarity_ratio = most_common_count / len(solutions)
        return similarity_ratio >= similarity_threshold

    def generate_all(self, iteration, aco_type):
        """
        Calls the ibps function which optimises paramaters based on iteration.

        Generates p paths (specified in the initialization) and updates the graph using the evaporation rate.

        Checks when the algorithm has converged, and resets number of paths and evaporation rate values.
        Args:
            itertation (int): the current iteration of the algorithm
            aco_type (string): the type of aco being ran
        Returns:
            list: A list of tuples, each containing a generated path and its fitness score.
        """
        if aco_type == "ibps":
            self.ibps(iteration)

        all_paths_and_fitnesses = []
        number_of_paths = self.paths
        while number_of_paths != 0:
            all_paths_and_fitnesses.append(self.generate_paths())
            number_of_paths -= 1
        fitnesses_list = [fitness for path, fitness in all_paths_and_fitnesses]

        if aco_type == "ibps":
            if self.converged(fitnesses_list):
                self.isconverged = True
                self.paths = 100
                self.evaporation_rate = 0.9

        self.graph.update_paths(all_paths_and_fitnesses, self.evaporation_rate)
        return all_paths_and_fitnesses


def trials(t, configurations, aco_type="normal"):
    """
    Runs multiple trials for different configurations and collects the results.

    For each configuration, the function runs a specified number of trials,
    generates paths using the aco and records the fitness values for each trial.
    The results are stored in a dictionary, with the configuration labels as keys.

    Args:
        t (int): The number of trials to run for each configuration.
        configurations (dict): A dictionary where the keys are configuration labels
                               and the values are tuples containing the following
                               parameters for each configuration:
                               - items: The items to be distributed.
                               - bins: The number of bins.
                               - paths: The number of paths to generate.
                               - evaporation: The evaporation rate for the aco algorithm.

    Returns:
        dict: A dictionary where each key is a configuration label and each value
              is another dictionary with trial labels as keys and lists of fitness
              values for each trial as the corresponding values.
    """
    all_results = {}
    for label, config in configurations.items():
        items, bins, paths, evaporation = config

        i = None
        if aco_type == "ibps":
            if paths != 100:
                raise ValueError(
                    "aco_IBPS() does not accept anything other than 100 paths"
                )
            if evaporation != 0.9:
                raise ValueError("aco_IBPS() does not accept anything other than 0.9 e")

        trials_results = {}
        for trial in range(1, t + 1):
            paths_and_fitnesses = []
            seed = random.seed(trial)
            print(trial, label)
            bp_graph = Graph(items, number_of_bins=bins, seed=seed)
            aco = Aco(bp_graph, paths, evaporation)

            if paths == 100:
                i = 100
            if paths == 10:
                i = 1000
            if aco_type == "ibps":
                i = 1000

            for iteration in tqdm(range(i)):
                paths_and_fitnesses.extend(aco.generate_all(iteration, aco_type))
            fitnesses = [fitness for _, fitness in paths_and_fitnesses]
            trials_results[f"Trial {trial}"] = fitnesses

        all_results[label] = trials_results
    return all_results


def bar_graphs_for_aco(results, folder_name):
    """
    Generates and saves a bar graphs for the best fitnesses from every trial for the original aco.
    Args:
        results (dict): A dictionary containing the aco results with all configurations.
                        The keys are trial names, and the values are lists of fitness values.
        folder_name (str): The name of the folder where the generated plots will be saved.
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

     # loop the configurations and bp type ( bp1 or bp2)
    for bp_type, configurations in results.items():
        num_configs = len(configurations)
        grid_size = math.ceil(
            num_configs**0.5
        ) 
        # for making the nice grid plot thing
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
        fig.suptitle(f"Best Fitness Values for {bp_type}", fontsize=15)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)


        axes = axes.flatten()

        for idx, (config_name, trials) in enumerate(configurations.items()):
            trial_names = list(trials.keys())
            best_fitness_values = [
                np.min(fitness_values) for fitness_values in trials.values()
            ]
            mean = np.mean(best_fitness_values)

            ax = axes[idx]

            ax.bar(trial_names, best_fitness_values, color="darkslategray", width=0.6)
            ax.bar(
                len(trial_names), mean, color="teal", width=0.6, label="Mean Fitness"
            )
            ax.set_title(f"{config_name}", fontsize=10)
            ax.set_xlabel("Trials", fontsize=8)
            ax.set_ylabel("Best Fitness", fontsize=8)
            ax.set_xticks(range(len(trial_names) + 1))
            ax.set_xticklabels(
                trial_names + ["Mean"], rotation=45, ha="right", fontsize=7
            )
            max_fitness = max(best_fitness_values)

            ax.set_ylim(0, max_fitness * 1.1)

            for i, v in enumerate(best_fitness_values):
                ax.text(
                    i,
                    v + (0.005 * max(best_fitness_values)),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

            ax.text(
                len(trial_names),
                mean + (0.02 * max_fitness),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )

        for idx in range(len(configurations), grid_size * grid_size):
            axes[idx].axis("off")

        plot_filename = os.path.join(folder_name, f"{bp_type}_bar_graph.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)


def bar_graph_for_aco_IBPS(results, folder_name):
    """
    Generates and saves a bar graph for the best and converged fitness values from the ibps enhanced aco.
    Args:
        results (dict): A dictionary containing the aco results with a single configuration.
                        The top level key must be the BP list. (BP1 / BP2)
                        The keys are trial names, and the values are lists of fitness values.
        folder_name (str): The name of the folder where the generated plots will be saved.

    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # loop thru results and make the bar graphs same as before (ignore the spaghettiness please)
    for bp_type, configurations in results.items():
        trial_names = list(configurations.keys())
        best_fitness_values = [
            np.min(fitness_values) for fitness_values in configurations.values()
        ]
        convergence_values = [
            fitness_values[-1] for fitness_values in configurations.values()
        ]  

        fig, (ax_best, ax_converged) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"IPBS enhanced {bp_type} best and converged values", fontsize=15)

        ax_best.bar(trial_names, best_fitness_values, color="darkslategray", width=0.6)
        mean_best = np.mean(best_fitness_values)
        ax_best.bar(
            len(trial_names),
            mean_best,
            color="gray",
            width=0.6,
            label="Mean Best Fitness",
        )
        ax_best.set_title("Best Fitness Values", fontsize=12)

        ax_best.set_xticks(range(len(trial_names) + 1))
        ax_best.set_xticklabels(
            trial_names + ["Mean"], rotation=45, ha="right", fontsize=8
        )
        ax_best.set_ylim(0, max(best_fitness_values) * 1.1)

        for i, v in enumerate(best_fitness_values):
            ax_best.text(
                i,
                v + 0.02 * max(best_fitness_values),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_best.text(
            len(trial_names),
            mean_best + 0.02 * max(best_fitness_values),
            f"{mean_best:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
        # to measure how close the convergence is to best results!!
        ax_converged.bar(trial_names, convergence_values, color="teal", width=0.6)
        mean_convergence = np.mean(convergence_values)
        ax_converged.bar(
            len(trial_names),
            mean_convergence,
            color="lightgray",
            width=0.6,
            label="Mean Convergence Fitness",
        )
        ax_converged.set_title("Convergence Fitness Values", fontsize=12)

        ax_converged.set_xticks(range(len(trial_names) + 1))
        ax_converged.set_xticklabels(
            trial_names + ["Mean"], rotation=45, ha="right", fontsize=8
        )
        ax_converged.set_ylim(0, max(convergence_values) * 1.1)

        for i, v in enumerate(convergence_values):
            ax_converged.text(
                i,
                v + 0.02 * max(convergence_values),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_converged.text(
            len(trial_names),
            mean_convergence + 0.02 * max(convergence_values),
            f"{mean_convergence:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

        plot_filename = os.path.join(
            folder_name, f"{bp_type}_Best_vs_Convergence_Fitness_Bar_Plots.png"
        )
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)


def reformat_aco_results(results):
    """
    Reorganizes the data from aco into a nested dictionary where the top-level key is the BP type,
    followed by the combination of paths and evaporation rate. Easier to visualise with matplotlib.

    Args:
        results (dict): the dictionary returned from the aco.

    Returns:
        dict: A nested dictionary where the first level is grouped by BP type,
              the second level is grouped by the combination of paths and evaporation rate,
              and the final level contains the trial results.
    """
    transformed_data = {}

    for key, value in results.items():
        bp_type, paths, evaporation = key.split(", ", 2)

        if bp_type not in transformed_data:
            transformed_data[bp_type] = {}
        if paths + ", " + evaporation not in transformed_data[bp_type]:
            transformed_data[bp_type][paths + ", " + evaporation] = {}

        transformed_data[bp_type][paths + ", " + evaporation] = value
    return transformed_data


def run_trials(run_aco_with_IBPS=False):
    """
    Runs a set of trials and saves the results as graphs and json files.
    Args:
        run_aco_with_IBPS (Bool), Optional: Run the further experiment aco by setting to true.
    Returns:
        None
    """
    if run_aco_with_IBPS == False:
        results_folder = "Aco_results"
        results = trials(t=5, configurations=all_configs)
    else:
        results_folder = "Aco_IBPS_results"
        results = trials(t=5, configurations=further_experiment, aco_type="ibps")

    os.makedirs(results_folder, exist_ok=True)

    with open(os.path.join(results_folder, "aco_result.json"), "w") as file:
        json.dump(results, file, indent=4)

    data_by_bp = reformat_aco_results(results)

    with open(os.path.join(results_folder, "aco_result_reformatted.json"), "w") as file:
        json.dump(data_by_bp, file, indent=4)

    if run_aco_with_IBPS == False:
        bar_graphs_for_aco(data_by_bp, "Aco_Graphs")
    else:
        bar_graph_for_aco_IBPS(results, "Aco_IBPS_Graphs")

# Set run_aco_with_IBPS = True for the further experiment enhanced version
run_trials(run_aco_with_IBPS=False) 


