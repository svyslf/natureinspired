import cProfile
from io import StringIO
import json
import os
import pprint
import pstats

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import Graph
import random
import matplotlib.font_manager


plt.rcParams["font.family"] = "monospace"

bp1_items = [i for i in range(1, 501)]
bp2_items = [(i * i) / 2 for i in range(1, 501)]

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


class Ant:
    def __init__(self, graph, p, e):
        self.graph = graph
        self.paths = p
        self.evaporation_rate = e
        self.best_fitness = float("inf")

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

    def generate_all(self):
        all_paths_and_fitnesses = []
        number_of_paths = self.paths
        while number_of_paths != 0:
            all_paths_and_fitnesses.append(self.generate_paths())
            number_of_paths -= 1

        self.graph.update_paths(all_paths_and_fitnesses, self.evaporation_rate)
        return all_paths_and_fitnesses


def trials(t, configurations):

    all_results = {}
    for label, config in configurations.items():
        items, bins, paths, evaporation = config

        i = None

        trials_results = {}
        for trial in range(1, t + 1):
            paths_and_fitnesses = []
            seed = random.seed(trial)
            print(trial, label)
            bp_graph = Graph.Graph(items, number_of_bins=bins, seed=seed)
            ant = Ant(bp_graph, paths, evaporation)

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
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for config_name, trials in results.items():
        config_folder = os.path.join(folder_name, config_name)
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)

        for trial_name, fitness_values in trials.items():

            fig, ax = plt.subplots(figsize=(12, 12))
            steps = np.arange(len(fitness_values))
            ax.plot(
                steps,
                fitness_values,
                label="Fitness Values",
                linewidth=1.5,
                color="blue",
            )

            poly_coeffs = np.polyfit(steps, fitness_values, poly_degree)
            poly_eq = np.poly1d(poly_coeffs)
            poly_fit_values = poly_eq(steps)

            ax.plot(
                steps,
                poly_fit_values,
                label=f"Polynomial Fit (Degree {poly_degree})",
                color="red",
                linestyle="--",
            )

            ax.set_title(
                f"{config_name} - {trial_name} Fitness with Polynomial Fit", fontsize=12
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Fitness")
            ax.legend()

            plot_filename = os.path.join(
                config_folder, f"{trial_name}_Fitness_Plot.png"
            )
            plt.savefig(plot_filename)
            plt.close(fig)


def all_configurations_all_trials_grid(results, config):
    num_configs = len(results)
    grid_size = int(np.ceil(np.sqrt(num_configs)))  # Square grid dimensions
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(10, 10), constrained_layout=False
    )

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i, (config_name, trials) in enumerate(results.items()):
        # Create a new subplot for each configuration
        axes[i].set_title(config_name)
        axes[i].set_xlabel("Evaluation")
        axes[i].set_ylabel("Fitness")
        axes[i].grid(True)

        for trial_name, fitness_values in trials.items():
            # Plot each trial's fitness values
            axes[i].plot(fitness_values, linewidth=1.5, label=trial_name)

        # Add legend for each subplot
        axes[i].legend()

    # Hide any empty subplots if configurations < grid_size^2
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6
    )  # Increased hspace

    plt.suptitle(
        f"{config} fitness over 10k fitness evaluations for each configuration",
        fontsize=15,
    )
    plt.savefig(f"{config} linear Fitness evals")
    # plt.show()


def performance_bar_graph_by_trial_grid(results, config):
    num_configs = len(results)
    grid_size = int(np.ceil(np.sqrt(num_configs)))  # Square grid dimensions
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(10, 10), constrained_layout=False
    )

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i, (config_name, trials) in enumerate(results.items()):
        # Prepare data for best and worst fitness values
        best_fitness = []
        worst_fitness = []
        trial_labels = []

        for trial_name, fitness_values in trials.items():
            trial_labels.append(trial_name)
            best_fitness.append(min(fitness_values))
            worst_fitness.append(max(fitness_values))

        # Bar positions
        x = np.arange(len(trial_labels))  # the label locations
        bar_width = 0.35  # width of the bars

        # Create bars for best and worst fitness
        axes[i].bar(
            x - bar_width / 2,
            best_fitness,
            bar_width,
            label="Best Fitness",
            color="green",
        )
        # axes[i].bar(x + bar_width / 2, worst_fitness, bar_width, label='Worst Fitness', color='red')

        # Create a new subplot for each configuration
        axes[i].set_title(config_name)
        axes[i].set_xlabel("Trial")
        axes[i].set_ylabel("Fitness")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(trial_labels)
        axes[i].grid(True)

        for j in range(len(trial_labels)):
            axes[i].text(
                x[j] - bar_width / 2,
                best_fitness[j] + 10,
                str(best_fitness[j]),
                ha="center",
                va="bottom",
                fontsize=10,
                color="darkgreen",
            )
            # axes[i].text(x[j] + bar_width / 2, worst_fitness[j]+10, str(worst_fitness[j]), ha='center', va='bottom', fontsize=10, color='darkred')

        # Add legend for each subplot
        axes[i].legend()

    # Hide any empty subplots if configurations < grid_size^2
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6
    )  # Increased hspace

    plt.suptitle(f"{config} Best and Worst Fitness over all Trials", fontsize=15)
    plt.savefig(f"{config} bar Fitness evals")

    # plt.show()


def performance_linear_graph_by_trial_grid(results):
    # Iterate through each configuration and its corresponding trials
    for config_name, trials in results.items():
        # Prepare the list of all trials under this configuration
        all_trials = [
            (trial_name, fitness_values)
            for trial_name, fitness_values in trials.items()
        ]

        num_trials = len(all_trials)
        grid_size = int(np.ceil(np.sqrt(num_trials)))  # Square grid dimensions
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(10, 10), constrained_layout=False
        )

        # Flatten axes for easy indexing
        axes = axes.flatten()

        for i, (trial_name, fitness_values) in enumerate(all_trials):
            # Plot each trial's fitness values in its own subplot
            axes[i].plot(fitness_values, linewidth=1.5)
            axes[i].set_title(f"{trial_name}", fontsize=10)
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel("Fitness")

        # Hide any empty subplots if number of trials < grid_size^2
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6
        )  # Increased hspace for readability

        plt.suptitle(f"{config_name} Fitness over all trials", fontsize=15)
        plt.savefig(f"{config_name}_Fitness_evals.png")
        plt.close(fig)  # Close the figure to free up memory


def run_profiled_trials():

    results = trials(t=5, configurations=all_configs)
    with open("trial_results.json", "w") as file:
        json.dump(results, file, indent=4)
    performance_linear_graphs_by_trial_grid(results, folder_name="BP1.s", poly_degree=3)


profiler = cProfile.Profile()
profiler.enable()
run_profiled_trials()
profiler.dump_stats("profile_results.prof")
profiler.disable()
