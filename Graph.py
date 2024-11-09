import numpy as np
import pprint


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
                
                if item_index == self.number_of_items-1:
                    self.graph[keyName][self.end_node] = np.random.random()

                if item_index == 0:
                    self.graph[self.start_node][keyName] = np.random.random()

                if item_index != self.number_of_items-1:
                    for next_bin in range(1, number_of_bins + 1):
                        nextKey = (next_bin, items[item_index+1])
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
                self.graph[current_node][next_node] += (100 / (fitness + epsilon))

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
                self.graph[current_node][next_node] += (100 / (fitness + epsilon))

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





