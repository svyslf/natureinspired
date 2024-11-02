import numpy as np
import pprint


class Graph:
    def __init__(self, items, number_of_bins, seed=None):
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
    
    def create(self):
        return self.graph

    def edges(self, node):
        return self.graph[node]

    def update_paths(self, path_and_fitnesses, e):
        epsilon = 1e-10

        for path, fitness in path_and_fitnesses:
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                self.graph[current_node][next_node] += (100 / (fitness + epsilon))

        for node, neighbors in self.graph.items():
            for next_node in neighbors:
                neighbors[next_node] *= 1-e
                
    def nodes(self):
        return sorted(self.graph.keys())

    def display_graph(self):
        pprint.pprint(self.graph)




