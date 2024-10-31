import networkx as nx
import numpy as np

class Graph:
    def __init__(self, items, number_of_bins, seed):
        self.graph = nx.DiGraph()  
        self.start_node = 'S' 
        self.end_node = 'E'      
        self.items = items
        self.number_of_items = len(items)
        self.number_of_bins = number_of_bins
        
        if seed is not None:
            np.random.seed(seed)

    def create_graph(self):
        # Add start and end nodes
        self.graph.add_node(self.start_node)
        self.graph.add_node(self.end_node)


        for item_index, item in enumerate(self.items):
            for bin in range(1, self.number_of_bins + 1):
                key_name = (bin, item)
                self.graph.add_node(key_name)  # Adding the node

                # Connect to end_node if it's the last item
                if item_index == self.number_of_items - 1:
                    self.graph.add_edge(key_name, self.end_node, weight=np.random.random())

                # Connect start_node to the first item in each bin
                if item_index == 0:
                    self.graph.add_edge(self.start_node, key_name, weight=np.random.random())

                # Connect current node to next nodes if not the last item
                if item_index != self.number_of_items - 1:
                    for next_bin in range(1, self.number_of_bins + 1):
                        next_key = (next_bin, self.items[item_index + 1])
                        self.graph.add_edge(key_name, next_key, weight=np.random.random())

    def display_graph(self):
        return self.graph.edges(data='weight')

    def update_paths(self, path_and_fitnesses, e):
        epsilon = 1e-10
        # Step 1: Update weights along the given path
        for path, fitness in path_and_fitnesses:
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                self.graph[current_node][next_node]['weight'] += (100 / (fitness + epsilon))

        # Step 2: Apply the decay factor to all edges
        for u, v, data in self.graph.edges(data=True):
            data['weight'] *= 1-e
    