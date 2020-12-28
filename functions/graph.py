from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.neighbours = defaultdict(set)
        self.nodes = set()
        self.__edges_num = 0

    def add_edge(self, src, dst):
        if dst in self.edges[src]:
            return
        self.__edges_num += 1
        self.edges[src].add(dst)
        self.nodes.update([src, dst])
        self.neighbours[src].add(dst)
        self.neighbours[dst].add(src)
        
    def is_directed(self):
        for src, dest in self.edges.items():
            is_direct = False
            for node in dest:
                if src not in self.edges[node]:
                    is_direct = True
                    break
            if is_direct:
                break
        return is_direct
    
    def number_of_nodes(self):
        return len(self.nodes)
    
    def number_of_edges(self):
        return self.__edges_num
    
    def nodes_with_edges(self):
        return len(self.edges.keys())
    
    def density(self):
        # TODO: the graph is dense or sparse?
        num_nodes = self.number_of_nodes()
        num_edges = self.number_of_edges()
        density = 2 * num_edges / (num_nodes * (num_nodes - 1))
        if self.is_directed():
            density = density / 2
        return density
    
    def degree(self, node):
        return len(self.neighbours[node])
    
    def degree_distribution_plot(self):
        degrees = Counter([self.degree(n) for n in self.nodes])
        x, y = zip(*degrees.items())
        plt.figure(figsize=(14, 6))
        plt.xlabel('Degree')
        plt.xscale('log')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title('Degree distribution (logscale)')
        plt.grid(alpha=0.5, zorder=1)
        plt.scatter(x, y, marker='.', zorder=2)
        plt.show()
    
    def __getitem__(self, src):
        if len(self.edges[src]) == 0:
            raise KeyError(src)
        return list(self.edges[src])