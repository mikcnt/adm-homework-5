from collections import defaultdict

class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.nodes = set()
        self.__edges_num = 0

    def add_edge(self, src, dst):
        if dst in self.edges[src]:
            return
        self.__edges_num += 1
        self.edges[src].add(dst)
        self.nodes.update([src, dst])
        
    def is_directed(self):
        pass
    
    def number_of_nodes(self):
        return len(self.nodes)
    
    def number_of_edges(self):
        return self.__edges_num
    
    def nodes_with_edges(self):
        return len(self.edges.keys())
    
    def __getitem__(self, src):
        if len(self.edges[src]) == 0:
            raise KeyError(src)
        return list(self.edges[src])