from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class Graph:
    """Class for directed and undirected graph.

    Attributes:
        edges (defaultdict): Edges of the graph, in the form of src (key) -> dst (values, can be more than 1).
        nodes (set): Set containing the whole list of nodes of the graph.
        neighbours (defaultdict): For each node of the graph, the complete list of neighbours is contained in this dictionary.
            Note that we consider both out-neighbours and in-neighbours, if the graph is directed.

    """
    
    def __init__(self):
        self.edges = defaultdict(set)
        self.nodes = set()
        self.in_neighbours = defaultdict(set)
        self.out_neighbours = defaultdict(set)
        self.__edges_num = 0

    def add_edge(self, src, dst):
        """Add a new edge to the existing graph.

        Args:
            src (int): Source node of the edge.
            dst (int): Arriving node of the edge.
        """
        # If the edge already exist, we won't do anything
        if dst in self.edges[src]:
            return
        self.__edges_num += 1
        # Add the actual edge
        self.edges[src].add(dst)
        # Add both source and dest nodes to the set of nodes
        self.nodes.update([src, dst])
        # Keep track of the neighbours of each node in the graph
        self.in_neighbours[dst].add(src)
        self.out_neighbours[src].add(dst)
        
    def is_directed(self):
        """Check if the graph is directed, by looking at the source nodes of the edges and searching for an inverse edge."""
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
        """The density of a graph is defined as follows:
        - 2*E / (V * (V - 1)) if the graph is undirected;
        - E / (V * (V - 1)) if the graph is directed.

        Returns:
            float: Density of the graph (between 0 and 1).
        """
        # TODO: the graph is dense or sparse?
        num_nodes = self.number_of_nodes()
        num_edges = self.number_of_edges()
        density = 2 * num_edges / (num_nodes * (num_nodes - 1))
        if self.is_directed():
            density = density / 2
        return density
    
    def in_degree(self, node):
        """Cardinality of the in-neighbours of the node."""
        return len(self.in_neighbours[node])
    
    def out_degree(self, node):
        """Cardinality of the out-neighbours of the node."""
        return len(self.out_neighbours[node])
    
    def degree(self, node):
        """The degree of a node is defined as the number of neighbours it has.
        This corresponds to the cardinality of the union of the in-neighbours and the out-neighbours.
        """
        return len(set.union(self.in_neighbours[node], self.out_neighbours[node]))
    
    def degree_distribution_plot(self):
        """Plot the degree distribution in logarithmic scale."""
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
    
    def pages_within_clicks(self, v, d):
        """Returns the set of pages reachable from a given node v and a number of clicks d.

        Args:
            v (int): Starting page (node).
            d (int): Number of clicks.

        Returns:
            set: Set containing all the reachable pages within d clicks.
        """
        src_pages = {v}
        pages = set()
        for _ in range(d):
            temp = set()
            # temp corresponds to the set of pages we reach at each click
            for src in src_pages:
                temp.update(self.edges[src])
            # Add temp to the pages we found in the previous iteration
            pages.update(temp)
            # For the next iteration, we are going to search for the pages we can reach
            # starting from the ones we just obtained
            src_pages = temp
        return pages
    
    def __getitem__(self, src):
        if len(self.edges[src]) == 0:
            raise KeyError(src)
        return list(self.edges[src])