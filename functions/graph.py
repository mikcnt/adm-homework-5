from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from collections import deque
from .utils import categories_in_graph


class Graph:
    """Class for directed and undirected graph.

    Attributes:
        cat_link_dict (dict): Dictionary containing categories as keys and the links for each category as values.
        edges (defaultdict): Edges of the graph, in the form of src (key) -> dst (values, can be more than 1).
        nodes (set): Set containing the whole list of nodes of the graph.
        in_neighbours (defaultdict): Dictionary containing the in-neighbours of a node.
        out_neighbours (defaultdict): Dictionary containing the out-neighbours of a node.
        __edges_num (int): Number of edges in the graph.

    """

    def __init__(self):
        self.cat_link_dict = categories_in_graph()
        self.edges = defaultdict(set)
        self.nodes = set()
        self.in_neighbours = defaultdict(set)
        self.out_neighbours = defaultdict(set)
        self.__edges_num = 0

    def __getitem__(self, node):
        if node not in self.nodes:
            raise KeyError(node)
        return list(self.edges[node])

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
        plt.xlabel("Degree")
        plt.xscale("log")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.title("Degree distribution (logscale)")
        plt.grid(alpha=0.5, zorder=1)
        plt.scatter(x, y, marker=".", zorder=2)
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

    def bfs_shortest_path(self, src, dst):
        """Returns the shortest path between a starting node and a target node,
        according to the Dijkstra algorithm.

        Args:
            src (int): First node of the path.
            dst (int): Ending node of the path.

        Returns:
            list: If a path between src and dst nodes exist, it is returned. Otherwise, it is returned None.
        """
        # Base case
        if src == dst:
            return [src]
        # Keep track of the visited nodes
        # We are going to stop if all the nodes are visited
        visited = {src}
        queue = deque([(src, [])])
        while queue:
            current, path = queue.popleft()
            visited.add(current)
            for neighbor in self.edges[current]:
                # Stopping criteria: when path is found
                if neighbor == dst:
                    return path + [current, neighbor]
                if neighbor in visited:
                    continue
                queue.append((neighbor, path + [current]))
                visited.add(neighbor)
        # We reach this part only if there is no path between vertices
        return None

    def most_central_article(self, category):
        """Compute the most central node of a category.

        Args:
            category (str): String name of the input category.

        Returns:
            int: Most central node, which represents the node with highest in-degree.
        """
        nodes_in_cat = self.cat_link_dict[category]
        in_degrees = {}
        for node in nodes_in_cat:
            in_degrees[node] = self.in_degree(node)
        return max(in_degrees, key=in_degrees.get)

    def min_clicks(self, category, pages):
        """Returns the minimum number of clicks to reach a set of pages from the
        most central node of a category.

        Args:
            category (str): String name of the input category.
            pages (set): Set containing the input pages for which we want the number of clicks.

        Returns:
            int: Minimum number of clicks.
        """
        # Base cases:
        # nodes in `pages` are not in the selected category
        # nodes in `pages` are not in the graph nodes
        nodes_in_cat = set(self.cat_link_dict[category])
        if pages.intersection(self.nodes) != pages:
            print("Not possible")
            return
        if nodes_in_cat.intersection(pages) != pages:
            print("Not possible")
            return

        most_central = self.most_central_article(category)

        # Compute the distances between all the nodes in pages
        # and the most central node of the category
        # These are represented by the lenght of the shortest paths
        distances = []
        for node in pages:
            path = self.bfs_shortest_path(most_central, node)
            if not path:
                print("Not possible")
                return
            distances.append(len(path))

        # We return the max of these distances, since it is the
        # value for which we are sure we hit all the nodes in `pages`
        return max(distances)

    def nodes_in_category(self, category):
        """Returns the nodes in the category actually contained in the graph nodes.

        Args:
            category (str): String name of the input category.

        Returns:
            set: Set containing the graph nodes contained in the category.
        """
        return set(self.cat_link_dict[category]).intersection(self.nodes)


def category_subgraph(graph, category1, category2):
    """Create subgraph containing the edges with source nodes in the
    first category and destination nodes in the second one.

    Args:
        graph (Graph): Starting graph.
        category1 (str): String name of the first input category (source nodes).
        category2 (str): String name of the second input category (destination nodes).

    Returns:
        Graph: Subgraph with source in cat1 and destination in cat2.
    """
    subgraph = Graph()

    nodes_in_cat1 = set(graph.cat_link_dict[category1])
    filtered_nodes1 = nodes_in_cat1.intersection(graph.nodes)

    nodes_in_cat2 = set(graph.cat_link_dict[category2])
    filtered_nodes2 = nodes_in_cat2.intersection(graph.nodes)

    for src in filtered_nodes1:
        dest_nodes = graph[src]
        for dst in dest_nodes:
            if dst not in filtered_nodes2:
                continue
            subgraph.add_edge(src, dst)

    return subgraph