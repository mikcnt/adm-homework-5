import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from collections import deque
from .utils import read_cat_link


class Graph(object):
    """Class for directed and undirected graph.

    Attributes:
        cat_link_dict (dict): Dictionary containing categories as keys and the links for each category as values.
        categories (list): List of all the possible categories. Even empty categories are kept.
        edges (defaultdict): Edges of the graph, in the form of src (key) -> dst (values, can be more than 1).
        nodes (set): Set containing the whole list of nodes of the graph.
        in_neighbours (defaultdict): Dictionary containing the in-neighbours of a node.
        out_neighbours (defaultdict): Dictionary containing the out-neighbours of a node.
        edges_num (int): Number of edges in the graph.

    """

    def __init__(self):
        self.cat_link_dict = read_cat_link()
        self.categories = self.retrieve_categories()
        self.edges = defaultdict(set)
        self.nodes = set()
        self.in_neighbours = defaultdict(set)
        self.out_neighbours = defaultdict(set)
        self.edges_num = 0

    def __getitem__(self, node):
        """Item getter. Returns the out-neighbours of a node.

        Args:
            node (int): Input node.

        Raises:
            KeyError: Error is raised if there is no such node in the graph.

        Returns:
            list: List of nodes for which there is an edge node -> n.
        """
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
        self.edges_num += 1
        # Add the actual edge
        self.edges[src].add(dst)
        # Add both source and dest nodes to the set of nodes
        self.nodes.update([src, dst])
        # Keep track of the neighbours of each node in the graph
        self.in_neighbours[dst].add(src)
        self.out_neighbours[src].add(dst)

    def add_edges_from(self, edges):
        """Add a list of edge to the existing graph.

        Args:
            edges (list): List of tuples/lists in the form (src, dst),
                          representing edges.
        """
        for src, dst in edges:
            self.add_edge(src, dst)

    def is_directed(self):
        """Check if the graph is directed. Graph is considered directed
        if there is an inverse path for each edge of the graph."""
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
        """Returns the number of nodes in the graph (even those without edges)."""
        return len(self.nodes)

    def number_of_edges(self):
        """Returns the number of edges in the graph."""
        return self.edges_num

    def density(self):
        """The density of a graph is defined as follows:
        - 2*E / (|V| * (|V| - 1)) if the graph is undirected;
        - |E| / (|V| * (|V| - 1)) if the graph is directed.

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

    def shortest_path(self, src, dst):
        """Returns the shortest path between a starting node and a target node.

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
            for neighbour in self.edges[current]:
                # Stopping criteria: when path is found
                if neighbour == dst:
                    return path + [current, neighbour]
                if neighbour in visited:
                    continue
                queue.append((neighbour, path + [current]))
                visited.add(neighbour)
        # We reach this part only if there is no path between vertices
        return None

    def get_distances(self, src):
        """Given a node src, returns the distances between src and all the other nodes of the graph.

        Args:
            src (int): Source node from which we compute distancies.

        Returns:
            dict: Dictionary containing all the distancies.
        """
        # Distances are going to be kept in the form
        # {node: distance between src and node}
        distances = {src: 0}
        queue = deque([src])
        while queue:
            start = queue.popleft()
            # Current distance is +1 from before
            dist = distances[start] + 1
            # next end nodes
            for end in self.edges[start]:
                # Skip nodes for which we already have computed the distance
                if end in distances:
                    continue
                # Set distances
                distances[end] = dist
                queue.append(end)
        return distances

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
            path = self.shortest_path(most_central, node)
            if not path:
                print("Not possible")
                return
            distances.append(len(path))

        # We return the max of these distances, since it is the
        # value for which we are sure we hit all the nodes in `pages`
        return max(distances)

    def retrieve_categories(self):
        """Returns all the categories from the given data file (even empty ones)."""
        return list(self.cat_link_dict.keys())

    def category_to_id(self, category):
        """Convers a category string name to a unique integer id."""
        return self.categories.index(category)

    def id_to_category(self, cat_id):
        """Converts back a category id to its string name."""
        return self.categories[cat_id]

    def nodes_in_category(self, category):
        """Returns the nodes in the category actually contained in the graph nodes.

        Args:
            category (str): String name of the input category.

        Returns:
            set: Set containing the graph nodes contained in the category.
        """
        return set(self.cat_link_dict[category]).intersection(self.nodes)


def induced_subgraph(graph, category1, category2):
    """Create subgraph containing the edges with nodes in category1 or category2.

    Args:
        graph (Graph): Starting graph.
        category1 (str): String name of the first input category.
        category2 (str): String name of the second input category.

    Returns:
        Graph: Subgraph with source in cat1 and destination in cat2.
    """
    # Find the nodes in each of the category
    # The subgraph will have the union of these nodes as nodes
    cat1_nodes = graph.nodes_in_category(category1)
    cat2_nodes = graph.nodes_in_category(category2)
    union_nodes = cat1_nodes.union(cat2_nodes)
    # The edges of the subgraph will be the ones that starts either in cat1 or cat2
    # and end either in cat1 or cat2
    edges = []
    for node in union_nodes:
        for neigh in graph[node]:
            if neigh in union_nodes:
                edges.append((node, neigh))

    # Add the edges and update the nodes (some node could have 0 edges
    # and therefore it could have been skipped when adding the edges)
    subgraph = Graph()
    subgraph.add_edges_from(edges)
    subgraph.nodes.update(union_nodes)

    return subgraph


class DisjointPaths(object):
    """Class to compute disjoint paths between nodes of a graph.

    Attributes:
        graph (Graph): Input graph.
        paths (list): List containing the disjoint paths.
        visited (dict): Dictionary used to check if an edge has been traversed.

    """

    def __init__(self, graph):
        self.graph = graph

    def search(self, src, dst):
        """Main function externally used. Exploits the private function to retrieve
        and return the disjoint paths.

        Args:
            src (int): Source node.
            dst (int): Target node.

        Returns:
            list: List containing the disjoint paths from src to dst.
        """
        self.paths = []
        self.visited = defaultdict(bool)
        self.__search(src, dst, src, [src])
        return self.paths

    def __search(self, src, dst, node, actual_path):
        """Helper private function used to compute the disjoint paths between source and node.
        Being a recursive function, we keep note of the node we're visiting, and the path
        we're traversing.

        Args:
            src (int): Source node.
            dst (int): Target node.
            node (int): Node on which we call the recursion.
            actual_path (list): List containing the path we're traversing.
        """
        # We use a copy to avoid inplace modifications during each
        # recursive call
        path = actual_path.copy()
        # Once we find the destination, we can add the path to the list
        # of disjoint paths and stop the recursion
        if node == dst:
            self.paths.append(path)
            return

        # Iterate on the neighbours of the actual node
        for neigh in self.graph[node]:
            # As path, create a new branch for each neighbour
            new_path = path.copy()
            # Check if the edge has already been visited
            if self.visited[(node, neigh)]:
                continue
            # Otherwise set the flag to True
            self.visited[(node, neigh)] = True
            # Store the path
            new_path.append(neigh)
            # Call again the function to go in depth
            self.__search(src, dst, neigh, new_path)
        # Once we've concluded all the calls, just exit
        return


def min_edge_cut(graph, src, dst):
    """Computes the minimum number of edges to remove to disconnect two nodes of a graph.

    Args:
        graph (Graph): Input graph.
        src (int): Starting node.
        dst (int): Destination node.

    Returns:
        int: Minimum number of cuts to be made on the edges in order to disconnect the two nodes.
        0 if the nodes are not connected.
    """
    d = DisjointPaths(graph)
    disjoint_paths = d.search(src, dst)
    return len(disjoint_paths)


def distances_from_category(graph, cat, unique_cat_dict):
    """Computes the ordered distances of all the categories between one central category.
    Distances are computed as follows:
    dist(c1, c2) = median([shortest_path(n1, n2) for every pair (n1, n2) s.t. n1 in c1 and n2 in c2]).

    Args:
        graph (Graph): Input graph.
        cat (str): Central category for which we want the distances.
        unique_cat_dict (dict):

    Returns:
        list: List of tuples in the form (category, distance from central category).
              All the distances are kept except the distance between `cat` and itself.
    """
    # Extract the nodes in the category
    cat_nodes = graph.nodes_in_category(cat)
    distances = {}
    # Compute the distances between every node of the central category and
    # every other node of the graph
    # Distances is going to be in the form node -> {u: dist from u, v: dist from v, ...}
    for node in cat_nodes:
        distances[node] = graph.get_distances(node)

    # cat_dist = {c1: [dist from u, dist from v, ...], c2: [...], ...}
    # where u, v, ... are nodes of c1
    cat_dist = defaultdict(list)
    # u are the nodes in the central category
    for u in distances:
        for v in distances[u].keys():
            # Compute the category where v belongs
            v_cat = unique_cat_dict[v]
            # For all the nodes in the central category, store the distance between
            # them and the nodes in each other category
            cat_dist[v_cat].append(distances[u][v])

    # Distance between category and itself
    del cat_dist[cat]

    # For each category, retrieve the distance using the median of all the lenght
    # of the shortest paths
    fin_dist = {}
    for cat, path_len in cat_dist.items():
        fin_dist[cat] = np.median(path_len)

    # Sort the distances and return them
    return [(cat, dist) for cat, dist in fin_dist.items()]


def create_category_graph(graph, unique_cat_dict):
    """Create new graph that connects the categories of the starting graph.
    For example, if the starting graph has an edge 1 -> 2, then the new graph
    will have an edge category(1) -> category(2)

    Args:
        graph (Graph): Starting graph.
        unique_cat_dict (dict): Dictionary mapping each link to its category.

    Returns:
        Graph: Graph containing categories (ids) as nodes and links between categories as edges.
        N.B.: if there are two nodes pointing from category A to category B, only one is kept.
    """
    cat_graph = Graph()
    # category is just an alias to retrieve the category (id) for a given node
    category = lambda x: graph.category_to_id(unique_cat_dict[x])
    # Iterate through all the edges of the starting graph and add the edges to the new graph
    for v, neighbours in graph.edges.items():
        v_cat = category(v)
        for n in neighbours:
            n_cat = category(n)
            cat_graph.add_edge(v_cat, n_cat)
    return cat_graph


def pagerank(graph, d=0.85, max_iter=100, tol=1.0e-6):
    """Computes the PageRank score for each node of the graph.

    Args:
        graph (Graph): Input graph.
        d (float, optional): Damping factor. Defaults to 0.85.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance factor used to check convergence. Defaults to 1.0e-6.

    Returns:
        dict: Dictionary containing the PageRank score for each node of the graph.
    """
    # If the graph is empty
    if len(graph.nodes) == 0:
        return {}

    N = len(graph.nodes)

    # This will be useful when computing the denominator in the PR calculation
    out_degree = graph.out_degree

    # Iteration 0: initialize pagerank
    # PR_0(n_i) = 1 / N for every n_i in the graph
    x = dict.fromkeys(graph.nodes, 1.0 / N)

    dangling_weights = dict.fromkeys(graph.nodes, 1.0 / N)

    # Dangling nodes are the nodes without outgoing links
    dangling_nodes = [n for n in graph.nodes if out_degree(n) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        # Keep the pagerank scores of the last iteration
        xlast = x
        # New pagerank scores start at 0
        # For each node, we sum the PR of its neighbours divided by their out-degree
        x = dict.fromkeys(xlast.keys(), 0)
        # We first compute the score for the dangling nodes
        # Which will be part of the computation of the score for all the other nodes
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for in_n in graph.in_neighbours[n]:
                x[n] += d * xlast[in_n] / out_degree(in_n)
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * dangling_weights[n]
        # Check for convergence
        # (difference between new PR and last PR is < than tolerance for each node)
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    # If this isn't the case, PR didn't converge
    print("Pagerank didn't converge in {}.".format(max_iter))
    return


def dict_to_table(graph, pr_dict):
    """Utility function used to generate a list of tuples from 
    the dictionary of the pagerank scores."""
    cat_name = graph.id_to_category

    return [(cat_name(cat_id), score) for cat_id, score in pr_dict.items()]
