import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import networkx as nx


# For reproducibility
np.random.seed(42)

# Preprocess 'wiki-topcats-categories.txt' to extract one single category for each link
def links_per_cat(line):
    category, links = line.split(";")
    category = category.split(":")[1]
    links = links.strip().split(" ")
    try:
        links = [int(link) for link in links]
    except:
        return None
    return category, links


def link_category_dict():
    link_cat = defaultdict(list)

    with open("data/wiki-topcats-categories.txt") as file:
        for line in file:
            line = line.strip()
            if links_per_cat(line):
                category, links = links_per_cat(line)
                for link in links:
                    link_cat[link].append(category)

    return link_cat


def category_link_dict(link_cat_dict):
    # Extract one random category for each link
    for link, cats in link_cat_dict.items():
        size = len(cats)
        if size == 1:
            link_cat_dict[link] = cats[0]
        else:
            rand_number = np.random.randint(low=0, high=size)
            rand_choice = cats[rand_number]
            link_cat_dict[link] = rand_choice

    # Get the inverse dictionary, having categories as keys and links as values
    cat_link_dict = defaultdict(list)
    for link, cat in link_cat_dict.items():
        cat_link_dict[cat].append(link)

    return cat_link_dict


def write_category_processed(cat_link_dict):
    with open("data/wiki-topcats-categories-processed.txt", "a") as file:
        for cat, links in cat_link_dict.items():
            links = " ".join([str(link) for link in links])
            file.write(cat + ": " + links + "\n")


def categories_in_graph():
    cat_link_dict = {}
    with open("data/wiki-topcats-categories-processed.txt") as f:
        for line in f:
            category, links = line.split(":")
            links = links.strip()
            links = links.split()
            links = [int(link) for link in links]
            cat_link_dict[category] = links

    return cat_link_dict


def graph_example():
    """Draw directed graph example. Used to show `min_edge_cut` algorithm rationale."""
    edges = [
        (1, 2),
        (1, 3),
        (1, 6),
        (1, 7),
        (2, 4),
        (3, 4),
        (6, 8),
        (7, 8),
        (4, 5),
        (8, 5),
        (1, 5),
    ]

    pos = {
        1: (2, 6),
        2: (4, 10),
        3: (4, 8),
        4: (8, 9),
        5: (10, 6),
        6: (4, 4),
        7: (7, 5),
        8: (8, 3),
    }

    G = nx.DiGraph()
    G.add_edges_from(edges)
    plt.figure(figsize=(10, 5))
    nx.draw(G, pos, with_labels=True)


def categories_size(graph, threshold=1000):
    table = []
    for category in graph.cat_link_dict.keys():
        size = len(graph.nodes_in_category(category))
        if size > threshold:
            table.append([category, size])

    table = sorted(table, key=lambda x: x[1], reverse=True)

    max_lenght = max([len(x[0]) for x in table])

    width = max_lenght + 3

    for category, size in table:
        print("{} | {}".format(category.ljust(width), str(size).ljust(width)))