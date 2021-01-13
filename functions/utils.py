import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import networkx as nx


# For reproducibility
np.random.seed(42)


def links_per_cat(line):
    """Given a line of the "wiki-topcats-categories.txt" file,
    extract the category and the links (first argument is the category,
    following ones are the links).

    Args:
        line (str): String containing category and links. Each line is in the form:
                    "Category;category_name: link_1 link_2 ..."

    Returns:
        str, list: Tuple containing string for the category and list of ints for the links.
    """
    category, links = line.split(";")
    category = category.split(":")[1]
    links = links.strip().split(" ")
    try:
        links = [int(link) for link in links]
    except:
        return None
    return category, links


def link_cat():
    """Computes the link-category dictionary iterating on the lines of
    the wiki-topcats-categories.txt file.

    Returns:
        dict: Dictionary in the form {link1: [cat1, cat2, ...], ...}.
    """
    link_cat_dict = defaultdict(list)

    with open("data/wiki-topcats-categories.txt") as file:
        for line in file:
            line = line.strip()
            if links_per_cat(line):
                category, links = links_per_cat(line)
                for link in links:
                    link_cat_dict[link].append(category)

    return link_cat_dict


def cat_link():
    """Computes the category-link dictionary iterating on the lines of
    the wiki-topcats-categories.txt file.

    Returns:
        dict: Dictionary in the form {cat1: [link1, link2, ...], ...}.
    """
    cat_link_dic = defaultdict(list)

    with open("data/wiki-topcats-categories.txt") as file:
        for line in file:
            line = line.strip()
            if links_per_cat(line):
                category, links = links_per_cat(line)
                cat_link_dic[category] = links

    return cat_link_dic


def unique_link_cat(link_cat_dict, cat_link_dict):
    """Computes the dictionary having one single category for link.

    Args:
        link_cat_dict (dict): Dictionary containing link as keys and categories as values.
        cat_link_dict (dict): Dictionary containing category as keys and links as values.

    Returns:
        dict: Dictionary containing link as keys and unique category as values.
    """
    unique_link_cat_dict = {}
    cat_link_dict = cat_link()
    invalid_categories = {
        cat
        for cat, links in cat_link_dict.items()
        if len(links) < 5000 or len(links) > 30000
    }

    # Extract one random category for each link
    for link, cats in link_cat_dict.items():
        cats = [cat for cat in cats if cat not in invalid_categories]
        size = len(cats)
        # Ignore the category if it has zero links
        if size == 0:
            continue
        # If there is just one category, select it
        elif size == 1:
            unique_link_cat_dict[link] = cats[0]
        # Otherwise select random category from the list
        else:
            rand_number = np.random.randint(low=0, high=size)
            rand_choice = cats[rand_number]
            unique_link_cat_dict[link] = rand_choice

    return unique_link_cat_dict


def unique_cat_link(unique_link_cat_dict):
    """Computes the reverse dictionary for the one that maps unique categories to links.

    Args:
        unique_link_cat_dict (dict): Dictionary containing link as keys and unique category as values.

    Returns:
        dict: Dictionary mapping each category to its links. Each link has a unique category.
    """
    # Get the inverse dictionary, having categories as keys and links as values
    uniq_cat_link = defaultdict(list)
    for link, cat in unique_link_cat_dict.items():
        uniq_cat_link[cat].append(link)

    return uniq_cat_link


def write_category_processed(unique_cat_link_dict):
    """Creates txt file containing the unique-mapping between categories and links.

    Args:
        unique_cat_link_dict (dict): Dictionary containing the unique-mapping
                                     between categories and links.
    """
    with open("data/wiki-topcats-categories-processed.txt", "a") as file:
        for cat, links in unique_cat_link_dict.items():
            links = " ".join([str(link) for link in links])
            file.write(cat + ": " + links + "\n")


def read_cat_link():
    """Reads the mapping dictionary {cat: link} from the preprocessed file.
    Use if the preprocessed file has already been created.

    Returns:
        dict: Dictionary containing the {cat: link} unique mapping.
    """
    cat_link_dict = {}
    with open("data/wiki-topcats-categories-processed.txt") as f:
        for line in f:
            category, links = line.split(":")
            links = links.strip()
            links = links.split()
            links = [int(link) for link in links]
            cat_link_dict[category] = links

    return cat_link_dict

def read_link_cat():
    link_cat_dict = {}

    with open('data/wiki-topcats-categories-processed.txt') as file:
        for line in file:
            line = line.strip()
            category, links = line.split(':')
            links = links.split()
            links = [int(link) for link in links]
            for link in links:
                link_cat_dict[link] = category

    return link_cat_dict


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


def categories_size(graph):
    """Returns table containing the number of nodes for category of a graph.

    Args:
        graph (Graph): Input graph.

    Returns:
        list: List of tuples in the form (cat, size).
    """
    table = []
    for category in graph.cat_link_dict.keys():
        size = len(graph.nodes_in_category(category))
        table.append((category, size))

    # table = sorted(table, key=lambda x: x[1], reverse=True)
    return table


def generate_table(table, title="", amount='all', sort_key=lambda x: x[1], reverse=True):
    """Prints a pretty table from a list of tuples/lists.

    Args:
        table (list): List of tuples/lists in the form (name, attribute).
        title (str, optional): Title of the table. Defaults to "". If empty, no title is shown.
        amount (int or 'all', optional): Amount of rows of the table to print.
                                         If 'all', all the rows are printed. Defaults to 'all'.
        sort_key (function, optional): Sorting key for the table.
                                       Defaults to lambda x: x[1] (sort by score).
        reverse (bool, optional): If False, results are sorted from lowest to highest. Defaults to True.
    """
    if sort_key is not None:
        table = sorted(table, key=sort_key, reverse=reverse)
    if amount != 'all':
        table = table[:amount]
    max_lenght = max([len(x[0]) for x in table])
    max_lenght1 = max([len(str(x[1])) for x in table])

    width = max_lenght + 3
    all_width = width + max_lenght1 + 3
    if title:
        centered_title = title.center(all_width)
        print(centered_title)
    print("-" * all_width)
    for category, score in table:
        print("{} | {}".format(category.ljust(width), str(score).ljust(width)))