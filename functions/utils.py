import numpy as np
from collections import defaultdict

# For reproducibility
np.random.seed(42)

# Preprocess 'wiki-topcats-categories.txt' to extract one single category for each link
def links_per_cat(line):
    category, links = line.split(';')
    category = category.split(':')[1]
    links = links.strip().split(' ')
    try:
        links = [int(link) for link in links]
    except:
        return None
    return category, links

def link_category_dict():
    link_cat = defaultdict(list)

    with open('data/wiki-topcats-categories.txt') as file:
        for line in file:
            line = line.strip()
            if links_per_cat(line):
                category, links = links_per_cat(line)
                for link in links:
                    link_cat[link].append(category)
    
    return link_cat

def unique_category(link_cat):
    for link, cats in link_cat.items():
        size = len(cats)
        if size == 1:
            link_cat[link] = cats[0]
        else:
            rand_number = np.random.randint(low=0, high=size)
            rand_choice = cats[rand_number]
            link_cat[link] = rand_choice
            
    # Get the inverse dictionary, having categories as keys and links as values
    cat_link = defaultdict(list)
    for link, cat in link_cat.items():
        cat_link[cat].append(link)
            
    with open('data/wiki-topcats-categories-processed.txt', 'a') as file:
        for cat, links in cat_link.items():
            links = ' '.join([str(link) for link in links])
            file.write(cat + ': ' + links + '\n')
            
    return cat_link