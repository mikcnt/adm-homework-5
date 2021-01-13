# Homework 5 - Explore Wikipedia's hyperlinks network
## Task
The goal of this homework was to become familiar with graphs. We were also expected to write important algorithms from scratch. In our case, we wrote the implementation for modified versions of the breadth-first search and the depth-first search, used both to explore the graph, both to compute shortest paths. Finally, we were challenged to analyze Wikipedia categories, initially by computing the distances between them. As a second step, we were also asked to implement the PageRank algorithm, to compare the PR scores of the categories.
## Usage
### Installation
In the repository, it is included `requirements.txt`, which consists in a file containing the dependencies list. These can be installed using pip (even if working on a conda environment):

`pip install -r requirements.txt`
### Data folder
The data files are obviously obscured by a `.gitignore` file. To execute the notebook without any kind of modification in the code, it should be sufficient to move the data files to the directory `data`. In order to work, the project folder should at least contain the following files in the shown order:
```
├── data
│   ├── wikigraph_reduced.csv
│   ├── wiki-topcats-categories.txt
│   └── wiki-topcats-page-names.txt
├── functions
│   ├── graph.py
│   └── utils.py
└── main.ipynb
```

## Repo structure
The repository consists of the following files:
* __`main.ipynb`__:
    > This is the core of this repository. In fact it contains the results of our implementations and researches. Notice that there are actually very few lines of code in this notebook, the rest is contained in the functions directory.
* __`functions`__:
    > This directory contains the implementation of the functions we call in the main Jupyter Notebook file. Inside it we can find two files: __`graph.py`__ is a module containing all the scripts used to work with graphs, while __`utils.py`__ is a set of utility functions, mainly used during the preprocess part.
* __`data`__:
    > This directory should contain the actual data used in the algorithms. For obvious reasons, the main data files are obscured by the .gitignore and just the preprocessed file generated during the preprocessing state has been uploaded.
* __`requirements.txt`__:
    > A txt file containing the dependecies of the project; see the usage part for details.