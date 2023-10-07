# CC_NA - Climate Change Network Analysis
This repository contains several methods and analyses for historical climate change policy data.
Our main model splits countries and policies into a bipartite graph where edges represent the participation of countries in specific policies.
From that, we will try out the following analysis methods:
- Temporally calculating various graphwide metrics (from 1985-present) (TODO - copy source)
- Change point analysis (TODO - copy source)
- Converting bipartite model into an influence graph to calculate influence-passivity values (TODO - copy source)

### Setup

Start by creating a virtual environment and installing all the required packages. In commandline:

```
python3 -m venv env
pip install -r requirements.txt
```

The package is set up for the 4 policy dataframes in the `input/policy/` directory.
Feel free to try use your own data (will require preprocessing) with the various network science tools implemented here.

### Codebase

The entire codebase is build inside the `ccna/` directory. Several of the key features:
1. The "PolicyNet" class is responsible for building our bipartite graph, calculating temporal metrics, and constructing our influence graph.
2. The "IP" class takes the influence graph and calculates our influence-passivity values (in a Pandas dataframe).
3. The `ccna/change_point.py` script contains all methods for change point detection.

### Analyses

There are several notebooks which reflect our analyses at different points throughout the project.
- `notebooks/01 - final report.ipynb` contains our analyses from the final project report of CSE 5245: Introduction to Network Science taught by Professor Srinivasan Parthasarathy at The Ohio State University. This notebook shows much of the data processing steps, ideas behind the method implementation, and more.
- `notebooks/02 - change point detection.ipynb` documents some of our main structural changes in building the graph, cleaning up readability issues from before, and furthering the change point detection analyses.
- `notebooks/03 - walkthrough.ipynb` goes step-by-step into how to use the package based on its current implementation. Much of the behind the scenes work is obscured in this notebook, as it is more so tailored to current usage.
