import pandas as pd
import numpy as np
import networkx as nx

# number of countries with positive contribution
def country_count(G):
    total = 0
    for node, data in G.nodes(data=True):
        if data["bipartite"] == 0 and G.degree(node) > 0:
            total += 1
    return total

# number of policies which are being implemented
def policy_count(G):
    total = 0
    for node, data in G.nodes(data=True):
        if data["bipartite"] == 1 and G.degree(node) > 0:
            total += 1
    return total

# returns a list of every pair of edges that of which country_node is the source
def calculate_pairs(G, country_node):
    policies = [p for c, p in G.edges(country_node)]
    N = len(policies)
    pairs = []
    for i in range(N):
        for j in range(N):
            if i > j:
                pairs.append((policies[i], policies[j]))
    return pairs

# given that pairs can begin with either node, maps like pairs to the same value
def pair_mapper(policy_nodes):
    mapper = {}
    val = 0
    for i in policy_nodes:
        for j in policy_nodes:
            if (j, i) in mapper:
                mapper[(i, j)] = mapper[(j, i)]
            else:
                mapper[(i, j)] = val
                val += 1
    return mapper

# returns the butterfly count in a graph G, defined above
def butterflies(G):
    # initialize
    country_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    policy_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    # get counts for each pair of policies simultaneously implemented
    pair_vals = []
    pair_map = pair_mapper(policy_nodes)
    for country in country_nodes:
        for pair in calculate_pairs(G, country):
            pair_vals.append(pair_map[pair])

    # calculate butterflies
    butterflies = 0
    for val in pd.Series(pair_vals).value_counts():
        butterflies += val * (val - 1) // 2
    return butterflies

# returns the caterpillar count in graph G, defined above
def caterpillars(G):
    # initialize
    country_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    policy_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    # get counts for each pair of policies simultaneously implemented
    caterpillars = 0
    for country in country_nodes:
        for a, b in calculate_pairs(G, country):
            caterpillars += (G.degree(a) + G.degree(b) - 2)
    
    return caterpillars