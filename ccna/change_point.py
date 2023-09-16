from typing import Callable
import networkx as nx
import numpy as np
from typing import Set, Tuple, List

def standard_laplacian(G: nx.MultiGraph) -> np.ndarray:
    return nx.normalized_laplacian_matrix(G).toarray()

def signless_laplacian(G:  nx.MultiGraph) -> np.ndarray:
    A = nx.adjacency_matrix(G)
    D = np.diag(nx.laplacian_matrix(G).diagonal())
    return A + D

# TODO: Ask Dr. Srini about this.
def normalized_signless_laplacian(G:  nx.MultiGraph) -> np.ndarray:

    laplac = signless_laplacian(G)
    
    max_value = np.max(laplac)
    return laplac / max_value
    # D = np.diag(nx.laplacian_matrix(G).diagonal())
    # squared_D = np.linalg.matrix_power(D, 2)
    # inv_D = np.linalg.inv(squared_D)
    
    # norm_laplac = inv_D @ laplac @ inv_D

    # return norm_laplac

# TODO: Add citation to the paper I got this information from
def compute_year_signature(G : nx.MultiGraph, laplacian_func: Callable) -> np.ndarray:
    """_summary_

    Args:
        G (nx.MultiGraph): A Multigraph subgraph that depicts a specific year of a graph.
        
        laplacian_func (Callable): Either the normal laplacian or the signless laplacian

    Returns:
        np.ndarray: Returns a numpy array containing a numeric signature of a year
    """
    
    laplacian = laplacian_func(G)
    singular_values = np.linalg.svd(laplacian)[1]
    l2_norm = np.linalg.norm(singular_values, ord = 2)
    norm_singluar_values = singular_values / l2_norm
    
    return norm_singluar_values

"""
Given series of network graphs

Args:
    windowSize (int): _description_
    G (MultiGraph): A networkx Multigraph that has a single policy attribute across all nodes
    years (List[int]): A list of years to compute change detection over

Returns:
    _type_: _description_
"""
def change_point_detection(windowSize: int, G : nx.MultiGraph, years : List[int], laplacian_func: Callable) -> Tuple[List[float], List[int]]:

    dict_signatures = {}
    # Compute the signatures for each person
    for year in years:
        sub_G = return_subgraph_year(G, year)
        dict_signatures[year] = compute_year_signature(sub_G, laplacian_func)


    window_dicts = {}
    for i in range(len(years) - windowSize):
        year_subset = years[i:i+windowSize]

        weights_avg_list = []

        for year in year_subset:
            weights_avg_list.append(dict_signatures[year])
    
        weights_avg = np.stack(weights_avg_list, axis = 1).mean(axis = 1)

        window_dicts[years[windowSize + i]] = weights_avg / np.linalg.norm(weights_avg, ord = 2)

    Z_list = []
    year_list = []

    for year in window_dicts.keys():
        baseline = window_dicts[year]
        single_year = dict_signatures[year]

        Z = 1-np.transpose(single_year) @ baseline

        Z_list.append(Z)
        year_list.append(year)

    return (Z_list, year_list)
    