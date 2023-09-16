import pandas as pd
import numpy as np
import networkx as nx
from ccna.cluster_metrics import *
import sys
from typing import Set, Tuple, List
from ccna.two_way_dict import create_country_name_lookup

"""
Class for creating a bipartite country-policy network

Params
base_dir (str) : ie. /Users/ME/Documents/CC_NA/
"""
class PolicyNet:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        subsidy, sub_country_nodes, sub_policy_nodes = self.load_subsidy_data()
        green_bonds, bond_country_nodes, bond_policy_nodes = self.load_green_bonds()
        taxes, tax_country_nodes, tax_policy_nodes = self.load_taxes()
        expenditures, exp_country_nodes, exp_policy_nodes = self.load_expenditures()

        self.G = nx.MultiGraph()
        country_node_attributes = self.create_country_node_attributes(
            tax_country_nodes, exp_country_nodes, bond_country_nodes, sub_country_nodes
        )
        self.G.add_nodes_from(country_node_attributes.keys())
        nx.set_node_attributes(self.G, country_node_attributes)

        policy_node_attributes = self.create_policy_node_attributes(
            tax_policy_nodes, exp_policy_nodes, bond_policy_nodes, sub_policy_nodes
        )
        self.G.add_nodes_from(policy_node_attributes.keys())
        nx.set_node_attributes(self.G, policy_node_attributes)

        self.tax_years = self.add_edges(taxes, tax_country_nodes, tax_policy_nodes)
        self.bond_years = self.add_edges(green_bonds, bond_country_nodes, bond_policy_nodes)
        self.subsidy_years = self.add_edges(subsidy, sub_country_nodes, sub_policy_nodes)
        self.exp_years = self.add_edges(expenditures, exp_country_nodes, exp_policy_nodes)

        self.two_way_country_lookup = create_country_name_lookup(base_dir)

    def _obtain_country_and_policies(self, df : pd.DataFrame) -> Tuple[Set, Set]:
        county_nodes = set(df["ISO2"].unique())
        policy_nodes = set(df["CTS_Name"].unique())
        return (county_nodes, policy_nodes)
    
    def load_subsidy_data(self):
        subsidy = pd.read_csv(f'{self.base_dir}input/policy/Fossil_Fuel_Subsidies.csv', index_col=0, keep_default_na=False, na_values="")
        subsidy = subsidy[subsidy.Unit == "Percent of GDP"]
        sub_country_nodes, sub_policy_nodes = self._obtain_country_and_policies(subsidy)
        sub_policy_nodes = {'Explicit; Coal','Explicit; Electricity','Explicit; Natural Gas','Explicit; Petroleum','Implicit; Accidents',
                           'Implicit; Coal','Implicit; Congestion','Implicit; Electricity','Implicit; Foregone Vat','Implicit; Global Warming',
                           'Implicit; Local Air Pollution','Implicit; Natural Gas','Implicit; Petroleum','Implicit; Road Damage'}
        return subsidy, sub_country_nodes, sub_policy_nodes

    def load_green_bonds(self):
        green_bonds = pd.read_csv(f"{self.base_dir}input/policy/Green_Bonds.csv", index_col=0, keep_default_na=False, na_values="")
        green_bonds = green_bonds[np.logical_not(green_bonds["ISO2"].isna())]
        green_bonds = green_bonds[green_bonds.Unit == "US Dollars"]
        bond_country_nodes, bond_policy_nodes = self._obtain_country_and_policies(green_bonds)
        return green_bonds, bond_country_nodes, bond_policy_nodes

    def load_taxes(self):
        taxes = pd.read_csv(f"{self.base_dir}input/policy/Environmental_Taxes.csv", index_col=0, keep_default_na=False, na_values="")
        taxes = taxes[taxes.Unit == "Percent of GDP"]
        tax_country_nodes, tax_policy_nodes = self._obtain_country_and_policies(taxes)
        return taxes, tax_country_nodes, tax_policy_nodes

    def load_expenditures(self):
        expenditures = pd.read_csv(f"{self.base_dir}input/policy/Environmental_Protection_Expenditures.csv", index_col=0, keep_default_na=False, na_values="")
        expenditures = expenditures[expenditures.Unit == "Percent of GDP"]
        exp_country_nodes, exp_policy_nodes = self._obtain_country_and_policies(expenditures)
        return expenditures, exp_country_nodes, exp_policy_nodes

    def create_country_node_attributes(self, tax_country_nodes, exp_country_nodes, bond_country_nodes, sub_country_nodes):
        all_countries = tax_country_nodes | exp_country_nodes | bond_country_nodes | sub_country_nodes
        country_node_attributes = {}
        for country in all_countries:
            policies = set()
            if country in tax_country_nodes:
                policies.add("Environmental Taxes")
            if country in exp_country_nodes:
                policies.add("Environmental Protection Expenditures")
            if country in bond_country_nodes:
                policies.add("Green Bonds")
            if country in sub_country_nodes:
                policies.add("Environmental Subsidies")

            country_node_attributes[country] = {"bipartite": 0, "policies": policies}

        return country_node_attributes

    def create_policy_node_attributes(self, tax_policy_nodes, exp_policy_nodes, bond_policy_nodes, sub_policy_nodes):
        all_policies = tax_policy_nodes | exp_policy_nodes | bond_policy_nodes | sub_policy_nodes
        policy_node_attributes = {}
        for policy in all_policies:
            policies = set()
            if policy in tax_policy_nodes:
                policies.add("Environmental Taxes")
            elif policy in exp_policy_nodes:
                policies.add("Environmental Protection Expenditures")
            elif policy in bond_policy_nodes:
                policies.add("Green Bonds")
            elif policy in sub_policy_nodes:
                policies.add("Environmental Subsidies")

            policy_node_attributes[policy] = {"bipartite": 1, "policies": policies}

        return policy_node_attributes
    
    """
    Adds the edge data from the dataframe into the graph. Returns the list of years that this policy is invested in
    Returns a list of years where there were policies implemented
    """
    def add_edges(self, df : pd.DataFrame, countries : Set[str], policies : Set[str]) -> List[int]:   
        # obtain the columns that correspond to the year
        years = df.filter(regex= "F\d\d\d\d", axis = 1).columns
        df_temp = df.set_index(keys = ["ISO2", "CTS_Name"])
        return_years = []
        
        edge_key = 0
        for year in years:
            year_edge_list = []
            for ind in df_temp.index:
                country = ind[0]
                policy = ind[1]
                if country not in countries or policy not in policies:
                    continue
                weight = df_temp.loc[ind].at[year]
                
                if weight > 0:
                    edge_data = {"weight" : weight, "year" : int(year[1:])}
                    
                    edge = (country, policy, edge_key, edge_data) #create edge
                    year_edge_list.append(edge)
                    edge_key += 1

            # Take out years with no policies
            if not (len(year_edge_list) == 0):         
                self.G.add_edges_from(year_edge_list)
                return_years.append(int(year[1:]))
        return return_years
    
    """
    Helper function to return the subgraph of G for the specificed year
    """
    def return_subgraph_year(self, G: nx.MultiGraph, year: int) -> nx.MultiGraph:
        sub_graph = G.copy()

        # Remove edges with years not equal to the specified year
        edges_to_remove = {(u, v, key) for u, v, key, data in sub_graph.edges(keys=True, data=True) if data["year"] != year}
        sub_graph.remove_edges_from(edges_to_remove)

        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = [node for node in sub_graph.nodes() if sub_graph.degree(node) == 0]
        sub_graph.remove_nodes_from(isolated_nodes)

        return sub_graph
    
    """
    Helper function to return the subgraph of G for the specificed policies
    """
    def return_subgraph_policy(self, G: nx.MultiGraph, filter_policies: set[str]) -> nx.MultiGraph:
        sub_graph = G.copy()
        nodes_to_remove = {node for node, data in sub_graph.nodes(data=True) if len(filter_policies & data["policies"]) == 0}
        sub_graph.remove_nodes_from(nodes_to_remove) # consequentially removes the edges
        return sub_graph
    
    """
    Return a dictionary of various metrics for the specified graph
    """
    def metrics(self, G : nx.MultiGraph):
        countries = country_count(G)
        policies = policy_count(G)
        cats = caterpillars(G)
        buts = butterflies(G)
        ratio = 0 if cats == 0 else buts / cats
        return {"Countries": countries, "Policies": policies, "Edges": G.number_of_edges(),
                "Butterflies": buts, "Caterpillars": cats, "Ratio": ratio}

    """
    Return a dataframe displaying various metrics for each year
    """
    def temporal_metrics(self, G : nx.MultiGraph, start_year : int, end_year : int):
        results = []
        for year in range(start_year, end_year + 1):
            subgraph = self.return_subgraph_year(G, year)
            results.append(self.metrics(subgraph))
        return pd.DataFrame(results)
    
    """
    Iterates through the temporal graph to find total influences and total policies
    """
    def influence_values(self, G : nx.MultiGraph):
        # dictionaries for storing any influence values
        total_policies = {}
        pairs = {}

        # loop through temporal and add influence values
        min_year = min(data["year"] for c, p, data in G.edges(data=True))
        max_year = max(data["year"] for c, p, data in G.edges(data=True))
        for year in range(min_year, max_year):
            I = {(c, p) for c, p, data in G.edges(data=True) if data["year"] == year} # influencers
            F = {(c, p) for c, p, data in G.edges(data=True) if data["year"] == year + 1} # followers

            # loop through edges in each I and F, track all influences
            for c_I, p_I in I:
                total_policies[c_I] = total_policies[c_I] + 1 if c_I in total_policies else 1
                for c_F, p_F in F:
                    if p_I == p_F and c_I != c_F and (c_F, p_F) not in I:
                        pairs[(c_I, c_F)] = pairs[(c_I, c_F)] + 1 if (c_I, c_F) in pairs else 1
        return pairs, total_policies
    
    """
    Based on influence values, constructs a Directed Graph of influence
    """
    def influence_graph(self, G : nx.MultiGraph):
        pairs, total_policies = self.influence_values(G)

        edges = set()
        for i, f in pairs: #find all influencer-follower edges S_ij
            weight = pairs[(i, f)] / total_policies[i] #divide each influence by total number of policies Q_i
            edge = (i, f, weight)
            edges.add(edge)
        
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        return G