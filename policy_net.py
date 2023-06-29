import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from cluster_metrics import country_count, butterflies, caterpillars
import matplotlib.pyplot as plt
import sys
from typing import Set, Tuple, List

# Creates a Two-way look up table
class TwoWayDict(dict):
    
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2
    
    def from_dict(self, user_dict):
        for key, value in user_dict.items():
            dict.__setitem__(self, key, value)
            dict.__setitem__(self, value, key)

class PolicyNet:

    def __init__(self):
        subsidy, sub_country_nodes, sub_policy_nodes = self.load_subsidy_data()
        green_bonds, bond_country_nodes, bond_policy_nodes = self.load_green_bonds()
        taxes, tax_country_nodes, tax_policy_nodes = self.load_taxes()
        expenditures, exp_country_nodes, exp_policy_nodes = self.load_expenditures()

        G = nx.MultiGraph()
        country_node_attributes = self.create_country_node_attributes(
            tax_country_nodes, exp_country_nodes, bond_country_nodes, sub_country_nodes
        )
        G.add_nodes_from(country_node_attributes.keys())
        nx.set_node_attributes(G, country_node_attributes)

        policy_node_attributes = self.create_policy_node_attributes(
            tax_policy_nodes, exp_policy_nodes, bond_policy_nodes, sub_policy_nodes
        )
        G.add_nodes_from(policy_node_attributes.keys())
        nx.set_node_attributes(G, policy_node_attributes)

        tax_years = self.add_edges(G, taxes, tax_country_nodes, tax_policy_nodes)
        bond_years = self.add_edges(G, green_bonds, bond_country_nodes, bond_policy_nodes)
        subsidy_years = self.add_edges(G, subsidy, sub_country_nodes, sub_policy_nodes)
        exp_years = self.add_edges(G, expenditures, exp_country_nodes, exp_policy_nodes)

        country_name_conversion = self.load_country_name_conversion()
        two_way_country_lookup = self.create_country_name_lookup(country_name_conversion)
    
    def load_subsidy_data(self):
        subsidy = pd.read_csv('input/policy/Fossil_Fuel_Subsidies.csv', index_col=0, keep_default_na=False, na_values="")
        sub_country_nodes, sub_policy_nodes = self.obtain_country_and_policies(subsidy)
        return subsidy, sub_country_nodes, sub_policy_nodes

    def load_green_bonds(self):
        green_bonds = pd.read_csv("input/policy/Green_Bonds.csv", index_col=0, keep_default_na=False, na_values="")
        green_bonds = green_bonds[np.logical_not(green_bonds["ISO2"].isna())]
        bond_country_nodes, bond_policy_nodes = self.obtain_country_and_policies(green_bonds)
        return green_bonds, bond_country_nodes, bond_policy_nodes

    def load_taxes(self):
        taxes = pd.read_csv("input/policy/Environmental_Taxes.csv", index_col=0, keep_default_na=False, na_values="")
        tax_country_nodes, tax_policy_nodes = self.obtain_country_and_policies(taxes)
        return taxes, tax_country_nodes, tax_policy_nodes

    def load_expenditures(self):
        expenditures = pd.read_csv("input/policy/Environmental_Protection_Expenditures.csv", index_col=0, keep_default_na=False, na_values="")
        exp_country_nodes, exp_policy_nodes = self.obtain_country_and_policies(expenditures)
        return expenditures, exp_country_nodes, exp_policy_nodes

    def create_country_node_attributes(self, tax_country_nodes, exp_country_nodes, bond_country_nodes, sub_country_nodes):
        all_countries = tax_country_nodes | exp_country_nodes | bond_country_nodes | sub_country_nodes
        country_node_attributes = {}
        for country in all_countries:
            policies = []
            if country in tax_country_nodes:
                policies.append("Environmental Taxes")
            if country in exp_country_nodes:
                policies.append("Environmental Protection Expenditures")
            if country in bond_country_nodes:
                policies.append("Green Bonds")
            if country in sub_country_nodes:
                policies.append("Environmental Subsidies")

            country_node_attributes[country] = {"bipartite": 0, "policies": policies}

        return country_node_attributes

    def create_policy_node_attributes(self, tax_policy_nodes, exp_policy_nodes, bond_policy_nodes, sub_policy_nodes):
        all_policies = tax_policy_nodes | exp_policy_nodes | bond_policy_nodes | sub_policy_nodes
        policy_node_attributes = {}
        for policy in all_policies:
            policies = []
            if policy in tax_policy_nodes:
                policies.append("Environmental Taxes")
            elif policy in exp_policy_nodes:
                policies.append("Environmental Protection Expenditures")
            elif policy in bond_policy_nodes:
                policies.append("Green Bonds")
            elif policy in sub_policy_nodes:
                policies.append("Environmental Subsidies")

            policy_node_attributes[policy] = {"bipartite": 1, "policies": policies}

        return policy_node_attributes

    def load_country_name_conversion(self):
        country_name_conversion = pd.read_csv("input/wikipedia-iso-country-codes.csv")
        country_name_conversion.rename(columns={"English short name lower case": "Country Name", "Alpha-2 code": "ISO2", "Alpha-3 code": "ISO3"}, inplace=True)
        return country_name_conversion

    def create_country_name_lookup(self, country_name_conversion):
        two_way_country_lookup = TwoWayDict()
        oneway_lookup = country_name_conversion.set_index("Country Name")["ISO2"].to_dict()
        two_way_country_lookup.from_dict(oneway_lookup)
        return two_way_country_lookup


    def obtain_country_and_policies(self, df : pd.DataFrame) -> Tuple[Set, Set]:
        county_nodes = set(df["ISO2"].unique())
        policy_nodes = set(df["CTS_Name"].unique())
        
        return (county_nodes, policy_nodes)
    
    """
    Adds the edge data from the dataframe into the graph. Returns the list of years that this policy is invested in
    Returns a list of years where there were policies implemented
    """
    def add_edges(self, G : nx.MultiGraph, df : pd.DataFrame, countries : Set[str], policies : Set[str]) -> List[int]:   
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
                weight = df_temp.loc[ind].at[year]
                
                if weight > 0:
                    edge_data = {"weight" : weight, "year" : int(year[1:])}
                    
                    edge = (country, policy, edge_key, edge_data) #create edge
                    year_edge_list.append(edge)
                    edge_key += 1

            # Take out years with no policies
            if not (len(year_edge_list) == 0):         
                G.add_edges_from(year_edge_list)
                return_years.append(int(year[1:]))
        return return_years
    
    """
    Helper function to return the subgraph of G for the specificed year
    """
    def return_subgraph_year(self, G : nx.MultiGraph, year : int) -> nx.MultiGraph:        
        # If a node has no edge, it will be removed. We don't want this so we make a copy of the nodes and will add them back into the graph latter.
        sub_graph = nx.MultiGraph()
        
        # Add nodes to subgraph
        nodes = G.nodes(data=True)
        sub_graph.add_nodes_from(nodes)
        
        # Add edges to subgraph
        edges = [(u, v, key, data) for u, v, key, data in G.edges(keys=True, data=True) if data["year"] == year]
        sub_graph.add_edges_from(edges)
        
        return sub_graph
    
    """
    Helper function to return the subgraph of G for the specificed year
    """
    def return_subgraph_policy(self, G : nx.MultiGraph, filter_policy : str) -> nx.MultiGraph:
        
        sub_graph = nx.MultiGraph()
        
        # Adds subset of Nodes that are affiliated with a specific policy (Countries that take part in a type of policy)
        nodes = [(node, data) for node, data in G.nodes(data=True) if filter_policy in data["policies"]]
        sub_graph.add_nodes_from(nodes)
        
        # Adds Edges between nodes 
        node_names = sub_graph.nodes()
        edges = [(node_country, node_policy, key, data) for node_country, node_policy, key, data in G.edges(keys=True, data=True) if node_policy in node_names]
        sub_graph.add_edges_from(edges)

        return sub_graph

    """
    Get a dictionary where the key is the year and the value is the set of bipartite edges for the given year
    """
    def get_temporal(self, weighted = True):
        reindexed = self.data.set_index(["ISO2", "CTS_Code"])
        temporal = {}
        for year in self.years: #create edge lists for each year
            edge_list = []
            for country in self.countries: #find every country-bond combo
                subset = reindexed.loc[country]
                for policy in subset.index: #bond
                    weight = subset.loc[policy]['F' + str(year)] #edge weights are bond values
                    if weight > 0:
                        edge = (country, policy, weight) if weighted else (country, policy)
                        edge_list.append(edge)
            temporal[year] = edge_list
        return temporal

    """
    From the year -> edges map, return a dataframe displaying various metrics for each year
    """
    def temporal_metrics(self):
        results = []
        temporal = self.get_temporal(weighted = True)
        for year in self.years:
            self.B.remove_edges_from(self.B.edges)
            self.B.add_weighted_edges_from(temporal[year])
            cats = caterpillars(self.B)
            buts = butterflies(self.B)
            ratio = 0 if cats == 0 else buts / cats
            results.append({"Year": year, "Countries": country_count(self.B, self.countries), "Edges":\
                            self.B.number_of_edges(), "Butterflies": buts, "Caterpillars": cats, "Ratio": ratio})
        return pd.DataFrame(results)
    
    """
    Iterates through the temporal graph to find total influences and total policies
    """
    def influence_values(self):
        #dictionaries for storing any influence values
        temporal = self.get_temporal(weighted = False)
        total_policies = {}
        pairs = {}

        #loop through temporal and add influence values
        for year in self.years:
            if year + 1 in temporal:
                I = temporal[year] #influencers
                F = temporal[year + 1] #followers

                for c_I, p_I in I:
                    #keep track of total implementations
                    if c_I in total_policies:
                        total_policies[c_I] += 1
                    else:
                        total_policies[c_I] = 1

                    #keep track of influences
                    for c_F, p_F in F:
                        if p_I == p_F and c_I != c_F and (c_F, p_F) not in I:
                            if (c_I, c_F) in pairs:
                                pairs[(c_I, c_F)] += 1
                            else:
                                pairs[(c_I, c_F)] = 1

        return pairs, total_policies
    
    """
    Based on influence values, constructs a Directed Graph of influence
    """
    def influence_graph(self):
        pairs, total_policies = self.influence_values()

        edges = set()
        for i, f in pairs: #find all influencer-follower edges S_ij
            weight = pairs[(i, f)] / total_policies[i] #divide each influence by total number of policies Q_i
            edge = (i, f, weight)
            edges.add(edge)
        
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        return G