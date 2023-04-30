import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

#number of countries with positive contribution
def country_count(B, country_nodes):
    deg_policy, deg_country = bipartite.degrees(B, country_nodes)
    sum = 0
    for c, d in deg_country:
        if d > 0:
            sum += 1
    return sum

#returns a list of every pair of edges that of which country_node is the source
def calculate_pairs(G, country_node):
    policies = [p for c, p in G.edges(country_node)]
    N = len(policies)
    pairs = []
    for i in range(N):
        for j in range(N):
            if i > j:
                pairs.append((policies[i], policies[j]))
    return pairs

#given that pairs can begin with either node, maps like pairs to the same value
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

#returns the butterfly count in a graph G, defined above
def butterflies(G):
    #initialize
    country_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    policy_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    #get counts for each pair of policies simultaneously implemented
    pair_vals = []
    pair_map = pair_mapper(policy_nodes)
    for country in country_nodes:
        for pair in calculate_pairs(G, country):
            pair_vals.append(pair_map[pair])

    #calculate butterflies
    butterflies = 0
    for val in pd.value_counts(pair_vals):
        butterflies += val * (val - 1) // 2
    return butterflies

#returns the caterpillar count in graph G, defined above
def caterpillars(G):
    #initialize
    country_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    policy_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    #get counts for each pair of policies simultaneously implemented
    caterpillars = 0
    for country in country_nodes:
        for a, b in calculate_pairs(G, country):
            caterpillars += (G.degree(a) + G.degree(b) - 2)
    
    return caterpillars

class CCNA:

    def __init__(self, name):
        #load data
        self.data = pd.read_csv("input/policy/" + name + ".csv",\
                                index_col=0, keep_default_na=False, na_values="")
        self.data = self.data[np.logical_not(self.data["ISO2"].isna())] #filter down
        if len(self.data.Unit.unique()) == 2:
            self.data = self.data[self.data.Unit == "Percent of GDP"]
        
        #other attributes
        self.countries = self.data.ISO2.unique()
        self.policies = self.data.CTS_Code.unique()
        self.years = [int(year[1:]) for year in self.data.columns if len(year) == 5 and year[0] == 'F']

        #create global bipartite graph
        self.B = nx.Graph()
        self.B.add_nodes_from(self.countries, bipartite=0)
        self.B.add_nodes_from(self.policies, bipartite=1)

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
    
class IP:

    def __init__(self, G):
        self.G = G
        self.countries = list(G.nodes)

        # Calculate acceptance and rejection rates for each pair
        self.acceptance, self.rejection = self.pairwise_weights()

        # Store influence and passivity values for each country
        self.influence = np.ones(len(self.countries))
        self.passivity = np.ones(len(self.countries))

    def pairwise_weights(self):
        acceptance = []
        rejection = []
        return acceptance, rejection