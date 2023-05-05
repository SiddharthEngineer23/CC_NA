import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import sys
from helper import country_count, butterflies, caterpillars

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
        self.N = len(self.countries)

        # Store influence and passivity values for each iteration
        self.influence = []
        self.passivity = []

        # Calculate acceptance and rejection rates for each pair
        self.acceptance, self.rejection = self.pairwise_weights()

    def pairwise_weights(self):
        # Acceptance rates
        adjacency = np.zeros((self.N, self.N)) + nx.adjacency_matrix(self.G)
        b = adjacency.sum(axis=0)
        inverse = np.divide(1, b, where = (b!=0)) 
        acceptance = np.multiply(inverse, adjacency)

        # Rejection rates
        not_adjacency = np.ones((self.N, self.N)) - adjacency
        b = not_adjacency.sum(axis=0)
        inverse = np.divide(1, b, where = (b!=0))
        rejection = np.multiply(inverse, not_adjacency)

        return acceptance, rejection

    def update_passivity(self, I):
        P = []
        for i in range(self.N):
            P_i = 0
            for j in range(self.N):
                P_i += self.rejection[j, i] * I[j]
            P.append(P_i)
        return np.array(P)
    
    def update_influence(self, P):
        I = []
        for i in range(self.N):
            I_i = 0
            for j in range(self.N):
                I_i += self.acceptance[i, j] * P[j]
            I.append(I_i)
        return np.array(I)

    def iterate(self, M):
        I = np.ones(self.N)
        P = np.ones(self.N)

        # Run M iterations
        for m in range(M):
            P = self.update_passivity(I)
            I = self.update_influence(P)
            P /= P.sum()
            I /= I.sum()
            self.influence.append(I)
            self.passivity.append(P)

    def chart_convergence(self):
        M = len(self.influence)
        influence = self.influence[M - 1]
        passivity = self.passivity[M - 1]

        # We'll go through our iterations and calculate the MSE with our final vector
        I_diffs = []
        P_diffs = []
        for m in range(M):
            I_diff = ((self.influence[m] - influence)**2).mean()
            P_diff = ((self.passivity[m] - passivity)**2).mean()
            I_diffs.append(I_diff)
            P_diffs.append(P_diff)

        return np.array(I_diffs), np.array(P_diffs)


# Temporal graph
print("Running temporal graph...")
names = ["Environmental_Protection_Expenditures", "Environmental_Taxes", "Fossil_Fuel_Subsidies", "Green_Bonds"]
obj = CCNA(names[int(sys.argv[1])])
metrics = obj.temporal_metrics()
print("Temporal Metrics:\n", metrics)

# # Plot temporal metrics
# N = len(metrics.index)
# x = metrics.Year[:N-1]
# y = metrics.Ratio[:N-1]
# plt.plot(x, y)
# plt.xlabel("Year")
# plt.ylabel("Ratio")
# plt.title("Butterfly to Caterpillar Ratio for\n" + names[int(sys.argv[1])] + " Dataset")
# plt.savefig('Ratio_' + names[int(sys.argv[1])] + '.png')

# Directed influence graph
print("Building directed graph...")
G = obj.influence_graph()
metrics = {"Nodes": G.number_of_nodes(), "Edges": G.number_of_edges(), "Clustering": nx.average_clustering(G),\
           "Countries": len(obj.countries), "Policies": len(obj.policies)}
print("Directed Graph Values:\n", metrics)

# Initializing IP values
print("Loading graph into IP algorithm...")
ip_obj = IP(G)
print("IP algorithm initialized.")

# Computing influence and passivity
M = int(sys.argv[2])
print("Running", M, "iterations...")
ip_obj.iterate(M)
influence = ip_obj.influence[M - 1]
passivity = ip_obj.passivity[M - 1]

# Derive some results
top_influencers = pd.DataFrame({"Country": ip_obj.countries, "Influence": influence,\
                                "Passivity": passivity}).sort_values('Influence', ascending=False).head(10)
most_passive = pd.DataFrame({"Country": ip_obj.countries, "Influence": influence,\
                                "Passivity": passivity}).sort_values('Passivity', ascending=False).head(10)
top_influencers.to_csv(names[int(sys.argv[1])] + '.csv')

# # Plot convergence
# x = range(M)
# y_i, y_p = ip_obj.chart_convergence()
# plt.plot(x, np.log(y_i))
# plt.plot(x, np.log(y_p))
# plt.xlabel("Number of Iterations: " + str(M))
# plt.ylabel("Mean Squared Difference (Log Scale)")
# plt.title("Convergence of IP Algorithm for " + names[int(sys.argv[1])] + " Dataset")
# plt.savefig('Convergence_' + names[int(sys.argv[1])] + '_' + str(M) + '.png')