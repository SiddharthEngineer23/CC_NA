import pandas as pd
import numpy as np
import networkx as nx
import sys
from ccna.policy_net import PolicyNet

"""
Class which takes an influence graph as input and computes influence-passivity values
"""
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

if __name__ == "__main__":
    # Temporal graph
    print("Running temporal graph...")
    names = ["Environmental_Protection_Expenditures", "Environmental_Taxes", "Fossil_Fuel_Subsidies", "Green_Bonds"]
    obj = PolicyNet(names[int(sys.argv[1])])
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