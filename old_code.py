
    def __init__(self):
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

    def __init__(self):
        # Load in Subsidy data
        subsidy = pd.read_csv('input/policy/Fossil_Fuel_Subsidies.csv', index_col=0, keep_default_na=False, na_values="")
        sub_country_nodes, sub_policy_nodes = self.obtain_country_and_policies(subsidy)

        # Load in green bonds
        green_bonds = pd.read_csv("input/policy/Green_Bonds.csv", index_col=0, keep_default_na=False, na_values="")
        # Filtered Down to countries
        green_bonds = green_bonds[np.logical_not(green_bonds["ISO2"].isna())]
        bond_country_nodes, bond_policy_nodes = self.obtain_country_and_policies(green_bonds)

        # Load in Taxes
        taxes = pd.read_csv("input/policy/Environmental_Taxes.csv", index_col=0, keep_default_na=False, na_values="")
        tax_country_nodes, tax_policy_nodes = self.obtain_country_and_policies(taxes)

        # Load in Expenditures:
        expenditures = pd.read_csv("input/policy/Environmental_Protection_Expenditures.csv", index_col=0, keep_default_na=False, na_values="")
        exp_country_nodes, exp_policy_nodes = self.obtain_country_and_policies(expenditures)

        G = nx.MultiGraph()

        # Territories are included in this.
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
                
            country_node_attributes[country] = {"bipartite" : 0, "policies" : policies}
            
        G.add_nodes_from(all_countries)
        nx.set_node_attributes(G,country_node_attributes)
            
        # Now we will add the policy nodes using a very similar approach:
        all_policies = tax_policy_nodes | exp_policy_nodes  | bond_policy_nodes  | sub_policy_nodes
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
                
            policy_node_attributes[policy] = {"bipartite" : 1, "policies" : policies}
            
        G.add_nodes_from(all_policies)
        nx.set_node_attributes(G, policy_node_attributes)

        taxes_sub = taxes[taxes.Unit == "Percent of GDP"].fillna(0)
        tax_years = self.add_edges(G, taxes_sub, tax_country_nodes, tax_policy_nodes)

        green_bonds_sub = green_bonds.fillna(0)
        bond_years = self.add_edges(G, green_bonds_sub, bond_country_nodes, bond_policy_nodes)

        subsidy_sub = subsidy_sub = subsidy[subsidy["Unit"] == "Percent of GDP"]
        subsidy_years = self.add_edges(G, subsidy_sub, sub_country_nodes, sub_policy_nodes)


        exp_sub = expenditures[expenditures["Unit"] == "Percent of GDP"]
        exp_years = self.add_edges(G, exp_sub, exp_country_nodes, exp_policy_nodes)

        # Create Country Name Lookup Table
        country_name_conversion = pd.read_csv("input/wikipedia-iso-country-codes.csv")
        country_name_conversion.rename(columns = {"English short name lower case" : "Country Name", "Alpha-2 code" : "ISO2", "Alpha-3 code" : "ISO3"}, inplace = True)

        two_way_country_lookup = TwoWayDict()
        oneway_lookup = country_name_conversion.set_index("Country Name")["ISO2"].to_dict()
        two_way_country_lookup.from_dict(oneway_lookup)
        """