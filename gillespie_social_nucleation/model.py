import numpy as np

class State:
    def __init__(self, rng, N_agents, M_links, alpha, beta, group_benefit, group_cost):
        self.N_agents = N_agents
        self.M_links = M_links

        self.group_benefit = group_benefit
        self.group_cost = group_cost
        self.rho_hat = 2*group_cost/group_benefit

        self.beta = beta
        self.alpha = alpha
        self.eps = 0.8

        self.opinions = rng.random(N_agents)
        self.groups = [[i] for i in range(N_agents)]
        self.adjacency = np.identity(N_agents)

        self.time = 0
        self.rng = rng

        self.reaction_tallies = {
            'coagulation' : 0,
            'densification' : 0,
            'dissolution' : 0,
        }

        self.time_hist = [0]
        self.adjacency_hist = [np.identity(N_agents)]
        self.lcc_frac_hist = [1/N_agents]
        self.reaction_tallies_hist = [(0, 0, 0)]

    def get_propensities(self):
        propensities = np.zeros((self.N_agents*(self.N_agents + 2), 1))

        for i, group_i in enumerate(self.groups):
            for j, group_j in enumerate(self.groups):
                if i != j:
                    propensities[i*self.N_agents + j] = self.coagulation_propensity(group_i, group_j)

        for i, group_i in enumerate(self.groups):
            propensities[self.N_agents*self.N_agents + i] = self.densification_propensity(group_i)

        for i, group_i in enumerate(self.groups):
            propensities[self.N_agents*(self.N_agents + 1) + i] = self.dissolution_propensity(group_i)

        return propensities

    def exec_reaction_by_index(self, index):
        if index < self.N_agents * self.N_agents:
            group_1_index = index // self.N_agents
            group_2_index = index % self.N_agents

            group_1 = self.groups[group_1_index]
            group_2 = self.groups[group_2_index]

            self.coagulate_groups(group_1, group_2)

        elif index < self.N_agents * (self.N_agents + 1):
            group = self.groups[index - self.N_agents * self.N_agents]
            self.densify_group(group)

        else:
            group = self.groups[index - self.N_agents * (self.N_agents+1) ]
            self.dissolve_group(group)

    def coagulation_propensity(self, group_1, group_2):
        if len(group_1) > 0 and len(group_2) > 0:
            opinion_matrix = self.get_opinion_matrix_for_cartesian_prod_of_groups(group_1, group_2)

            return 2*self.M_links/self.N_agents**2*len(group_1)*len(group_2)*opinion_matrix.sum()
        else:
            return 0

    def densification_propensity(self, group):
        if len(group) > 0:
            opinion_matrix = self.get_opinion_matrix_for_cartesian_prod_of_groups(group, group)
            group_adjacency = self.adjacency[tuple(np.meshgrid(group, group, indexing = 'ij'))]

            if len(group) > 5 and ((group_adjacency == 0)*opinion_matrix).sum() == 0:
                breakpoint()
            
            return 2*self.M_links/self.N_agents**2*( ((group_adjacency == 0)*opinion_matrix).sum()/2 )
        else:
            return 0

    def dissolution_propensity(self, group):
        if len(group) > 0:
            group_adjacency = self.adjacency[tuple(np.meshgrid(group, group, indexing = 'ij'))]
            no_links = (group_adjacency - np.identity(len(group))).sum()/2
            rho_group = 2*no_links/len(group)

            return len(group)/self.N_agents*np.exp(self.beta*self.rho_hat/rho_group)
        else:
            return 0

    def get_opinion_matrix_for_cartesian_prod_of_groups(self, group_1, group_2):
        eff_opinions_group_1 = self.get_effective_opinions_for_group(group_1)
        eff_opinions_group_2 = self.get_effective_opinions_for_group(group_2)
        
        return (np.abs(np.subtract.outer(eff_opinions_group_1, eff_opinions_group_2)) - self.eps) > 0

    def get_effective_opinions_for_group(self, group):
        group_opinions = self.opinions[group]
        eff_opinions = group_opinions * (1-self.alpha) + group_opinions.mean() * self.alpha 
        return eff_opinions

    def coagulate_groups(self, group_1, group_2):
        opinion_matrix = self.get_opinion_matrix_for_cartesian_prod_of_groups(group_1, group_2)
        possible_links = opinion_matrix.nonzero()

        if(len(possible_links[0]) == 0):
            breakpoint()
        
        link_to_add = self.rng.integers(0, len(possible_links[0]))

        agent_1_to_link = group_1[possible_links[0][link_to_add]]
        agent_2_to_link = group_2[possible_links[1][link_to_add]]

        self.adjacency[agent_1_to_link, agent_2_to_link] = 1
        self.adjacency[agent_2_to_link, agent_1_to_link] = 1

        group_1 += group_2
        group_2.clear()

        self.M_links -= 1
        self.reaction_tallies['coagulation'] += 1

    def densify_group(self, group):
        opinion_matrix = self.get_opinion_matrix_for_cartesian_prod_of_groups(group, group)
        group_adjacency = self.adjacency[tuple(np.meshgrid(group, group, indexing = 'ij'))]

        possible_links = ((group_adjacency == 0) * opinion_matrix).nonzero()

        link_to_add = self.rng.integers(0, len(possible_links[0]))
    
        agent_1_to_link = group[possible_links[0][link_to_add]]
        agent_2_to_link = group[possible_links[1][link_to_add]]

        self.adjacency[agent_1_to_link, agent_2_to_link] = 1
        self.adjacency[agent_2_to_link, agent_1_to_link] = 1

        self.M_links -= 1
        self.reaction_tallies['densification'] += 1
        
    def dissolve_group(self, group):
        agent_to_remove = group[self.rng.integers(0, len(group))]

        agent_adjacency = self.adjacency[agent_to_remove, :]
        no_links_removed = agent_adjacency.sum() - 1

        self.adjacency[agent_to_remove, agent_adjacency.nonzero()] = 0
        self.adjacency[agent_adjacency.nonzero(), agent_to_remove] = 0
        self.adjacency[agent_to_remove, agent_to_remove] = 1

        group.remove(agent_to_remove)

        for group in self.groups:
            if not group:
                group.append(agent_to_remove)
        
        self.M_links += no_links_removed
        self.reaction_tallies['dissolution'] += 1

    def advance_time(self, tau):
        self.time += tau

        self.time_hist.append(self.time)
        self.adjacency_hist.append(self.adjacency.copy())
        self.lcc_frac_hist.append(self.get_lcc_fraction())
        self.reaction_tallies_hist.append((self.reaction_tallies['coagulation'], self.reaction_tallies['densification'], self.reaction_tallies['dissolution']))

    def get_statistics(self):
        return(self.time_hist, self.adjacency_hist, self.lcc_frac_hist, self.reaction_tallies_hist)

    def get_time(self):
        return self.time

    def get_lcc_fraction(self):
        return max([len(group) for group in self.groups])/self.N_agents
