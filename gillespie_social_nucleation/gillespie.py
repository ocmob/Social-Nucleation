import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import default_rng

from model import State

np.seterr('raise')

def build_graph_from_adj_matrix(adj_matrix):
    G = nx.Graph()

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1 and i != j:
                G.add_edge(i,j)
    
    return G

END_TIME = 100000

rng = default_rng(69)
state = np.zeros(2)

N_agents = 100
M_cr = N_agents/2
mu = 3
M_links = mu * M_cr

state = State(rng, N_agents, M_links, 0, 1, 1, float('-inf'))

while (state.get_time() < END_TIME) and (state.M_links > 0):
    r = rng.random(2)

    props = state.get_propensities()
    props_sum = props.sum()

    if props_sum == 0:
        breakpoint()

    for i in range(1, len(props)):
        if r[0] < props[:i].sum()/props_sum:
            state.exec_reaction_by_index(i-1)
            break

    tau = 1/r[1]*np.log(1/r[1])
    state.advance_time(tau)

    print('Time:', state.get_time(), 'r_react:', r[0], 'Reaction index:', i, ' '*100, end = "\r")

print("Ended.")
print("Time:", state.get_time())
print("Free links:", state.M_links)

time, adjacency, lcc_frac, reaction_tallies = state.get_statistics()

nx.draw_networkx(build_graph_from_adj_matrix(adjacency[0]))
plt.show()

nx.draw_networkx(build_graph_from_adj_matrix(adjacency[-1]))
plt.show()

plt.plot(time, lcc_frac)
plt.show()

plt.plot(time, reaction_tallies)
plt.show()
