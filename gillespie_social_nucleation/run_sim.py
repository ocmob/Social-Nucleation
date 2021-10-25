import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing
import logging
from multiprocessing import Pool
from numpy.random import default_rng

from model import State
from gillespie import gillespie

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

np.seterr('raise')

def build_graph_from_adj_matrix(adj_matrix):
    G = nx.Graph()

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1 and i != j:
                G.add_edge(i,j)
    
    return G

END_TIME = 25000
rng = default_rng(69)

N_agents = 100
M_cr = N_agents/2
mu_sweep = np.linspace(0.1, 6, 50)
alpha = 0
beta = 1
eps = 0.1
group_benefit = 1
group_cost = float('-inf')

lcc_frac_mu_sweep_results = []

states = [State(rng, N_agents, np.ceil(mu * M_cr), alpha, beta, eps, group_benefit, group_cost) for mu in mu_sweep]

def gillespie_end_time(state):
    return gillespie(state, END_TIME)[2][-1]

with Pool(processes=6) as pool:
    res = pool.map(gillespie_end_time, states)

plt.plot(mu_sweep, res)
plt.show()



#fig, ax = plt.subplots(2, 2)
#
#len_adj_last_index = len(adjacency) - 1
#
#for i in range(1, 5):
#    ax_i = (i - 1) // 2
#    ax_j = (i - 1) % 2
#    current_ax = ax[ax_i, ax_j]
#    current_ts_index = int(np.ceil(i/4*len_adj_last_index))
#
#    nx.draw_networkx(build_graph_from_adj_matrix(adjacency[current_ts_index]), ax = current_ax)
#    current_ax.set_title('Time: {:.4f}'.format(time[current_ts_index]))
#
#plt.show()
#
#plt.plot(time, lcc_frac, label = '$n^{LCC}/N$')
#plt.legend()
#plt.show()
#
#plt.plot(time, reaction_tallies, label = ['No of coagulations', 'No of densifications', 'No of dissolustions'])
#plt.legend()
#plt.show()
