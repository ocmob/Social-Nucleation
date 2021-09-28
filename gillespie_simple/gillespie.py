import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

END_TIME_SEC = 1000

K1 = 1e-3
K2 = 1e-2
K3 = 1.2
K4 = 1

A = lambda state: state[0]
B = lambda state: state[1]

PROPENSITIES = [
    lambda state: A(state)*(A(state) - 1)*K1,
    lambda state: A(state)*B(state)*K2,
    lambda state: K3,
    lambda state: K4
]

def update_state(state, alphas, r1):
    if r1 < alphas[1]/alphas[0]:
        state[0] -= 2
    elif r1 < (alphas[1] + alphas[2])/alphas[0]:
        state -= 1
    elif r1 < (alphas[1] + alphas[2] + alphas[3])/alphas[0]:
        state[0] += 1
    else:
        state[1] += 1

rng = default_rng(69)

state = np.zeros(2)
state_hist = []
state_hist.append(state.copy())

alphas = np.zeros(5)

t_hist = [0]
t_current = 0

while t_current < END_TIME_SEC:
    r = rng.random(2)

    for i, prop in enumerate(PROPENSITIES):
        alphas[i+1] = PROPENSITIES[i](state)

    alphas[0] = alphas[1:].sum()

    tau = 1/alphas[0]*np.log(1/r[0])

    update_state(state, alphas, r[1])

    t_current += tau

    t_hist.append(t_current)
    state_hist.append(state.copy())

plt.plot(t_hist, state_hist)
plt.show()
