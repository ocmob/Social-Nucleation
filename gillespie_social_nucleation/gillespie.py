import numpy as np

def gillespie(state, end_time):
    rng = state.get_rng()

    while (state.get_time() < end_time) and (not state.trap_state_reached()):
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

    return state.get_statistics()
