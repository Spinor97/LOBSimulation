from scipy import stats
import numpy as np

def sim_hawkes(T, lambda_, mu, beta):
    times = []

    s = 0
    bar_lam = mu
    omega = 0
    add = False

    while s < T:
        bar_lam = mu + (bar_lam - mu) * np.exp(-beta * omega)
        if add:
            bar_lam += lambda_(s - times[-1])

        u = stats.uniform.rvs()
        d = stats.uniform.rvs()

        omega = -np.log(u) / bar_lam
        s += omega

        if d * bar_lam <= mu + (bar_lam - mu) * np.exp(-beta * omega) and s <= T:
            times.append(s)
            add = True

    return times

def mockProcess(T, lambda_, mus, betas, num_events):
    results = []

    for i in range(num_events):
        times = sim_hawkes(T, lambda_, mus[i], betas[i])
        results.extend([(t, i) for t in times])

    results.sort(key = lambda x: x[0])

    return results

def indicator(bid, ask):
    return (bid - ask) / (ask + bid)

def generateStates(process, rules, initial:list):
    """
    input:
    process: list of tuples (time, event_type)
    rules: tuple consisting distributions corresponding to each type of event to generate the change of state and the state to be added
    initial: initial state(bid && ask)
    """
    states = []
    indicator = indicator(initial[0], initial[1])

    for i in range(len(process)):
        time, event_type = process[i]
        dist, state = rules[event_type]
        add = dist.rvs()
        initial[state] += add
        states.append((time, indicator))

    return states

if __name__ == "__main__":
    T = 100
    lambda_ = lambda t: 0.1
    mu = 0.1
    beta = 0.1
    mus = [0.1, 0.1]
    betas = [0.1, 0.1]
    num_events = 2

    process = mockProcess(T, lambda_, mus, betas, num_events)
    rules = (stats.norm, 0), (stats.norm, 1)
    initial = [100, 100]

    states = generateStates(process, rules, initial)
    print(states)
