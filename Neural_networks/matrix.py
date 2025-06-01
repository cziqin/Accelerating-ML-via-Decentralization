import numpy as np


def fully_connected_graph(num_agents, weight):
    W = np.ones((num_agents, num_agents))
    W *= weight

    return W
