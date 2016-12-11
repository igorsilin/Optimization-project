import numpy as np

def compute_clusters_from_labels(labels):

    clusters = []

    for i in set(labels):
        cluster = [j for j in xrange(len(labels)) if labels[j] == i]
        clusters.append(cluster)

    return clusters


def compute_adj_mat(n_nodes, edge_list):
    A = np.zeros([n_nodes, n_nodes])
    for i in xrange(len(edge_list)):
        A[edge_list[i][0], edge_list[i][1]] = A[edge_list[i][1], edge_list[i][0]] = 1
    return A