import numpy as np
from cvxpy import *

def SDP1(n_nodes, A, n_clusters):

	X = Semidef(n_nodes)

	constraints = [X >= 0,
	               diag(X) == 1,
	               sum_entries(X, axis=1) == n_nodes / n_clusters]

	objective = Maximize(sum_entries(mul_elemwise(A, X)))

	prob = Problem(objective, constraints)
	prob.solve()

	labels_pred = [-1]*n_nodes
	current_cluster = 0
	for i in xrange(n_nodes):
		if (labels_pred[i] == -1):
			labels_pred[i] = current_cluster

			for j in xrange(i+1, n_nodes):
				if (X.value[i,j] >= 0.5):
					labels_pred[j] = current_cluster

			current_cluster += 1

	return labels_pred


def SDP2(n_nodes, A, n_clusters):

	X = Semidef(n_nodes)

	constraints = [X >= 0,
				   X <= 1,
	               trace(X) == n_nodes,
	               sum_entries(X) == n_nodes ** 2 / n_clusters]

	objective = Maximize(sum_entries(mul_elemwise(A, X)))

	prob = Problem(objective, constraints)
	prob.solve()

	labels_pred = [-1]*n_nodes
	current_cluster = 0
	for i in xrange(n_nodes):
		if (labels_pred[i] == -1):
			labels_pred[i] = current_cluster

			for j in xrange(i+1, n_nodes):
				if (X.value[i,j] >= 0.5):
					labels_pred[j] = current_cluster

			current_cluster += 1

	return labels_pred