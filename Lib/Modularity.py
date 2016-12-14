import numpy as np

def Newman(n_nodes, A):

	m = np.sum(A)/2
	e = A / (2. * m)
	a = np.sum(e, axis = 0)

	Q = np.sum(np.diagonal(e)) - np.sum(a ** 2)

	active = [True] * n_nodes

	cluster = [[s] for s in xrange(n_nodes)]
	clusters = [[s] for s in xrange(n_nodes)]

	current_s = 0
	current_t = 0
	
	current_max = -10
	for _ in xrange(n_nodes-1):
		current_max = -10

		for s in xrange(n_nodes):
			for t in xrange(s+1, n_nodes):
				if (active[s] and active[t]):
					if (e[s,t] + e[t,s] - 2*a[s]*a[t] > current_max):
						current_max = e[s,t] + e[t,s] - 2*a[s]*a[t]
						current_s = s
						current_t = t

		if (current_max < 0):
			clusters = []
			for s in xrange(n_nodes):
				if (active[s]):
					clusters.append(cluster[s])

		e[current_s,:] = e[current_s,:] + e[current_t,:]
		e[:,current_s] = e[:,current_s] + e[:,current_t]
		a[current_s] = a[current_s] + a[current_t]

		cluster[current_s] = cluster[current_s] + cluster[current_t]

		active[current_t] = False
		
		Q += current_max

	if (current_max >= 0):
		clusters = []
		for s in xrange(n_nodes):
			if (active[s]):
				clusters.append(cluster[s])

	labels = [-1] * n_nodes
	for i in xrange(len(clusters)):
		for j in xrange(len(clusters[i])):
			labels[clusters[i][j]] = i
	return labels


	