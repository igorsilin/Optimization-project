import numpy as np

def compute_nmi(labels_true, labels_pred):
	
	from sklearn.metrics import normalized_mutual_info_score
	
	return normalized_mutual_info_score(labels_true, labels_pred)

def match_straight(clusters_true, clusters_pred):

	g = {}

	for i in xrange(len(clusters_true)):

		A = set(clusters_true[i])
		f1_max = -1000
		g[i] = -1
		for j in xrange(len(clusters_pred)):
			B = set(clusters_pred[j])

			if (len(A & B) != 0):
				precision = 1.0 * len(A & B) / len(B)
				recall = 1.0 * len(A & B) / len(A)
				f1 = 2.0 * precision * recall / (precision + recall)

				if (f1 > f1_max):
					f1_max = f1
					g[i] = j

	return g

def match_reverse(clusters_true, clusters_pred):

	g_ = {}

	for j in xrange(len(clusters_pred)):
		B = set(clusters_pred[j])
		f1_max = -1000
		g_[j] = -1
		for i in xrange(len(clusters_true)):
			A = set(clusters_true[i])

			if (len(A & B) != 0):
				precision = 1.0 * len(A & B) / len(B)
				recall = 1.0 * len(A & B) / len(A)
				f1 = 2.0 * precision * recall / (precision + recall)

				if (f1 > f1_max):
					f1_max = f1
					g_[j] = i
	
	return g_

def compute_recall(clusters_true, clusters_pred):

	res = 0.0
	g = match_straight(clusters_true, clusters_pred)
	for i in xrange(len(clusters_true)):
		if g[i] != -1:
			A = set(clusters_true[i])
			B = set(clusters_pred[g[i]])
			res = res + 1.0 * len(A & B) / len(A)
	
	recall = res / len(clusters_true)
	return recall     

def compute_precision(clusters_true, clusters_pred):

	res = 0.0
	g = match_straight(clusters_true, clusters_pred)

	for i in xrange(len(clusters_true)):
		if g[i] != -1:
			A = set(clusters_true[i])
			B = set(clusters_pred[g[i]])
			res = res + 1.0 * len(A & B) / len(B)
	
	precision = res / len(clusters_true)
	return precision  

def compute_avg_f1(clusters_true, clusters_pred):
		
	res1 = 0.0
	res2 = 0.0
	g = match_straight(clusters_true, clusters_pred)
	g_ = match_reverse(clusters_true, clusters_pred)

	for i in xrange(len(clusters_true)):
		if g[i] != -1:
			A = set(clusters_true[i])
			B = set(clusters_pred[g[i]])
			precision = 1.0 * len(A & B) / len(B)
			recall = 1.0 * len(A & B) / len(A)
			f1 = 2.0 * precision * recall / (precision + recall)
			res1 = res1 + f1

	for j in xrange(len(clusters_pred)):
		if g_[j] != -1:
			B = set(clusters_pred[j])
			A = set(clusters_true[g_[j]])
			precision = 1.0 * len(A & B) / len(B)
			recall = 1.0 * len(A & B) / len(A)
			f1 = 2.0 * precision * recall / (precision + recall)
			res2 = res2 + f1
	
	avg_f1 = 1 / 2.0 * (res1 / len(clusters_true) + res2 / len(clusters_pred))
	return avg_f1

def compute_modularity(labels_pred, edge_list):

	n_edges = len(edge_list)
	m = 0.0   #finally m = sum w_ij
	A = 0.0   
	k = [0.0] * (max(labels_pred)+1)
	res = 0.0

	for i in xrange(n_edges):
		m += 1

		if (labels_pred[edge_list[i][0]] == labels_pred[edge_list[i][1]]):
			A += 2      
		
		k[labels_pred[edge_list[i][0]]] += 1
		k[labels_pred[edge_list[i][1]]] += 1

	for i in xrange((max(labels_pred)+1)):
		res += k[i]*k[i]

	Q = A/(2.0*m) - res/(4.0*m*m)        #modularity
	return Q


def compute_normalized_cut(labels_pred, clusters_pred, edge_list):

    res = [0.0]*(len(labels_pred))
    vol = [0.0]*(len(labels_pred))
    for i in xrange(len(edge_list)):
        vol[labels_pred[edge_list[i][0]]] += 1
        vol[labels_pred[edge_list[i][1]]] += 1
        if (labels_pred[edge_list[i][0]] != labels_pred[edge_list[i][1]]):
            res[labels_pred[edge_list[i][0]]] += 1
            res[labels_pred[edge_list[i][1]]] += 1

    normcut = 0.0
    for i in xrange(len(labels_pred)):
        if (vol[i] != 0):
            normcut += res[i]/vol[i]
        
    return normcut