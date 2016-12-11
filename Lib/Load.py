import numpy as np

def download_graph(filename):
    f = open('Data/'+filename+'.txt', "r")
    s = f.readline()
    n_nodes, n_edges = s.split(' ');
    n_nodes = int(n_nodes)
    n_edges = int(n_edges)
    edge_list = []

    for i in xrange(n_edges):
        s = f.readline()
        vertex1, vertex2 = s.split(' ');
        vertex1 = int(vertex1) - 1
        vertex2 = int(vertex2) - 1
        edge_list.append([vertex1, vertex2]) 
        
    f.close()

    return [n_nodes, edge_list]

def download_labels(filename):
    
    f = open('Data/' + filename + '_labels.txt', "r")
    labels_true = []
    s = f.readline()

    while s:
        labels_true.append(int(s));
        s = f.readline()

    f.close()

    return labels_true

def save_graph(A, name): #draw graph and save figure
    n_nodes = A.shape[0]
    edge_list = []
    for i in xrange(n_nodes):
        for j in xrange(i+1, n_nodes):
            if A[i,j] == 1:
                edge_list.append([i, j])
    f = open('Data/' + name + ".txt", "w")
    f.write(str(n_nodes) + ' ' + str(len(edge_list))+'\n')
    for i in xrange(len(edge_list)):
        f.write(str(edge_list[i][0]+1) + ' ' + str(edge_list[i][1]+1)+'\n')

    return

def save_labels(labels, name): #draw graph and save figure
    n_nodes = len(labels)
    f = open('Data/' + name + ".txt", "w")

    for i in xrange(n_nodes):
        f.write(str(labels[i]) + '\n')

    return