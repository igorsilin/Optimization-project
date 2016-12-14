import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

# Draws a graph
# Vertices with the same labels have the same colors
def visualize_clusters(adjacency_matrix, labels_pred, labels_true, title):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.spring_layout(G)
    
    nodes = [i for i in G.nodes()]
    
    colors_pred = np.array(labels_pred)
    colors_pred = colors_pred / np.max(colors_pred)
    
    colors_true = np.array(labels_true)
    colors_true = colors_true / np.max(colors_true)
    
    nx.draw_networkx_nodes(G, pos, nodes, ax=ax[0], node_size=100, cmap=plt.get_cmap('rainbow'), node_color = colors_pred)
    nx.draw_networkx_edges(G, pos, G.edges(), ax=ax[0])
    
    nx.draw_networkx_nodes(G, pos, nodes, ax=ax[1], node_size=100, cmap=plt.get_cmap('rainbow'), node_color = colors_true)
    nx.draw_networkx_edges(G, pos, G.edges(), ax=ax[1])
    
    ax[0].set_title(title + ', predicted labels')
    ax[1].set_title(title + ', true labels')
    
    ax[0].axis('off')
    ax[1].axis('off')
    
    fig.subplots_adjust(wspace=0.2, hspace=1.0,
                    top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
    
    return
