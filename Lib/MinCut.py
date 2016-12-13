import numpy as np
import scipy
from sklearn.cluster import KMeans
from cvxpy import *


def get_degree(A):
    return np.diag(np.sum(A, axis=0))

def get_nonnorm_lapl(A):
    cur_degr_mat = get_degree(A)
    return cur_degr_mat - A

def get_lapl_sym(A):
    nonnorm_lapl = get_nonnorm_lapl(A)
    cur_degree_mat = get_degree(A)
    multipl = np.linalg.inv(scipy.linalg.sqrtm(cur_degree_mat))
    return (multipl.dot(nonnorm_lapl)).dot(multipl)

def get_lapl_rw(A):
    nonnorm_lapl = get_nonnorm_lapl(A)
    cur_degree_mat = get_degree(A)
    multipl = np.linalg.inv(cur_degree_mat)
    return multipl.dot(nonnorm_lapl)

def unnorm_predict(A, k):
    eigenvec = scipy.linalg.eigh(get_nonnorm_lapl(A),eigvals=(0, k - 1))[1]
    kmeans = KMeans(n_clusters = k).fit(eigenvec)
    return kmeans.labels_

def lrw_predict(A, k):
    eigenvec = scipy.linalg.eigh(get_lapl_rw(A),eigvals=(0, k - 1))[1]
    kmeans = KMeans(n_clusters = k).fit(eigenvec)
    return kmeans.labels_

def lsym_predict(A, k):
    eigenvec = scipy.linalg.eigh(get_lapl_sym(A),eigvals=(0, k - 1))[1]
    nrm = np.abs(eigenvec).sum(axis=1)
    T = eigenvec / nrm.reshape(eigenvec.shape[0],1)
    kmeans = KMeans(n_clusters = k).fit(T)
    return kmeans.labels_