# M. Danilova, A. Podkopaev, N. Puchkin, I. Silin
#
# Optimization Methods Project
# Application of natural conjugate gradient
# in community detection

# Most of notations are taken from the paper
# Zhiqiang Xu, Yiping Ke, Yi Wang
# 'A Fast Inference Algorithm for Stochastic Block Model',
# 2014 IEEE International Conference on Data Mining

import scipy as sc
from scipy import special
from math import inf
from math import isnan


# Group assignment as maximum likelihood estimation
# via natural conjugate gradient
#
# X -- NxN matrix; adjacency matrix
# k -- positive integer; number of clusters
# tol -- float; tolerance
# maxIter -- integer; maximum number of iterations
# 
# tol and maxIter determine conditions of termination

def NCG_Clustering(X, k, tol=1e-6, maxIter=200):
    # Number of nodes
    N = X.shape[0]
    # Initial guess of parameter
    # Theta -- Nx(k-1) matrix,
    Theta = Initial_Guess(N, k)
    
    # Old value of log-likelihood
    L_old = -inf
    
    lambd = 1
    d = sc.zeros((N, k-1))
    g_norm = 1
        
    # Main loop for iterative search of optimal values
    for t in range (maxIter):
        #print('||Theta|| = ', sc.linalg.norm(Theta, ord=inf))
        # Nxk matrix
        Pi = Calculate_Pi(Theta)
        # k-dimensional vector
        alpha = Calculate_Alpha(Pi)
        # kxk matrices
        beta1, beta2 = Calculate_Beta(X, Pi)
        # Log-likelihood
        L = Log_Likelihood(X, Pi, alpha, beta1, beta2)
        
        if (isnan(L)):
            L = L_old
        
        # eta = (L - L_old) / |L|,
        # where L and L_old are current and old
        # values of log-likelihood respectively
        eta = (L - L_old) / sc.absolute(L)
        if ((0 <= eta) and (eta < tol)):
            break
        elif (eta >= 0):
            # Calculate Natural Conjugent Gradient
            d, g_norm = Calculate_NCG(X, Pi, alpha, beta1, beta2, d, g_norm)
            # Update values
            Theta += lambd * d
            L_old = L
        else:
            lambd *= 0.5
            Theta -= lambd * sc.absolute(eta) * d
            
    # Cluster assignments
    Z = sc.argmax(Pi, axis=1)
    
    return Z	
	
# Initial guess of parameter in NCG
# Generated from a standart normal distribution N(0,1)
# k is a number of components
def Initial_Guess(N, k):
    return sc.random.randn(N, k-1)

# Calculate parameters of distribution of group labels    
def Calculate_Pi(Theta):
    
    ones = sc.ones(Theta.shape[0])
    zeros = sc.zeros((Theta.shape[0], 1))
    A = sc.log(ones + sc.sum(sc.exp(Theta), axis=1))
    Pi = sc.concatenate([Theta, zeros], axis=1)
    Pi -= sc.outer(A, sc.ones(Pi.shape[1]))
    Pi = sc.exp(Pi)
    
    return Pi	


# Hyperparameter described in the paper mentioned above
# k-dimensional vector,
# where k is a number of clusters
def Calculate_Alpha(Pi):
    return sc.ones(Pi.shape[1]) + Pi.T.dot(sc.ones(Pi.shape[0]))

	
# Hyperparameter described in the paper mentioned above
# kxk matrix,
# where k is a number of clusters
def Calculate_Beta(X, Pi):
    k = Pi.shape[1]
    N = X.shape[0]
    ones = sc.ones((k, k))
    
    Y = sc.ones((N, N)) - sc.identity(N) - X
    Z = sc.ones((k, k)) - 0.5 * sc.identity(k)
    
    beta1 = ones + Z * Pi.T.dot(X.dot(Pi))
    beta2 = ones + Z * Pi.T.dot(Y.dot(Pi))
    
    return beta1, beta2

# Computes log-likelihood based on current values of parameters	
def Log_Likelihood(X, Pi, alpha, beta1, beta2):
    
    # Probability of observing an edge between clusters
    eps = 1e-10
    
    k = Pi.shape[1]
    N = X.shape[0]
    ones = sc.ones((k, k))
    Y = sc.ones((N, N)) - sc.identity(N) - X
    Z = sc.log(eps) * Pi.T.dot(X.dot(Pi)) + sc.log(1 - eps) * Pi.T.dot(Y.dot(Pi))
    Z = sc.tril(Z, -1)
    
    A = sc.sum(Pi * sc.log(Pi))
    B_alpha = sc.log(nbeta(alpha) / nbeta(sc.ones(alpha.shape[0])))
    
    B_beta = sc.special.beta(beta1, beta2) / sc.special.beta(1, 1)
    B_beta = sc.diag(sc.log(B_beta))
    
    L = sc.sum(Z) + A + B_alpha + sc.sum(B_beta)
    
    return L
	
# Generalization of beta function
# See paper notations
def nbeta(alpha):
    gamma = sc.special.gamma
    return sc.prod(gamma(alpha)) / gamma(sc.sum(alpha))

# Computes Natural Conjugate Gradient
#
def Calculate_NCG(X, Pi, alpha, beta1, beta2, d_old, g_norm_old):
    
    # Probability of observing an edge between clusters
    eps = 1e-10
    
    N = X.shape[0]
    k = Pi.shape[1]
    
    psi = sc.special.psi
    Y = sc.ones((N, N)) - sc.identity(N) - X
    
    d_log_B1 = psi(beta1) - psi(beta1 + beta2)
    d_log_B2 = psi(beta2) - psi(beta1 + beta2)
    
    d_log_B1 = sc.diag(sc.diag(d_log_B1)) + sc.log(eps) * sc.triu(sc.ones((k, k)), 1) + sc.log(eps) * sc.tril(sc.ones((k, k)), -1)
    d_log_B2 = sc.diag(sc.diag(d_log_B2)) + sc.log(1 - eps) * sc.triu(sc.ones((k, k)), 1) + sc.log(1 - eps) * sc.tril(sc.ones((k, k)), -1)
    
    # L_prime_i,j = dL / dPi_i,j
    L_prime = X.dot(Pi.dot(d_log_B1.T)) + Y.dot(Pi.dot(d_log_B2.T)) \
        + sc.outer(sc.ones(N), psi(alpha)) - sc.log(Pi) - sc.outer(sc.ones(N), sc.ones(k))
    
    u = sc.concatenate([sc.identity(k - 1), -sc.ones((1, k-1))], axis=0)
    
	# Natural gradient
    g = L_prime.dot(u)
    
    # Norm of natural gradient
    g_norm = 0
    for i in range (N):
        l = L_prime[i, :]
        p = Pi[i, :]
        B = sc.diag(p) - sc.outer(p, p)
        g_norm += l.dot(B.dot(l))
    
	# Natural Conjugate Gradient
    d = g + g_norm / g_norm_old * d_old
    
    return d, g_norm