import numpy as np
from sklearn.cluster import KMeans

def create_local_sim(A, k_hat):
	# Creating a dense similarity matrix with local scale
	# (sigma's) as shown in the slides, page 6
	# A: n*d data matrix where is n are the sample in d dim
	# k_hat: as defined in the slide
	# Return value: n*n similarity matrix
	n = A.shape[0]
	# error check to make sure there as many samples to satisfy the kth dist
	assert k_hat < n

	# TODO
	# create similarity matrix to get values of data points that are close together through graph
	W = np.zeros((n, n))
	
	# get the data squared and collect the mag of each data point in one vector
	# dot A with itself to get value supposed to subtract same as hw1
	#sum these terms to create a numerator of exp of Wi,j
	data_sq = A*A
	data_sq = np.sum(data_sq,axis=1).reshape(n,1) # sum each row (features of each data pt)
	AX_terms = -2*np.dot(A,A.T)
	numer = data_sq + AX_terms + data_sq.T

	# each element i,j of numer is the euclidean dist sq of each data point i,j 
	k_nearest_index = np.argsort(np.absolute(numer))[:,k_hat+1].flatten()
	
	# need to use sqrt of the euclid dist found in numer
	# use reshape so when dotted, makes matrix for the denom
	k_nearest = np.array([np.sqrt(numer[i][j]) for i,j in enumerate(k_nearest_index)]).reshape(n,1)
	#print k_nearest
	denom = 2*np.dot(k_nearest,k_nearest.T)
	
	W = np.exp(-1*numer/denom) 
	return W

def run(W, k, ncut):
	# Perform spectral clustering
	# W: n*n similarity matrix
	# k: number of clusters
	# ncut: perform ratio cut when ncut==False; normalized cut when ncut==True
	# Return value: n-vector that contains cluster assignments
	assert W.shape[0] == W.shape[1]
	n = W.shape[0]
	
	# create degree matrix and Laplacian from the degree and similarity matrices
	# generate eigen values and vectors
	D = np.sum(W,axis=1)
	L = np.diag(D) - W
	
	# TODO
	labels = np.zeros(n, dtype=np.uint32)
	if ncut == False:
		evals, evec= np.linalg.eig(L)
	else:
		D_inv_sqrt = 1/np.sqrt(D)
		evals, evec= np.linalg.eig(D_inv_sqrt.reshape(n,1)*L*D_inv_sqrt)
	
	# main error of this portion of the code was
	# 1) forgetting to account for the floats in the eq leading to zero for all eigenval
	# 2) finding the correct way to get indicies 
	k_evals = np.sort(evals)[:k]
	#grab the indicies of the smallest eigen vals
	e_index = np.array([np.where(evals == val) for val in k_evals]).reshape(k)
	evecs = np.zeros((k,n),dtype=np.float64)
	for i in xrange(k):
		evecs[i] = evec[:,e_index[i]]
	# ensure the evecs are the cols of the array to run Kmeans on
	evecs = np.array(evecs).T
	kmeans = KMeans(n_clusters=k).fit(evecs)
	labels = kmeans.labels_
	return labels
