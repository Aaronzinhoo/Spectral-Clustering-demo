import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import spectral_clustering

colors = ['red', 'blue', 'green', 'black', 'pink']
k_hat = 7       # k_hat as defined in the slides

def run(data_file, k, fig_file):
	# data_file: contains n*d data matrix
	# k: number of clusters
	# fig_file: for saving the clustering results

	A = np.genfromtxt(data_file, delimiter=',')  # data matrix
	n = A.shape[0]  # number of data points
	plt.figure()
	
	# To compute the similarity matrix 'W'
	# and clustering assignment 'labels' with
	# the following code
	W = spectral_clustering.create_local_sim(A, k_hat)
	for ncut in [False, True]:
		labels = spectral_clustering.run(W, k, ncut)
		plt.subplot(1, 2, ncut+1, aspect='equal')
		for i in range(k):
			subset = np.where(labels == i)[0]
			plt.plot(A[subset, 0], A[subset, 1], 'o', c=colors[i], markeredgewidth=0.0, markersize=3)
		plt.title('ncut={0}'.format(ncut))
	
	plt.tight_layout()
	plt.savefig(fig_file)

run('data1.txt', 3, 'spectral1.eps')
run('data2.txt', 5, 'spectral2.eps')
