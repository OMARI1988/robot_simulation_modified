import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# a GMM based on BCC for a data X and a maximum number of components k and coverince types cv_types
def gmm_bic(X, k, cv_types):

	n_samples = len(X)
	if k>(n_samples+1):
		k = n_samples+1

	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, k)
	for cv_type in cv_types:
	    for n_components in n_components_range:
		# Fit a mixture of Gaussians with EM
		gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
		gmm.fit(X)
		bic.append(gmm.bic(X))
		if bic[-1] < lowest_bic:
		    lowest_bic = bic[-1]
		    best_gmm = gmm

	return best_gmm, bic

