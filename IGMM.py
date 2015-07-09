import itertools
import numpy as np
from scipy import linalg
from GMM_BIC import *
from scipy.stats import ks_2samp
import numpy as np
from collections import namedtuple
import scipy
#import copy
from sklearn import mixture
from GMM_functions import *
#----------------------------------------------------------------------------------------------------------------#
# a GMM based on BCC for a data X and a maximum number of components k and coverince types cv_types
def igmm(X,gmm_N,gmm_M, N, M):

	# initilizations of the W statitics test
	W_statistics = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	W_results = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	# initilizations of the Hotelling's T squared test
	H_statistics = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	H_results = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	# initilization of the new gmm_N_M
	gmm_N_M = copy.copy(gmm_N)
	# 4. assign each new data in X to a cluster in gmm_M
	Y_ = gmm_M.predict(X)
	# 5. for every component in gmm_M
	for k in range(gmm_M.n_components):
		# check if kth component is postivie definite
		if not is_pos_def(gmm_M._get_covars()[k]):
			#print gmm_M._get_covars()[k]
			#print nearPD(gmm_M._get_covars()[k], nit=10)	########################################################################
			print 'not positive definite in the new clustes'
		# 6. Let Dk be the collection of all the data in component k.
		Dk = X[Y_==k]
		# 7. for every component in gmm_N
		for j, (mean_j, covar_j) in enumerate(zip(gmm_N.means_, gmm_N._get_covars())):
			# 8. calculate the W_statistic to determine if Dk has equal coverinece with covar_j
			if not is_pos_def(covar_j):
				#print covar_j
				#print nearPD(covar_j, nit=10) ########################################################################
				print 'not positive definite in the old clusters j'
			W_statistics[k][j],W_results[k][j] = Covariance_Test(Dk, covar_j)
			# 9. if Dk passed the W statitics test.
			if W_results[k][j] == 1.0:
				# 10. Perform the Hotelling's T squared test to see if Dk has the same mean as mean_j
				H_statistics[k][j],H_results[k][j] = Mean_Test(Dk, mean_j, covar_j)

				# 11. if Dk passed the Hotelling's T squared test
				if H_results[k][j] == 1.0:
					# 19 create a new component g N+M by merging j and k
					Mk = float(len(Dk))		# the number of points in new cluster k
					gmm_N_M = update_gmm(gmm_N_M, gmm_M, j, k, N, M, Mk)
	# 22. for each remaining components in k create a new component in g N+M
	row_sum = np.sum(H_results, axis=1)
	for k in range(gmm_M.n_components):
			# the number of points in new cluster k
			Dk = X[Y_==k]
			Mk = float(len(Dk))
			if row_sum[k] == 0.0:
				gmm_N_M = create_gmm(gmm_N_M, gmm_M, k, N, M, Mk)
	# 24. for each remaining component j in gmm_N
	col_sum = np.sum(H_results, axis=0)
	for j in range(gmm_N.n_components):
		if col_sum[j] == 0.0:
			gmm_N_M = update_gmm_weights(gmm_N_M, N, M, j)
	# Print the W statitics on the command window
	print_W_statitics(W_results,gmm_M.n_components,gmm_N.n_components,'W')
	print_W_statitics(H_results,gmm_M.n_components,gmm_N.n_components,'H')
	# 27. Merge statistically equivalent components in gmm_N_M
	sample = 50
	while 1:
		for k in range(gmm_N_M.n_components):
			mean1 = gmm_N_M.means_[k]
			cov1 = gmm_N_M._get_covars()[k]
			Dk = np.random.multivariate_normal(mean1,cov1,sample)
			flag = 0
			for j, (mean_j, covar_j) in enumerate(zip(gmm_N_M.means_, gmm_N_M._get_covars())):
				if k == j: continue
				#Check uf the covariance matrix is postive definite
				if not is_pos_def(covar_j):
					print 'not positive definite in the merged clusters'
					gmm_N_M = remove_component(gmm_N_M,j)
					flag = 1
					break
				# 8. calculate the W_statistic to determine if Dk has equal coverinece with covar_j
				W_statistics1 = Covariance_Test(Dk,covar_j)
				# 9. if Dk passed the W statitics test.
				if W_statistics1[1] == 1.0:
					# 10. Perform the Hotelling's T squared test to see if Dk has the same mean as mean_j
					H_statistics1 = Mean_Test(Dk, mean_j, covar_j)
					# 11. if Dk passed the Hotelling's T squared test
					if H_statistics1[1] == 1.0:
						print j,'and',k,'are now merged in gmm_N_M'
						gmm_N_M = merge_components(gmm_N_M, k, j, N)
						flag = 1
						break
			if flag == 1:
				break
		if flag == 0:
			print 'done merging'
			break
	return gmm_N_M

#----------------------------------------------------------------------------------------------------------------#
# 3.1 Testing for equality to a covariance matrix
def Covariance_Test(x,covar):
	# finding Lo (lower triangular matrix obtained by Cholesky decomposition of covar)
	Lo = np.linalg.cholesky(covar)
	Lo_inv = np.linalg.inv(Lo)
	d = len(Lo_inv)
	# finding yi
	yi = {}
	for i in x:
		y = np.dot(Lo_inv,np.vstack(i))
		for n in range(d):
			if n in yi:
				yi[n].append(y[n][0])
			else:
				yi[n] = [y[n][0]]
	# Computing the sample covariance matrix of yi, Sy
	Sy = np.zeros((d,d),dtype=np.float)
	for i in yi:
		Sy[i,i] = np.cov(yi[i])
	# computing the trace of a matrix Sy-I
	I = np.identity(d)
	c1 = np.matrix.trace((Sy - I)**2)/float(d)
	# compute the trace of matix Sy
	n = float(len(x))	# n is the number of the points in kluster k
	c2 = ( float(d)/n ) * ( 1/float(d) * np.matrix.trace(Sy) )**2
	# compute W
	W = c1 - c2 + float(d)/n
	# perform the test
	alpha = .005
	test = n*W*d/2.0
	p_value = scipy.stats.chi2.pdf(test, (d*(d+1))/2)
	result = 0
	if p_value>alpha:
		result = 1.0
	return test,result

#----------------------------------------------------------------------------------------------------------------#
# 3.2 Testing for equality to a mean vector
def Mean_Test(x,mean,S):
	# compute sample mean
	d = len(x[0])
	n = len(x)
	x_mean = []
	for i in range(d):
		x_mean.append(np.mean(x[:,i]))
	# compute the sample covariance
	#S = np.cov(x.T)
	S_inv = np.linalg.inv(S)
	# computing the T squared test
	c1 = np.transpose([mean - x_mean])
	T = n*np.dot(np.dot(np.transpose(c1),S_inv),c1)
	F = T[0][0]*float(n-d)/float(d*(n-1))
	alpha = .005 #Or whatever you want your alpha to be.
	p_value = scipy.stats.f.pdf(F, d, n-d)
	result = 0
	if p_value>alpha:
		result = 1.0
	return F,result

#----------------------------------------------------------------------------------------------------------------#
# 3.3.1 Merging Components
def update_gmm(gmm_N_M, gmm_M, j, k, N, M, Mk):
	mu_j = gmm_N_M.means_[j]
	S_j = gmm_N_M._get_covars()[j]
	pi_j = gmm_N_M.weights_[j]
	mu_k = gmm_M.means_[k]
	S_k = gmm_M._get_covars()[k]
	pi_k = gmm_M.weights_[k]
	# update the mean
	mu = ( N*pi_j*mu_j + Mk*mu_k )/( N*pi_j + Mk )
	# update the covariance matrix
	S = ((N*pi_j*S_j + Mk*S_k)/(N*pi_j+Mk)) + ((N*pi_j*np.dot(mu_j,mu_j.T)+Mk*np.dot(mu_k,mu_k.T))/(N*pi_j+Mk)) - np.dot(mu,mu.T)
	# update the weight
	pi = ( N*pi_j + Mk )/( N + M )

	# update gmm_N_M
	gmm_N_M.means_[j] = mu
	gmm_N_M.covars_[j] = distribute_covar_matrix_to_match_covariance_type(S, gmm_N_M.covariance_type, 1)
	gmm_N_M.weights_[j] = pi
	print j,'and',k,'are now merged in gmm_N and gmm_M'
	return gmm_N_M

#----------------------------------------------------------------------------------------------------------------#
# 3.3.2 Creating Components
def create_gmm(gmm_N_M, gmm_M, k, N, M, Mk):
	# create a new component k in gmm_N_M
	gmm_N_M.n_components += 1
	# add the mean
	gmm_N_M.means_ = np.vstack((gmm_N_M.means_,gmm_M.means_[k]))
	# add the covariance
	shape = np.shape(gmm_N_M.covars_)
	if len(shape) == 3:
	    new_covar = np.zeros((shape[0]+1,shape[1],shape[2]))
	    new_covar[0:-1,:,:] = gmm_N_M.covars_
	    new_covar[-1,:,:] = gmm_M._get_covars()[k]
	    gmm_N_M.covars_ = new_covar
	# add the weights
	gmm_N_M.weights_ = np.hstack((gmm_N_M.weights_,Mk/(N+M)))
	return gmm_N_M

#----------------------------------------------------------------------------------------------------------------#
# 3.3.3 Update weights of every j in gmm_N that has no match Component in gmm_M
def update_gmm_weights(gmm_N_M, N, M, j):
	gmm_N_M.weights_[j] = (N*gmm_N_M.weights_[j])/(N+M)
	return gmm_N_M

#----------------------------------------------------------------------------------------------------------------#
# 3.3.4 Final Merging for Similar Components
def merge_components(gmm_N_M, k, j, N):
	mu_j = gmm_N_M.means_[j]
	S_j = gmm_N_M._get_covars()[j]
	pi_j = gmm_N_M.weights_[j]
	mu_k = gmm_N_M.means_[k]
	S_k = gmm_N_M._get_covars()[k]
	pi_k = gmm_N_M.weights_[k]
	# spical case
	M=0
	Mk = N*pi_k
	# update the mean
	mu = ( N*pi_j*mu_j + Mk*mu_k )/( N*pi_j + Mk )
	# update the covariance matrix
	S = ((N*pi_j*S_j + Mk*S_k)/(N*pi_j+Mk)) + ((N*pi_j*np.dot(mu_j,mu_j.T)+Mk*np.dot(mu_k,mu_k.T))/(N*pi_j+Mk)) - np.dot(mu,mu.T)
	# update the weight
	pi = pi_j + pi_k
	# update gmm_N_M
	gmm_N_M.means_[j] = mu
	gmm_N_M.covars_[j] = distribute_covar_matrix_to_match_covariance_type(S, gmm_N_M.covariance_type, 1)
	gmm_N_M.weights_[j] = pi
	# removing the kth component
	gmm_N_M = remove_component(gmm_N_M,k)
	return gmm_N_M


#----------------------------------------------------------------------------------------------------------------#
# *** Removing Components that don't have a positive definite covariance matrix
def remove_component(gmm,k):
	# removing the kth mean component
	gmm.means_ = np.delete(gmm.means_, k, axis=0)
	# removing the kth covariance component
	shape = np.shape(gmm.covars_)
	new_covar = np.zeros((shape[0]-1,shape[1],shape[2]))
	counter = 0
	for i in range(gmm.n_components):
		if i == k: continue
		new_covar[counter] = gmm.covars_[i]
		counter += 1
	gmm.covars_ = new_covar
	# removing the kth weight component
	gmm.weights_ = np.delete(gmm.weights_, k, axis=0)
	gmm.n_components -= 1
	return gmm


#----------------------------------------------------------------------------------------------------------------#
# fixing the covariance matrix
def fixing_covar(covar_j,ctype,n_components):

        if ctype == 'full':
            return covar_j
        elif ctype == 'diag':
            return [np.diag(cov) for cov in covar_j]
        elif ctype == 'tied':
            return [covar_j] * n_components
        elif ctype == 'spherical':
            return [np.diag(cov) for cov in covar_j]

	if ctype != 'full':
		N = len(covar_j)
		covar = np.zeros((N,N),dtype=np.float)
		for i, c in enumerate(covar_j):
			covar[i,i] = c
		covar_j = covar
	return covar_j

#----------------------------------------------------------------------------------------------------------------#
# Print the W_statistics matrix on command window
def print_W_statitics(W,M,N,msg):
	print
	print('***************************************************')
	if msg == 'W':
		print('W statistics for %d new clusters and %d old clusters.' %(M,N))
	if msg == 'H':
		print('Hotelling test for %d new clusters and %d old clusters.' %(M,N))
	print('***************************************************')
	print
	rows = ['W_statistics']
	for i in range(N):
		rows.append('old_'+str(i))
	Row = namedtuple('Row',rows)

	all_data = []
	for i in range(M):
		d = {'W_statistics': 'new_'+str(i)}
		for j in range(N):
			d['old_'+str(j)] = str(W[i][j])
		data = Row(**d)
		all_data.append(data)
	pprinttable(all_data)
	print

#----------------------------------------------------------------------------------------------------------------#
# The function that does the printing
def pprinttable(rows):
	if len(rows) > 1:
		headers = rows[0]._fields
		lens = []
		for i in range(len(rows[0])):
			lens.append(len(max([x[i] for x in rows] + [headers[i]],key=lambda x:len(str(x)))))
		formats = []
		hformats = []
		for i in range(len(rows[0])):
			if isinstance(rows[0][i], int):
				formats.append("%%%dd" % lens[i])
			else:
				formats.append("%%-%ds" % lens[i])
			hformats.append("%%-%ds" % lens[i])
		pattern = " | ".join(formats)
		hpattern = " | ".join(hformats)
		separator = "-+-".join(['-' * n for n in lens])
		print hpattern % tuple(headers)
		print separator
		for line in rows:
			print pattern % tuple(line)
	elif len(rows) == 1:
		row = rows[0]
		hwidth = len(max(row._fields,key=lambda x: len(x)))
		for i in range(len(row)):
			print "%*s = %s" % (hwidth,row._fields[i],row[i])

#----------------------------------------------------------------------------------------------------------------#
# distribute_covar_matrix_to_match_covariance_type
def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template
    """
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv
