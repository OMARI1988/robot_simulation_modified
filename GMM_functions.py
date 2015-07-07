from mpl_toolkits.mplot3d import Axes3D
import colorsys
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import numpy as np
import pickle

def read_file(x):

	directory = '/home/omari/ros_ws/src/baxter_demos/share/'
	pkl_file = open(directory+x+'.pkl', 'rb')
	print ' - loading data..'
	data1 = pickle.load(pkl_file)
	print ' - file loaded..'
	hyp = {}
	hyp['valid_HSV_hyp'] = []
	hyp['valid_dis_hyp'] = []
	hyp['valid_dir_hyp'] = []
	POINTS_HSV = data1['HSV']
	POINTS_SPA = data1['SPA']
	hyp = data1['hyp']
	return hyp

def find_RGB_map(N):
	HSV_tuples = [(x*1.0/N, 1.0, 1.0) for x in range(N)]
	return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

#######################################
#----------- plot function -----------#
#######################################
# plot function takes as input the Data vector, the best GMM, the BIC results, the maximum components number, and the covariance types
def plot_data(X, best_gmm, bic, k, cv_types, GT, fig):
	plt.figure(num=fig, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
	#plt.figure(fig)

	n_samples = len(X)
	if k>(n_samples+1):
		k = n_samples+1

	n_components_range = range(1, k)

	bic = np.array(bic)
	color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	spl = plt.subplot(2, 1, 1)
	spl.cla()
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	color_iter = find_RGB_map(clf.n_components)
	# Plot the winner
	spl = plt.subplot(2, 1, 2, projection='3d')
	spl.cla()
	Y_ = clf.predict(X)
	print Y_
	for i in range(clf.n_components):
		if not np.any(Y_ == i):
			continue
		c=color_iter[i]
		print c
		if np.size(X[0]) == 1:
			spl.scatter([r for r in X[Y_ == i, 0]], [.5 for r in X[Y_ == i, 0]], [.5 for r in X[Y_ == i, 0]], c=c, marker='o', alpha=1.0)
		if np.size(X[0]) == 2:
			spl.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [.5 for r in X[Y_ == i, 0]], c=color_iter[i], marker='o')
		if np.size(X[0]) == 3:
			spl.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [r for r in X[Y_ == i, 2]], c=color_iter[i], marker='o')

	plt.xlim(-.2, 1.2)
	plt.ylim(-.2, 1.2)
	#plt.xticks(())
	#plt.yticks(())
	plt.title('Selected GMM: full model, '+str(clf.n_components)+' components')
	plt.subplots_adjust(hspace=.35, bottom=.02)

	if GT != 0:
		spl = plt.subplot(3, 1, 3)
		spl.cla()
		make_ellipses2(GT, spl)
		make_ellipses(clf, spl)
		plt.xlim(-20, 50)
		plt.ylim(-50, 50)



##############################################
#----------- plot winner function -----------#
##############################################
# plot function takes as input the Data vector, the best GMM, the BIC results, the maximum components number, and the covariance types
def plot_data_winner(X, best_gmm, bic, k, cv_types, fig, result, word):
	plt.figure(fig)

	n_samples = len(X)
	if k>(n_samples+1):
		k = n_samples+1

	n_components_range = range(1, k)

	bic = np.array(bic)
	color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	spl = plt.subplot(3, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	Best_number_of_clusters = np.mod(bic.argmin(), len(n_components_range)) + 1

	color_iter = find_RGB_map(Best_number_of_clusters)
	# Plot all clusters
	splot = plt.subplot(3, 1, 2, projection='3d')
	Y_ = clf.predict(X)
	for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_, color_iter)):
		if not np.any(Y_ == i):
			continue
		splot.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [r for r in X[Y_ == i, 2]], c=color, marker='o')

	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model, '+str(Best_number_of_clusters)+' components')

	# Plot the winner
	splot = plt.subplot(3, 1, 3, projection='3d')
	Y_ = clf.predict(X)
	for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_, color_iter)):
		if np.all(mean == result['mean']):
			if not np.any(Y_ == i):
				continue
			splot.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [r for r in X[Y_ == i, 2]], c='r', marker='o')
		else:
			if not np.any(Y_ == i):
				continue
			splot.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [r for r in X[Y_ == i, 2]], c='k', marker='o')

	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xticks(())
	plt.yticks(())
	plt.title(word+' cluster with a score of : '+str(result['score']))
	plt.subplots_adjust(hspace=.35, bottom=.02)

"""
######################################
#----------- Test GMM-BIC -----------#
######################################
def test_gmm(X, best_gmm, hyp, word):


    def max_min(delta,x):
	if x > delta[1]:
		delta[1] = x
	if x < delta[0]:
		delta[0] = x
	return delta

    #print best_gmm.n_components
    Y_ = best_gmm.predict(X)
    #print Y_
    max_score_n = 0

    result = {}
    for k, (mean, covar) in enumerate(zip(best_gmm.means_, best_gmm._get_covars())):
	Np = float(len(X[Y_ == k,0]))
	Nm = float(hyp['hyp'][word]['counter_HSV'])
	#print Nm
	score_n = 1.0/(1.0 + np.abs(1.0 - Np/Nm))

	correct_pc = 0
	frame_counter = 0
	delta_x = [np.inf,-np.inf]
	delta_y = [np.inf,-np.inf]
	delta_z = [np.inf,-np.inf]
	for frame in hyp['hyp'][word]['point_HSV_x']:
		frame_counter += 1
		x_val = hyp['hyp'][word]['point_HSV_x'][frame]
		y_val = hyp['hyp'][word]['point_HSV_y'][frame]
		z_val = hyp['hyp'][word]['point_HSV_z'][frame]
		#compute the deltas
		for p in range(len(x_val)):
			if best_gmm.predict([[x_val[p]/200.0,y_val[p]/200.0,z_val[p]/100.0]])[0] == k:
				delta_x = max_min(delta_x,x_val[p]/200.0)
				delta_y = max_min(delta_y,y_val[p]/200.0)
				delta_z = max_min(delta_z,z_val[p]/100.0)
		# compute the correct number of points in each frames
		for p in range(len(x_val)):
			if best_gmm.predict([[x_val[p]/200.0,y_val[p]/200.0,z_val[p]/100.0]])[0] == k:
				correct_pc += 1
				break
	score_c = 1.0/(1.0 + np.abs(1.0 - float(correct_pc)/float(frame_counter)))

	dx = 1-(delta_x[1]-delta_x[0])
	dy = 1-(delta_y[1]-delta_y[0])
	dz = 1-(delta_z[1]-delta_z[0])

	if dx == np.inf or dy == np.inf or dz == np.inf:
		dx = 0
		dy = 0
		dz = 0

	score_F = np.min([dx,dy,dz])
	print word, 'cluster number ',k,' score is ',score_n*score_c*score_F
	if score_n*score_c*score_F > max_score_n:
		#print 'we have a winner !'
		#print score_n,score_c
		max_score_n = score_n*score_c*score_F
		#mean_n = mean
		#covar_n = covar
		result['score'] = max_score_n
		result['score_n_all'] = score_n
		result['score_c_all'] = score_c
		result['Y_all'] = Y_
		result['k'] = k
		result['mean'] = mean
		result['covar'] = covar
    return result
"""
#----------------------------------------------------------------------------------------------------------------#
def make_ellipses(gmm, ax):
    for n, color in enumerate(find_RGB_map(gmm.n_components)):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 2
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

#----------------------------------------------------------------------------------------------------------------#
def make_ellipses2(GT, ax):
    means = GT[0]
    covars = GT[1]
    for n in range(len(covars)):
	v, w = np.linalg.eigh(np.vstack(covars[n])[:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 2
        ell = mpl.patches.Ellipse(means[n], v[0], v[1],
                                  180 + angle, color='k')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

#----------------------------------------------------------------------------------------------------------------#
# Find if the covariance matrix is positive def or not
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 1.0e-9)

#----------------------------------------------------------------------------------------------------------------#
# Find if the covariance matrix is singular or not (False means singular)
def is_invertible(a):
	return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

# needs thinking
#----------------------------------------------------------------------------------------------------------------#
def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


