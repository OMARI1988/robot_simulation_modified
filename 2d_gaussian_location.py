from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP
import pickle

#
# gmm_obj       = pickle.load( open( "/home/omari/Dropbox/robot_modified/EN/pickle/gmm_obj_1000.p", "rb" ) )
# print gmm_obj['bottom left']['F_POS']['gmm']
#
analysis                    = pickle.load( open( "/home/omari/Dropbox/robot_modified/EN/pickle/analysis_01000.p", "rb" ) )
all_valid_hypotheses   = analysis[2]
print all_valid_hypotheses['F_POS'].keys()
print all_valid_hypotheses['F_POS']['bottom left']

n = 1e5
x = y = NP.linspace(0, 1, 1000)
X, Y = NP.meshgrid(x, y)

## Bottom Left
Z1 = ML.bivariate_normal(X, Y, .065, .05, (.0244887710903*.7+.05)/.8, (0.969308507959*.7+.5)/.8)      #sigmax, sigmay,x,y

## Top right
Z1 = ML.bivariate_normal(X, Y, .045, .055, 1-(0.0544887710903/.8),((.969308507959*.7+.05)/.8))      #sigmax, sigmay,x,y

## Top left
Z1 = ML.bivariate_normal(X, Y, .04, .05, (0.04944887710903/.8),((.969308507959*.7+.05)/.8))      #sigmax, sigmay,x,y

## Bottom Right
Z1 = ML.bivariate_normal(X, Y, .035, .047, ((.999308507959*.7+.05)/.8),(0.0444887710903/.8))      #sigmax, sigmay,x,y

## middle
Z1 = ML.bivariate_normal(X, Y, .085, .097, (0.475241963479*.7+.05)/.8,(0.554526698269*.7+.05)/.8)      #sigmax, sigmay,x,y

## center
Z1 = ML.bivariate_normal(X, Y, .077, .067, (0.539486875965*.7+.05)/.8,(0.529321116022*.7+.05)/.8)      #sigmax, sigmay,x,y

## left_back
Z1 = ML.bivariate_normal(X, Y, .077, .067, (0.044473845502*.7+.05)/.8,((1-0.970782957933)*.7+.05)/.8)      #sigmax, sigmay,x,y

## right back
x1 = 0.0130292063894
y1 = 0.0310865327145
Z1 = ML.bivariate_normal(X, Y, .077, .067, ((1-0.0130292063894)*.7+.05)/.8,(0.0310865327145*.7+.05)/.8)      #sigmax, sigmay,x,y

## right top
x1 = 0.0105606971721
y1 = 0.989914833652
Z1 = ML.bivariate_normal(X, Y, .047, .1167, ((1-x1)*.7+.05)/.8,(y1*.7+.05)/.8)      #sigmax, sigmay,x,y

## far left
x1 = 0.989068095127
y1 = 0.994278396435
Z1 = ML.bivariate_normal(X, Y, .17, .127, ((1-x1)*.7+.05)/.8,(y1*.7+.05)/.8)      #sigmax, sigmay,x,y


ZD = Z1
# Z1 = ML.bivariate_normal(X, Y, .05, .05, .35/.8, .35/.8)      #sigmax, sigmay,x,y
# Z2 = ML.bivariate_normal(X, Y, .05, .05, .65/.8, .55/.8)      #sigmax, sigmay,x,y
# Z3 = ML.bivariate_normal(X, Y, .05, .05, .65/.8, .15/.8)      #sigmax, sigmay,x,y
# Z4 = ML.bivariate_normal(X, Y, .05, .05, .15/.8, .15/.8)      #sigmax, sigmay,x,y
# Z5 = ML.bivariate_normal(X, Y, .05, .05, .05/.8, .15/.8)      #sigmax, sigmay,x,y
# Z6 = ML.bivariate_normal(X, Y, .05, .05, .05/.8, .45/.8)      #sigmax, sigmay,x,y
# ZD = Z1#+Z2+Z3+Z4+Z5+Z6
x = X.ravel()
y = Y.ravel()
z = ZD.ravel()
gridsize=150
PLT.subplot(111)

# if 'bins=None', then color of each hexagon corresponds directly to its count
# 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then
# the result is a pure 2D histogram

PLT.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=None)
PLT.axis([x.min(), x.max(), y.min(), y.max()])
for i in range(1,8):
    PLT.plot([i/8.0,i/8.0],[0,1],'w',linewidth=2.0)
    PLT.plot([0,1],[i/8.0,i/8.0],'w',linewidth=2.0)
cb = PLT.colorbar()
cb.set_label('mean value')
PLT.savefig('/home/omari/Dropbox/Reports/AAAI16/Muhannad_with_modifications/pics/loc_far_left.png')
PLT.show()
