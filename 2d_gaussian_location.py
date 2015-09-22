from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP

n = 1e5
x = y = NP.linspace(0, 1, 1000)
X, Y = NP.meshgrid(x, y)

## Bottom Left
# Z1 = ML.bivariate_normal(X, Y, .065, .05, 1-((.969308507959*.7+.05)/.8), 0.0644887710903/.8)      #sigmax, sigmay,x,y

## Top right
# Z1 = ML.bivariate_normal(X, Y, .045, .055, 1-(0.0544887710903/.8),((.969308507959*.7+.05)/.8))      #sigmax, sigmay,x,y

## Top left
Z1 = ML.bivariate_normal(X, Y, .04, .05, (0.04944887710903/.8),((.969308507959*.7+.05)/.8))      #sigmax, sigmay,x,y

## Bottom Right
Z1 = ML.bivariate_normal(X, Y, .035, .047, ((.999308507959*.7+.05)/.8),(0.0444887710903/.8))      #sigmax, sigmay,x,y

## middle
Z1 = ML.bivariate_normal(X, Y, .085, .097, (0.475241963479*.7+.05)/.8,(0.554526698269*.7+.05)/.8)      #sigmax, sigmay,x,y

## center
Z1 = ML.bivariate_normal(X, Y, .077, .067, (0.539486875965*.7+.05)/.8,(0.529321116022*.7+.05)/.8)      #sigmax, sigmay,x,y

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
PLT.savefig('/home/omari/Dropbox/Reports/AAAI16/Muhannad_with_modifications/pics/center.png')
PLT.show()
