from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP

n = 1e5
x = y = NP.linspace(0, 1, 1000)
X, Y = NP.meshgrid(x, y)
Z2 = ML.bivariate_normal(X, Y, .05, .05, .55/.8, .45/.8)      #sigmax, sigmay,x,y
ZD = Z2
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
    PLT.plot([i/8.0,i/8.0],[0,1],'k',linewidth=2.0)
    PLT.plot([0,1],[i/8.0,i/8.0],'k',linewidth=2.0)
cb = PLT.colorbar()
cb.set_label('mean value')
PLT.savefig('/home/omari/Dropbox/myfig.png')
PLT.show()  
