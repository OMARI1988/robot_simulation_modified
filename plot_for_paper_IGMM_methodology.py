import numpy as np

from pylab import figure, show, rand
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

NUM = 4

start = [[.7,.7],[.2,.3],[.4,.71]]
end = [[.67,.67],[.16,.3],[.8,.33]]
final = [[.72,.69],[.18,.3]]

ells = []
ells.append(Ellipse(xy=start[0], width=.1, height=.3, angle=25, facecolor=[.2,.5,.9], alpha=.5))
ells.append(Ellipse(xy=start[1], width=.2, height=.1, angle=-15, facecolor=[.2,.5,.9], alpha=.5))
ells.append(Ellipse(xy=start[2], width=.2, height=.4, angle=-35, facecolor=[.2,.5,.9], alpha=.5))
#ells.append(Ellipse(xy=[.15,.8], width=.14, height=.2, angle=-8, facecolor=[.2,.5,.9], alpha=.5))

ells1 = []
ells1.append(Ellipse(xy=start[0], width=.1, height=.3, angle=25, facecolor=[.2,.5,.9], alpha=.5))
ells1.append(Ellipse(xy=start[1], width=.2, height=.1, angle=-15, facecolor=[.2,.5,.9], alpha=.5))
ells1.append(Ellipse(xy=start[2], width=.2, height=.4, angle=-35, facecolor=[.2,.5,.9], alpha=.5))
#ells1.append(Ellipse(xy=[.15,.8], width=.14, height=.2, angle=-8, facecolor=[.2,.5,.9], alpha=.5))

ells2 = []
ells2.append(Ellipse(xy=end[0], width=.12, height=.33, angle=21, facecolor=[.2,1,.3], alpha=.5))
ells2.append(Ellipse(xy=end[1], width=.18, height=.15, angle=-17, facecolor=[.2,1,.3], alpha=.5))
ells2.append(Ellipse(xy=end[2], width=.12, height=.15, angle=-35, facecolor=[.2,1,.3], alpha=.5))
ells3 = []
ells3.append(Ellipse(xy=end[0], width=.12, height=.33, angle=21, facecolor=[.2,1,.3], alpha=.5))
ells3.append(Ellipse(xy=end[1], width=.18, height=.15, angle=-17, facecolor=[.2,1,.3], alpha=.5))
ells3.append(Ellipse(xy=end[2], width=.12, height=.15, angle=-35, facecolor=[.2,1,.3], alpha=.5))

ells4 = []
# ells4.append(Ellipse(xy=[.7,.7], width=.1, height=.3, angle=25, facecolor=[.2,.5,.9], alpha=.5))
# ells4.append(Ellipse(xy=[.77,.67], width=.12, height=.33, angle=21, facecolor=[.2,1,.3], alpha=.5))
ells4.append(Ellipse(xy=final[0], width=.15, height=.34, angle=22, facecolor=[.2,.5,.9], alpha=.5))

# ells4.append(Ellipse(xy=[.2,.3], width=.2, height=.1, angle=-15, facecolor=[.2,.5,.9], alpha=.5))
# ells4.append(Ellipse(xy=[.16,.3], width=.18, height=.15, angle=-17, facecolor=[.2,1,.3], alpha=.5))
ells4.append(Ellipse(xy=final[1], width=.25, height=.2, angle=-16, facecolor=[.2,.5,.9], alpha=.5))

ells4.append(Ellipse(xy=start[2], width=.2, height=.4, angle=-35, facecolor=[.2,.5,.9], alpha=.5))
ells4.append(Ellipse(xy=end[2], width=.12, height=.15, angle=-35, facecolor=[.2,.5,.9], alpha=.5))

f, ax = plt.subplots(2, 2)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for e in ells:
	ax[0,0].add_artist(e)
	e.set_clip_box(ax[0,0].bbox)
ax[0,0].text(start[0][0]+.08,start[0][1]+.1, r' $F_2=N$', fontsize=15)
ax[0,0].text(start[1][0]-.1,start[1][1]+.07, r' $F_3=N$', fontsize=15)
ax[0,0].text(start[2][0]-.27,start[2][1]+.08, r' $F_1=N$', fontsize=15)

ax[0,0].text(start[0][0]-.02,start[0][1], r' $2$', fontsize=15)
ax[0,0].text(start[1][0],start[1][1]-.03, r' $3$', fontsize=15)
ax[0,0].text(start[2][0],start[2][1], r' $1$', fontsize=15)

ax[0,0].set_title('(a) Phrase-Feature histogram')


for e in ells2:
	ax[0,1].add_artist(e)
	e.set_clip_box(ax[0,1].bbox)
ax[0,1].set_title('(b) New data GMM')

ax[0,1].text(end[0][0]+.01, end[0][1]+.11, r' $1_{new}$', fontsize=15)
ax[0,1].text(end[1][0]-.01, end[1][1]+.09, r' $3_{new}$', fontsize=15)
ax[0,1].text(end[2][0], end[2][1]-.15, r' $2_{new}$', fontsize=15)


for e in ells1:
	ax[1,0].add_artist(e)
	e.set_clip_box(ax[1,0].bbox)
for e in ells3:
    ax[1,0].add_artist(e)
    e.set_clip_box(ax[1,0].bbox)
ax[1,0].set_title('(c) Update Phrase-Feature histogram')
color = ['k','b','r']
#for k,i in enumerate(start):
# line, = ax[1,0].plot(np.linspace(.8,start[0][0],2),np.linspace(.88,start[0][1],2), '--')
# plt.setp(line, color=color[0], linewidth=1.0)
# line, = ax[1,0].plot(np.linspace(.8,end[0][0],2),np.linspace(.88,end[0][1],2), '--')
# plt.setp(line, color=color[0], linewidth=1.0)




for e in ells4:
    ax[1,1].add_artist(e)
    e.set_clip_box(ax[1,1].bbox)
ax[1,1].text(.13, .8, r' $F_1=N$', fontsize=15)
ax[1,1].text(.67, .87, r' $F_2=N+1$', fontsize=15)
ax[1,1].text(.1, .1, r' $F_3=N+1$', fontsize=15)
ax[1,1].text(.8, .14, r' $F_4=1$', fontsize=15)

ax[1,1].set_title('(d) IGMM output')

for i in range(2):
	for j in range(2):
		plt.sca(ax[i, j])
		plt.xticks([0,.25,.5,.75,1], [])
		plt.yticks([0,.25,.5,.75,1], [])
		plt.grid()

show()
