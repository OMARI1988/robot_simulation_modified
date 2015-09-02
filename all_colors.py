import numpy as np
import cv2
import colorsys
import pickle
import operator

analysis                    = pickle.load( open( "/home/omari/Dropbox/robot_modified/EN/pickle/analysis_01000.p", "rb" ) )
# self.correct_commands       = analysis[0]
# self.wrong_commands         = analysis[1]
all_valid_hypotheses   = analysis[2]

colors = ['red','green','blue','cyan','sky','black','yellow','magenta','pink','purple','turquoise']
for color in colors:
	# sorted_x = sorted(all_valid_hypotheses['F_HSV'][color].items(), key=operator.itemgetter(1))
	# print


	img2 = np.zeros(shape=(300,300,3),dtype=np.uint8)+255
	img = cv2.imread('/home/omari/Dropbox/Reports/AAAI16/Muhannad/pics/default.png')
	mask_final = np.zeros(shape=(300,300,3),dtype=float)
	if color != 'all':
		mask_final += 255
		# cv2.circle(mask_final,(150,150),100,(255,255,255),-1)
		rgb = list(all_valid_hypotheses['F_HSV'][color].keys()[0])
		print color,rgb

		for i1 in np.linspace(.1,.5,10):
			mask = np.ones(shape=(300,300),dtype=bool)

			rgb1 = [np.max([0,i-i1])*255.0 for i in rgb]
			rgb2 = [np.min([1,i+i1])*255.0 for i in rgb]

			C1 = [rgb1[2],rgb1[1],rgb1[0]]
			C2 = [rgb2[2],rgb2[1],rgb2[0]]
			# print rgb1
			# print rgb2
			#
			# for h in np.linspace(0,360,2000):
			# 	for v in np.linspace(0,100,1000):
			# 		x = v*np.cos(float(h)*np.pi/180)+150
			# 		y = v*np.sin(float(h)*np.pi/180)+150
			# 		A = list([i*255.0 for i in colorsys.hsv_to_rgb(float(h)/360.0, np.min([1,float(v)/50.0]), np.min([1,2-float(v)/50.0]) )])
			# 		C = [A[2],A[1],A[0]]
			# 		ok = 1
			# 		for a1,a2,a3 in zip(C,rgb1,rgb2):
			# 			if a1>=a2 and a1<=a3:
			# 				pass
			# 			else:
			# 				ok=0
			# 		# print C
			# 		# print ok
			# 		# print '----'
			# 		if ok:
			# 			C_final = [C[2],C[1],C[0]]
			# 		else:
			# 			C_final = [255,255,255]
			# 		img[x,y,:] = C_final

			mask &= (img[:,:,0]>=C1[0]) & (img[:,:,0]<=C2[0])
			mask &= (img[:,:,1]>=C1[1]) & (img[:,:,1]<=C2[1])
			mask &= (img[:,:,2]>=C1[2]) & (img[:,:,2]<=C2[2])
			mask_final[mask] -= [25.5,25.5,25.5]

	# for count,i in enumerate(np.linspace(25.5,255,10)):
		# print img[mask_final<i]*(1+count/10)
		# img2[mask_final<i] = img[mask_final<i]
	img2 = mask_final
	cv2.circle(img2,(150,150),101,(0,0,0),1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img2,color,(10,25), font, 1.2,(0,0,0),2)

	cv2.imwrite('/home/omari/Dropbox/Reports/AAAI16/Muhannad/pics/'+color+'.png',img2)

img = np.zeros(shape=(300*2,300*5,3),dtype=np.uint8)+255
for count,color in enumerate(['all','red','green','blue','cyan','sky','black','yellow','magenta','white']):
	img2 = cv2.imread('/home/omari/Dropbox/Reports/AAAI16/Muhannad/pics/'+color+'.png')
	if count<5:
		x = 0
		y = count
	else:
		x = 1
		y = count-5

	print x,y
	print 300*x,300*(x+1)
	print 300*y,300*(y+1)
	img[300*x:300*(x+1),300*y:300*(y+1),:] = img2
	print count

cv2.imwrite('/home/omari/Dropbox/Reports/AAAI16/Muhannad/pics/all_colors.png',img)
