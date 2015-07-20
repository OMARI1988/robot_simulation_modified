from visual import *
from shapefile import *
from Polygon import *
import numpy as np
import wx
from xml_functions import *
import cv2
import pyscreenshot as ImageGrab
# http://www.anninaruest.com/pie/2014/07/inverse-kinematics-and-the-m100rak/
from random import randint

class Robot():
    #-----------------------------------------------------------------------------------------------------#     initial
    def __init__(self):
        self._initilize_values()
        self.all_sentences_count = 1
        self.draw_scene()
        self.draw_robot()
        self.Data = read_data()


    #-----------------------------------------------------------------------------------------------------#     initial
    def _initilize_values(self):
        self.chess_shift_x = 8
        self.chess_shift_y = 6
        self.len_arm1 = 8
        self.len_arm2 = 6
        self.len_gripper = 2
        self.len_base = 2
        self.l1 = 0
        self.l2 = self.len_arm1
        self.l3 = self.len_arm2 + self.len_gripper
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.step = 3
        self.frame_number = 0
        self.object = {}
        self.object_shape = {}
        self.words = []
        self.positions = {}
        # manage diroctories to store data and images
        self.image_dir = '/home/omari/Datasets/robot_modified/scenes/'
        if not os.path.isdir(self.image_dir):
	        print 'please change the diroctory in extract_data.py'

    #--------------------------------------------------------------------------------------------------------#
    def _fix_sentences(self):
        S = self.Data['commands'][self.scene]
        for i in S:
            S[i] = S[i].replace("    ", " ")
            S[i] = S[i].replace("   ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace(".", "")
            S[i] = S[i].replace(",", "")
            S[i] = S[i].replace("'", "")
            S[i] = S[i].replace("-", " ")
            S[i] = S[i].replace("/", " ")
            S[i] = S[i].replace("!", "")
            S[i] = S[i].replace("(", "")
            S[i] = S[i].replace(")", "")
            S[i] = S[i].replace("?", "")
            A = S[i].split(' ')
            while '' in A:         A.remove('')
            S[i] = ' '.join(A)

        self.Data['commands'][self.scene] = S

    #-----------------------------------------------------------------------------------------------------#     change data
    def _change_data(self):

        def _change(words,key):
            for i,word in enumerate(words):
                indices = [j for j, x in enumerate(s) if x == word]
                for m in indices:
                    s[m] = key[i]
            self.Data['commands'][self.scene][sentence] = ' '.join(s)

        change_prism = ['pyramid','prism','tetrahedron']
        change_prism_to = ['ball','sphere','orb']
        change_prisms = ['pyramids','prisms','tetrahedrons']
        change_prisms_to = ['balls','spheres','orbs']
        change_box = ['block','cube','box','slab','parallelipiped']
        change_box_to = ['cylinder','can','drum','drum','can']
        change_boxes = ['cubes','boxes','blocks','slabs']
        change_boxes_to = ['cylinders','cans','drums','drums']

        a1 = randint(0,1)
        a2 = randint(0,1)
        a3 = randint(0,1)
        c='nothing'
        d='nothing'
        e='nothing'
        print a1,a2,a3
        if a1:            c = 'black'
        if a2:            d = 'sphere'    # orb, ball
        if a3:            e = 'cylinder' # can,


        #if self.scene % 2 == 1:
        # change commands
        for sentence in self.Data['commands'][self.scene]:
            s = self.Data['commands'][self.scene][sentence].split(' ')
            if d == 'sphere':
                _change(change_prism,change_prism_to)
                _change(change_prisms,change_prisms_to)
            if e == 'cylinder':
                _change(change_box,change_box_to)
                _change(change_boxes,change_boxes_to)
            if c != 'nothing':
                _change(['red','maroon'],['black','black'])

        # change scenes
        I = self.Data['scenes'][self.scene]['initial']
        F = self.Data['scenes'][self.scene]['final']
        self.Data['scenes'][self.scene]['initial'] = 1000+I
        self.Data['scenes'][self.scene]['final'] = 1000+F

        # change layouts
        self.Data['layouts'][1000+I] = {}
        self.Data['layouts'][1000+F] = {}
        for obj in self.Data['layouts'][I]:
            self.Data['layouts'][1000+I][obj] = dict(self.Data['layouts'][I][obj])
        for obj in self.Data['layouts'][F]:
            self.Data['layouts'][1000+F][obj] = dict(self.Data['layouts'][F][obj])

        for obj in self.Data['layouts'][1000+I]:
            if e == 'cylinder':
                if self.Data['layouts'][1000+I][obj]['shape']=='cube':
                    self.Data['layouts'][1000+I][obj]['shape'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000+I][obj]['shape']=='prism':
                    self.Data['layouts'][1000+I][obj]['shape'] = 'sphere'

        for obj in self.Data['layouts'][1000+F]:
            if e == 'cylinder':
                if self.Data['layouts'][1000+F][obj]['shape']=='cube':
                    self.Data['layouts'][1000+F][obj]['shape'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000+F][obj]['shape']=='prism':
                    self.Data['layouts'][1000+F][obj]['shape'] = 'sphere'


        if c != 'nothing':
            for obj in self.Data['layouts'][1000+I]:
                if self.Data['layouts'][1000+I][obj]['color']=='red':
                    self.Data['layouts'][1000+I][obj]['color'] = c

            for obj in self.Data['layouts'][1000+F]:
                if self.Data['layouts'][1000+F][obj]['color']=='red':
                    self.Data['layouts'][1000+F][obj]['color'] = c

        # change gripper
        self.Data['gripper'][1000+I] = self.Data['gripper'][I]
        self.Data['gripper'][1000+F] = self.Data['gripper'][F]

    #-----------------------------------------------------------------------------------------------------#     print scentences
    def _print_scentenses(self):
        scene = self.scene
        self.sentences = {}
        for count,i in enumerate(self.Data['commands'][scene]):
            if i not in self.Data['comments']:
                print count,'-',self.Data['commands'][scene][i]
                self.all_sentences_count += 1
                self.sentences[count] = ['GOOD',self.Data['commands'][scene][i]]
                #for word in self.Data['commands'][scene][i].split(' '):
                #    if word not in self.words:
                #        self.words.append(word)
            else:
                print count,'-','SPAM'
                self.sentences[count] = ['SPAM',self.Data['commands'][scene][i]]
        #print self.words
        #print len(self.words)
        print '--------------------------'

    #-----------------------------------------------------------------------------------------------------#     initilize scene
    def _initialize_scene(self):
        self._add_objects_to_scene()
        self._initialize_robot()
        self._update_scene_number()

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _add_objects_to_scene(self):
        self.frame_number = 0
        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['initial']]   # initial layput
        for obj in l1:
            x = l1[obj]['position'][0]
            y = l1[obj]['position'][1]
            z = l1[obj]['position'][2]
            # finding the color
            c = self._find_color(l1[obj]['color'])
            # finding the shape1    TO SIMULATE SHAPE FEATURE VECTOR
            s = self._find_shape(l1[obj]['shape'])
            # finding the shape
            #if self.scene%2 == 0:
            if l1[obj]['shape'] == 'cube': self._cube(x,y,z*.9,c)
            if l1[obj]['shape'] == 'prism': self._prism(x,y,z*.9,c)
            #if self.scene%2 == 1:
            if l1[obj]['shape'] == 'cylinder': self._cylinder(x,y,z*.9,c)
            if l1[obj]['shape'] == 'sphere': self._sphere(x,y,z*.9,c)

            # inilizing the position vector to be saved later
            self.positions[obj] = {}
            self.positions[obj]['x'] = [float(x)]
            self.positions[obj]['y'] = [float(y)]
            self.positions[obj]['z'] = [float(z)]
            self.positions[obj]['color'] = c
            self.positions[obj]['shape'] = s

    #-----------------------------------------------------------------------------------------------------#     find color
    def _find_color(self,a):
        c = (0,0,0)
        if a == 'red': c = color.red                #(1,0,0)
        elif a == 'blue': c = color.blue            #(0,0,1)
        elif a == 'green': c = color.green          #(0,1,0)
        elif a == 'gray': c = (.5,.5,.5)            #(.6,.6,.6)
        elif a == 'cyan': c = color.cyan
        elif a == 'yellow': c = color.yellow
        elif a == 'white': c = color.white
        elif a == 'magenta': c = color.magenta
        elif a == 'black': c = (0,0,0)
        elif a == 'brown': c = (0.55, 0.27, 0.07)
        else:
            print '********* error no color match',a
        return c

    #-----------------------------------------------------------------------------------------------------#     find shape
    def _find_shape(self,a):
            if a == 'cube': c = 0.0
            if a == 'sphere': c = 0.33
            if a == 'cylinder': c = 0.66
            if a == 'prism': c = 1.0
            return c

    #-----------------------------------------------------------------------------------------------------#     initilize robot in the scene
    def _initialize_robot(self):
        initial_position = self.Data['gripper'][self.Data['scenes'][self.scene]['initial']]
        a1,a2,a3 = self._inverse_kinematics(initial_position[0],initial_position[1],initial_position[2])
        self.rotate_robot_init(-a1,-a2,-a3)
        self.positions['gripper'] = {}
        self.positions['gripper']['x'] = [float(initial_position[0])]
        self.positions['gripper']['y'] = [float(initial_position[1])]
        self.positions['gripper']['z'] = [float(initial_position[2])]

    #-----------------------------------------------------------------------------------------------------#     update scene number
    def _update_scene_number(self):
        self.label.text = 'Scene number : '+str(self.scene)

    #-----------------------------------------------------------------------------------------------------#     move robot
    def _move_robot(self,save):
        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['initial']]   # initial layput
        l2 = self.Data['layouts'][self.Data['scenes'][self.scene]['final']]     # final layput
        I = self.Data['scenes'][self.scene]['I_move']
        F = self.Data['scenes'][self.scene]['F_move']
        if I != []:
            # move the robot alone
            initial_position = self.Data['gripper'][self.Data['scenes'][self.scene]['initial']]
            final_position = l1[I]['position']
            if final_position != initial_position:
                a1,a2,a3 = self._inverse_kinematics(final_position[0],final_position[1],final_position[2])
                self.rotate_robot(-a1,-a2,-a3,save)
            if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'cube':
                z_offset = +.22
            if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'sphere':
                z_offset = +.22
            if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'cylinder':
                z_offset = -.18
            if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'prism':
                z_offset = -.18
            # move the robot and the object
            initial_position = final_position
            final_position = l2[F]['position']
            self.rotate_robot_with_object(initial_position, final_position, z_offset, save)
            # move the robot and the object down
            final_position = self.Data['gripper'][self.Data['scenes'][self.scene]['final']]
            a1,a2,a3 = self._inverse_kinematics(final_position[0],final_position[1],final_position[2])
            self.rotate_robot(-a1,-a2,-a3,save)

    #-----------------------------------------------------------------------------------------------------#     save motion
    def _save_motion(self):
        target = open('/home/omari/Datasets/robot_modified/motion/scene'+str(self.scene)+'.txt', 'w')
        for i in self.sentences:
            target.write('sentence:'+str(i)+'\n')
            target.write(self.sentences[i][0]+':'+self.sentences[i][1]+'\n')

        for key in self.positions:
            target.write('object:'+str(key)+'\n')
            target.write('x:')
            for i in self.positions[key]['x']:
                target.write("{:3.2f}".format(i)+',')
            target.write("\n")
            target.write('y:')
            for i in self.positions[key]['y']:
                target.write("{:3.2f}".format(i)+',')
            target.write("\n")
            target.write('z:')
            for i in self.positions[key]['z']:
                target.write("{:3.2f}".format(i)+',')
            target.write("\n")
            if key != 'gripper':
                c = self.positions[key]['color']
                s = self.positions[key]['shape']
                target.write('color:'+str(c[0])+','+str(c[1])+','+str(c[2]))
                target.write("\n")
                target.write('shape:'+str(s))
                target.write("\n")


        target.close()

    #-----------------------------------------------------------------------------------------------------#     clear scene
    def _clear_scene(self):
        keys = self.object.keys()
        for i in keys:
            self.object[i].visible = False
            self.object.pop(i)
            self.object_shape.pop(i)

    #-----------------------------------------------------------------------------------------------------#     object functions
    def _cube(self,x,y,z,c):
		self.object[(x,y,int(z/.9))] = box(pos=(4.5+x,2.5+y,.4+z),axis=(0,0,1),size=(.8,.8,.8),
		    color=c,material=materials.plastic)
		self.object_shape[(x,y,int(z/.9))] = 'cube'

    def _cylinder(self,x,y,z,c):
        self.object[(x,y,int(z/.9))] = cylinder(pos=(4.5+x,2.5+y,z),axis=(0,0,.8),radius=.4,
            color=c,material=materials.plastic)
        self.object_shape[(x,y,int(z/.9))] = 'cylinder'

    def _sphere(self,x,y,z,c):
        self.object[(x,y,int(z/.9))] = sphere(pos=(4.5+x,2.5+y,.4+z),radius=.4,
            color=c,material=materials.plastic)
        self.object_shape[(x,y,int(z/.9))] = 'sphere'

    def _prism(self,x,y,z,c):
        self.object[(x,y,int(z/.9))] = pyramid(pos=(4.5+x,2.5+y,z),axis=(0,0,1),size=(.8,.8,.8),
            color=c,material=materials.plastic)
        self.object_shape[(x,y,int(z/.9))] = 'prism'

    #-----------------------------------------------------------------------------------------------------#     rotate robot
    def rotate_robot(self,a0,a1,a2,save):
        p0 = np.linspace(self.a0,a0,self.step) # path 0
        p1 = np.linspace(self.a1,a1,self.step)
        p2 = np.linspace(self.a2,a2,self.step)
        for i in range(self.step):
            rate(10000)
            self.rotate_joint(self.base_faces,self.base_faces_origin,(0,self.chess_shift_y,0),p0[i],0,0)
            self.rotate_joint(self.arm1_faces,self.arm1_faces_origin,(0,self.chess_shift_y,self.len_base),
			    p0[i],p1[i],0)
            self.rotate_joint(self.arm2_faces,self.arm2_faces_origin,
			    (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
            self.rotate_joint(self.gripper1_faces,self.gripper1_faces_origin,
			    (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
            self.rotate_joint(self.gripper2_faces,self.gripper2_faces_origin,
			    (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
		    # saving the positions of objects and arm
            ang = [p0[i],p1[i],p2[i]]
            self._append_position(ang,0,'')
            if save == 'save':  self.saveSnapshot()
            else:               self.frame_number += 1
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    #-----------------------------------------------------------------------------------------------------#     rotate robot with objects
    def rotate_robot_with_object(self,initial_position,final_position,z_offset,save):
        OI = self.Data['scenes'][self.scene]['I_move']
        x = initial_position[0]
        y = initial_position[1]
        z = initial_position[2]
        x1 = final_position[0]
        y1 = final_position[1]
        z1 = final_position[2]
        a1,a2,a3 = self._inverse_kinematics(final_position[0],final_position[1],final_position[2])

        p0 = np.linspace(self.a0,-a1,self.step) # path 0
        p1 = np.linspace(self.a1,-a2,self.step)
        p2 = np.linspace(self.a2,-a3,self.step)

        #print pos2
        for i in range(self.step):
            rate(10000)
            self.rotate_joint(self.base_faces,self.base_faces_origin,(0,self.chess_shift_y,0),p0[i],0,0)
            self.rotate_joint(self.arm1_faces,self.arm1_faces_origin,(0,self.chess_shift_y,self.len_base),
                p0[i],p1[i],0)
            self.rotate_joint(self.arm2_faces,self.arm2_faces_origin,
                (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
            self.rotate_joint(self.gripper1_faces,self.gripper1_faces_origin,
                (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
            self.rotate_joint(self.gripper2_faces,self.gripper2_faces_origin,
                (self.len_arm1,self.chess_shift_y,self.len_base),p0[i],p1[i],p2[i])
            pos = self.forward_arms(p0[i],p1[i],p2[i])
            self.object[(x,y,z)].pos = (pos[0],pos[1],pos[2]+z_offset)
		    # saving the positions of objects and arm
            ang = [p0[i],p1[i],p2[i]]
            position = [pos[0]-4.5,pos[1]-2.5,pos[2]+z_offset]
            self._append_position(ang,position,OI)
            if save == 'save':  self.saveSnapshot()
            else:               self.frame_number += 1
        self.a0 = -a1
        self.a1 = -a2
        self.a2 = -a3

    #-----------------------------------------------------------------------------------------------------#     append positions
    def _append_position(self,ang,o,obj):
		    # saving the positions of objects and arm
            pos = self.forward_arms(ang[0],ang[1],ang[2])
            keys = self.positions.keys()
            for key in keys:
                if key == 'gripper':
                    self.positions[key]['x'].append(pos[0]-4.5)
                    self.positions[key]['y'].append(pos[1]-2.5)
                    self.positions[key]['z'].append(pos[2])
                elif key == obj:
                    self.positions[key]['x'].append(o[0])
                    self.positions[key]['y'].append(o[1])
                    self.positions[key]['z'].append(o[2])
                else:
                    self.positions[key]['x'].append(self.positions[key]['x'][self.frame_number])
                    self.positions[key]['y'].append(self.positions[key]['y'][self.frame_number])
                    self.positions[key]['z'].append(self.positions[key]['z'][self.frame_number])

    #-----------------------------------------------------------------------------------------------------#     rotate robot intial
    def rotate_robot_init(self,a0,a1,a2):
		self.rotate_joint(self.base_faces,self.base_faces_origin,(0,self.chess_shift_y,0),a0,0,0)
		self.rotate_joint(self.arm1_faces,self.arm1_faces_origin,(0,self.chess_shift_y,self.len_base),a0,a1,0)
		self.rotate_joint(self.arm2_faces,self.arm2_faces_origin,
		    (self.len_arm1,self.chess_shift_y,self.len_base),a0,a1,a2)
		self.rotate_joint(self.gripper1_faces,self.gripper1_faces_origin,
		    (self.len_arm1,self.chess_shift_y,self.len_base),a0,a1,a2)
		self.rotate_joint(self.gripper2_faces,self.gripper2_faces_origin,
		    (self.len_arm1,self.chess_shift_y,self.len_base),a0,a1,a2)
		self.a0 = a0
		self.a1 = a1
		self.a2 = a2

    #-----------------------------------------------------------------------------------------------------#     rotate joint
    def rotate_joint(self,obj,faces,shift,a0,a1,a2):
		for j,v in enumerate(faces):
			v1 = v - shift
			v2 = rotate(v1, angle=a2, axis=(0,1,0)) + (shift[0],0,0)
			v2 = rotate(v2, angle=a1, axis=(0,1,0))
			v2 = rotate(v2, angle=a0, axis=(0,0,1)) + (0,shift[1],shift[2])
			obj.pos[j] = v2

	#----------------------------------------------------------------------------------#
	# input: coordinates x,y,z of the target point, lengths l1,l2,l3 of the arms, were
	# l1 is the base height
	# l2 is the length of the first arm
	# l3 is the length of the second arm
	#
	#	     /\
	#	l2  /  \  l3
	#	   /	\
	#	  #
	#      l1 #
	#       #####
	#
	# output: angles a1,a2,a3 of the joints, in radians
    def _inverse_kinematics(self,x,y,z):
		z -= 1.8
		z *= .9
		x += 4.5
		y = 3.5 - y
		# used in case arm can't reach that location
		s  = "(%g,%g,%g) is out of range." % tuple(np.around([x,y,z],2))
		# compute the first angle
		a1 = np.arctan2(y,x)
		# compute the thirs angle
		r  = np.hypot(x,y)
		z -= self.l1
		u3 = ( r**2 + z**2 - self.l2**2 - self.l3**2 ) / ( 2*self.l2*self.l3 )
		if abs(u3)>1:    raise Exception(s)
		a3 = -np.arccos(u3)
		# compute the second angle
		v  = self.l2 + self.l3*u3
		w  = -self.l3 * np.sqrt(1-u3**2)  # this is sin(a3)>0 assuming 0<a3<pi
		a2 = np.arctan2(v*z-w*r,v*r+w*z)
		if a2<0 or a2>np.pi:    raise Exception(s)
		return a1,a2,a3

    #-----------------------------------------------------------------------------------------------------#     initial draw scene
    def draw_scene(self):
        self.display = display(title='simultaneous learning and grounding',
            x=0, y=0, width=600, height=600,
            center=(self.chess_shift_x,self.chess_shift_y,0),
            forward=(self.chess_shift_x-10,self.chess_shift_y,-7),
            background=(1,1,1))
        self.label = label(pos=(10,10,10), text='Scene number : ',height=20,color=(0,0,0))
        checkerboard = ( (0,1,0,1,0,1,0,1),
				 (1,0,1,0,1,0,1,0),
				 (0,1,0,1,0,1,0,1),
				 (1,0,1,0,1,0,1,0),
		 		 (0,1,0,1,0,1,0,1),
				 (1,0,1,0,1,0,1,0),
				 (0,1,0,1,0,1,0,1),
				 (1,0,1,0,1,0,1,0) )
        tex = materials.texture(data=checkerboard,
            mapping="sign",
            interpolate=False)
        chess1 = box(pos=(self.chess_shift_x,self.chess_shift_y,-.3),axis=(0,0,1),size=(.4,9,9),
            color=color.orange,material=materials.wood)
        chess2 = box(pos=(self.chess_shift_x,self.chess_shift_y,-.25),axis=(0,0,1), size=(.5,8,8),
            color=color.orange, material=tex)
        x = arrow(pos=(0,0,0),axis=(1,0,0),length=2,shaftwidth=.2,color=color.red)
        y = arrow(pos=(0,0,0),axis=(0,1,0),length=2,shaftwidth=.2,color=color.green)
        z = arrow(pos=(0,0,0),axis=(0,0,1),length=2,shaftwidth=.2,color=color.blue)

    #-----------------------------------------------------------------------------------------------------#     initial draw robot
    def draw_robot(self):
		base_1 = box(pos=(0,self.chess_shift_y,-.25),axis=(0,0,1), size=(.5,2,2),color=color.black,
		    material=materials.plastic)
		base_2 = Polygon( [(-1,0), (-.75,self.len_base), (.75,self.len_base), (1,0)] )
		base_3 = shapes.circle(pos=(0,self.len_base), radius=.75)
		base_4 = shapes.circle(pos=(0,self.len_base), radius=0.2)
		base_s = [(0,self.chess_shift_y-.5,0),(0,self.chess_shift_y+.5,0)]
		self.base = extrusion(pos=base_s, shape=base_2+base_3-base_4, color=color.red)

		arm1_1 = Polygon( [(0,.75), (self.len_arm1,.5), (self.len_arm1,-.5), (0,-.75)] )
		arm1_2 = shapes.circle(pos=(0,0), radius=.75)
		arm1_3 = shapes.circle(pos=(0,0), radius=0.2)
		arm1_4 = shapes.circle(pos=(self.len_arm1,0), radius=.5)
		arm1_5 = shapes.circle(pos=(self.len_arm1,0), radius=0.2)
		arm1_s = [(0,self.chess_shift_y+.5,self.len_base),(0,self.chess_shift_y+1.5,self.len_base)]
		self.arm1 = extrusion(pos=arm1_s, shape=arm1_1+arm1_2-arm1_3+arm1_4-arm1_5, color=color.blue)

		arm2_1 = Polygon( [(0,.5), (self.len_arm2,.4), (self.len_arm2,-.4), (0,-.5)] )
		arm2_2 = shapes.circle(pos=(0,0), radius=.5)
		arm2_3 = shapes.circle(pos=(0,0), radius=0.2)
		arm2_4 = shapes.circle(pos=(self.len_arm2,0), radius=.4)
		arm2_5 = shapes.circle(pos=(self.len_arm2,0), radius=0.2)
		arm2_s = [(self.len_arm1,self.chess_shift_y-.5,self.len_base),(self.len_arm1,self.chess_shift_y+.5,self.len_base)]
		self.arm2 = extrusion(pos=arm2_s, shape=arm2_1+arm2_2-arm2_3+arm2_4-arm2_5, color=color.red)

		gripper_1 = Polygon( [(0,.4), (self.len_gripper,.3), (self.len_gripper,-.3), (0,-.4)] )
		gripper_2 = shapes.circle(pos=(0,0), radius=.4)
		gripper_4 = shapes.circle(pos=(self.len_gripper,0), radius=.3)
		gripper1_s = [(self.len_arm1+self.len_arm2,self.chess_shift_y-.6,self.len_base),(self.len_arm1+self.len_arm2,self.chess_shift_y-.5,self.len_base)]
		gripper2_s = [(self.len_arm1+self.len_arm2,self.chess_shift_y+.5,self.len_base),(self.len_arm1+self.len_arm2,self.chess_shift_y+.6,self.len_base)]

		self.gripper1 = extrusion(pos=gripper1_s, shape=gripper_1+gripper_2+gripper_4, color=color.green)
		self.gripper2 = extrusion(pos=gripper2_s, shape=gripper_1+gripper_2+gripper_4, color=color.green)

		self.base_faces = self.base.create_faces()
		self.arm1_faces = self.arm1.create_faces()
		self.arm2_faces = self.arm2.create_faces()
		self.gripper1_faces = self.gripper1.create_faces()
		self.gripper2_faces = self.gripper2.create_faces()

		self.base_faces_origin = self.base_faces.pos.copy()
		self.arm1_faces_origin = self.arm1_faces.pos.copy()
		self.arm2_faces_origin = self.arm2_faces.pos.copy()
		self.gripper1_faces_origin = self.gripper1_faces.pos.copy()
		self.gripper2_faces_origin = self.gripper2_faces.pos.copy()

    def _saveSnapshot2(self):
        scene = str(self.scene)
        im=ImageGrab.grab(bbox=(10,10,510,510)) # X1,Y1,X2,Y2
        if not os.path.isdir(self.image_dir+scene+'/'):
            os.makedirs(self.image_dir+scene)
        if os.path.isfile(self.image_dir+scene+'/data.txt'):
            os.remove(self.image_dir+scene+'/data.txt')
        #print dir(self.display)

    def saveSnapshot(self):
        scene = str(self.scene)
        if not os.path.isdir(self.image_dir+scene+'/'):
            os.makedirs(self.image_dir+scene)
        if os.path.isfile(self.image_dir+scene+'/data.txt'):
            os.remove(self.image_dir+scene+'/data.txt')
        # based largely on code posted to wxpython-users by Andrea Gavana 2006-11-08
        dcSource = wx.ScreenDC()
        size = dcSource.Size

        # Create a Bitmap that will later on hold the screenshot image
	    # Note that the Bitmap must have a size big enough to hold the screenshot
	    # -1 means using the current default colour depth
        bmp = wx.EmptyBitmap(self.display.width,self.display.height-60)

	    # Create a memory DC that will be used for actually taking the screenshot
        memDC = wx.MemoryDC()

	    # Tell the memory DC to use our Bitmap
	    # all drawing action on the memory DC will go to the Bitmap now
        memDC.SelectObject(bmp)

	    # Blit (in this case copy) the actual screen on the memory DC
	    # and thus the Bitmap
        memDC.Blit( 0, # Copy to this X coordinate
        0, # Copy to this Y coordinate
        self.display.width, # Copy this width
        self.display.height-60, # Copy this height
        dcSource, # From where do we copy?
        self.display.x, # What's the X offset in the original DC?
        self.display.y+60  # What's the Y offset in the original DC?
        )

	    # Select the Bitmap out of the memory DC by selecting a new
	    # uninitialized Bitmap
        memDC.SelectObject(wx.NullBitmap)

        img = bmp.ConvertToImage()
        if self.frame_number<10:        j = '000'+str(self.frame_number)
        elif self.frame_number<100:     j = '00'+str(self.frame_number)
        elif self.frame_number<1000:    j = '0'+str(self.frame_number)
        if self.scene<10:      k = '000'+str(self.scene)
        elif self.scene<100:    k = '00'+str(self.scene)
        elif self.scene<1000:   k = '0'+str(self.scene)
        img.SaveFile(self.image_dir+str(self.scene)+'/scene_'+k+'_frame_'+j+'.png', wx.BITMAP_TYPE_PNG)
        #img.SaveFile(self.image_dir+self.scene+'_'+j+'.png', wx.BITMAP_TYPE_PNG)
        #img1 = cv2.imread(self.image_dir+self.scene+'_'+j+'.png')
        #cv2.imshow('test',img1)


        print self.frame_number,'image saved..'
        self.frame_number+=1

    def forward_arms(self,a1,a2,a3):
		z = self.l2*np.sin(a2)+self.l3*np.sin(a3+a2)
		r = self.l2*np.cos(a2)+self.l3*np.cos(a3+a2)
		x = r*np.cos(a1)
		y = r*np.sin(a1)
		return (x,y+self.chess_shift_y,1.8-z)
