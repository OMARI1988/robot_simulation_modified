from visual import *
# from shapefile import *
from Polygon import *
import numpy as np
import wx
from xml_functions import *
import cv2
import pyscreenshot as ImageGrab
# http://www.anninaruest.com/pie/2014/07/inverse-kinematics-and-the-m100rak/
from random import randint
import operator
import pickle
import os

import itertools
import copy
import matplotlib.pyplot as plt
from GMM_BIC import *
from GMM_functions import *

class Robot():
    #-----------------------------------------------------------------------------------------------------#     initial
    def __init__(self):
        self._initilize_values()
        self.all_sentences_count = 1
        self.draw_scene()
        # self.draw_robot()
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
        self.positions_f = {}
        # manage diroctories to store data and images
        self.image_dir = '/home/omari/Datasets/robot_modified/scenes/'
        self.image_dir2 = '/home/omari/Dropbox/robot_modified/IT/scenes/'
        if not os.path.isdir(self.image_dir):
	        print 'please change the diroctory in extract_data.py'
        # self.words_order = pickle.load( open( "/home/omari/Dropbox/robot_modified/IT/pickle/words_order.p", "rb" ) )

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
    def _change_data(self,a1,a2,a3):

        def _change(words,key):
            for i,word in enumerate(words):
                indices = [j for j, x in enumerate(s) if x == word]
                for m in indices:
                    s[m] = key[i]
            self.Data['commands'][self.scene][sentence] = ' '.join(s)

        change_prism = ['pyramid','prism','tetrahedron','triangle']
        change_prism_to = ['ball','sphere','orb','orb']
        change_prisms = ['pyramids','prisms','tetrahedrons','triangles']
        change_prisms_to = ['balls','spheres','orbs','orbs']
        change_box = ['block','cube','box','slab','parallelipiped','parallelepiped','brick','square']
        change_box_to = ['cylinder','can','drum','drum','can','can','can','can']
        change_boxes = ['cubes','boxes','blocks','slabs','parallelipipeds','bricks','squares']
        change_boxes_to = ['cylinders','cans','drums','drums','cans','cans','cans']

        # a1 = randint(0,1)
        # a2 = randint(0,1)
        # a3 = randint(0,1)
        c='nothing'
        d='nothing'
        e='nothing'
        # print a1,a2,a3
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
                if self.Data['layouts'][1000+I][obj]['F_SHAPE']=='cube':
                    self.Data['layouts'][1000+I][obj]['F_SHAPE'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000+I][obj]['F_SHAPE']=='prism':
                    self.Data['layouts'][1000+I][obj]['F_SHAPE'] = 'sphere'

        for obj in self.Data['layouts'][1000+F]:
            if e == 'cylinder':
                if self.Data['layouts'][1000+F][obj]['F_SHAPE']=='cube':
                    self.Data['layouts'][1000+F][obj]['F_SHAPE'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000+F][obj]['F_SHAPE']=='prism':
                    self.Data['layouts'][1000+F][obj]['F_SHAPE'] = 'sphere'


        if c != 'nothing':
            for obj in self.Data['layouts'][1000+I]:
                if self.Data['layouts'][1000+I][obj]['F_HSV']=='red':
                    self.Data['layouts'][1000+I][obj]['F_HSV'] = c

            for obj in self.Data['layouts'][1000+F]:
                if self.Data['layouts'][1000+F][obj]['F_HSV']=='red':
                    self.Data['layouts'][1000+F][obj]['F_HSV'] = c

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
        # self._initialize_robot()
        self._update_scene_number()


    #-----------------------------------------------------------------------------------------------------#     initilize scene
    def _draw_directions(self):

                # inilizing the position vector to be saved later
                self.m_obj = []
                self.objs = []
                for obj in self.positions:
                    x1 = self.positions[obj]['x']
                    y1 = self.positions[obj]['y']
                    z1 = self.positions[obj]['z']
                    counter = 0
                    for obj2 in self.positions_f:
                        x2 = self.positions_f[obj2]['x']
                        y2 = self.positions_f[obj2]['y']
                        z2 = self.positions_f[obj2]['z']

                        if (x2 == x1) and (y2 == y1) and (z2 == z1):
                            counter += 1
                    if counter == 0:
                        self.m_obj = obj
                    else:
                        self.objs.append(obj)


                # x = self.positions[self.m_obj]['x'][0]
                # y = self.positions[self.m_obj]['y'][0]
                # z = self.positions[self.m_obj]['z'][0]
                # c = self.positions[self.m_obj]['F_HSV']


                xi = self.positions_f[self.m_obj]['x'][0]
                yi = self.positions_f[self.m_obj]['y'][0]
                zi = self.positions_f[self.m_obj]['z'][0]

                xf = self.positions[self.m_obj]['x'][0]
                yf = self.positions[self.m_obj]['y'][0]
                zf = self.positions[self.m_obj]['z'][0]

                # finding the shape
                # if self.positions[self.m_obj]['F_SHAPE'] == 0.0: self._cube(x,y,z*.9,c)
                # if self.positions[self.m_obj]['F_SHAPE'] == 0.33: self._prism(x,y,z*.9,c)
                # if self.positions[self.m_obj]['F_SHAPE'] == 0.66: self._cylinder(x,y,z*.9,c)
                # if self.positions[self.m_obj]['F_SHAPE'] == 1.0: self._sphere(x,y,z*.9,c)
                self.distances = []
                for i in range(50):

                    x = xi + (xf-xi)*i/10.0
                    y = yi + (yf-yi)*i/10.0
                    z = zi + (zf-zi)*i/10.0
                    for obj in self.objs:
                        x2 = self.positions[obj]['x'][0]
                        y2 = self.positions[obj]['y'][0]
                        z2 = self.positions[obj]['z'][0]
                        dist = np.sqrt( (x2-x)**2 + (y2-y)**2 + (z2-z)**2  )
                        self.distances.append(dist)

                for i in range(10):

                    x = xi + (xf-xi)*i/10.0
                    y = yi + (yf-yi)*i/10.0
                    z = zi + (zf-zi)*i/10.0
                    for obj in self.objs:
                        x2 = self.positions[obj]['x'][0]
                        y2 = self.positions[obj]['y'][0]
                        z2 = self.positions[obj]['z'][0]
                        dist = np.sqrt( (x2-x)**2 + (y2-y)**2 + (z2-z)**2  )
                        if obj in [0,1,2,6,7]:
                            c = color.red
                        elif obj in [3]:
                            c = color.blue
                        elif obj in [4,5]:
                            c = color.cyan
                        else:
                            c = color.green
                        A = arrow(pos=(x+self.chess_shift_x-3.5,y+self.chess_shift_y-3.5,z+.25),axis=(x2-x,y2-y,z2-z),length=dist,shaftwidth=.05,color=c)

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _plot_distances(self):
        # print self.distances
        self.cv_types = ['full']
        n, bins, patches = plt.hist(self.distances, 50, normed=1, histtype='stepfilled')
        plt.show()

        gmm_c = self._bic_gmm(self.distances,1)
        # plt.plot(self.distances)

    #--------------------------------------------------------------------------------------------------------#
    # sub function for _convert_color_shape_location_to_gmm
    def _bic_gmm(self,points,plot):
        gmm1 = {}
        lowest_bic = np.infty
        bic = []
        X = []
        for point in points:
            if X == []:         X = [point]
            else:               X = np.vstack([X,point])
        k = np.minimum(len(X),7)
        best_gmm, bic = gmm_bic(X, k, self.cv_types)
        if plot:
            plot_data_1d(X, best_gmm, bic, k, self.cv_types,0, 1)
            plt.show()
            plt.savefig("/home/omari/Datasets/robot_modified/clusters/" + str(len(X[0])) + "-" + str(self.scene) +".png")

        for i, (mean, covar) in enumerate(zip(best_gmm.means_, best_gmm._get_covars())):
            gmm1[i] = {}
            gmm1[i]['mean'] = mean
            gmm1[i]['covar'] = covar
            gmm1[i]['N'] = 1.0
        return gmm1

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _add_objects_to_scene(self):
        self.frame_number = 0
        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['initial']]   # initial layput
        for obj in l1:
            x = l1[obj]['position'][0]
            y = l1[obj]['position'][1]
            z = l1[obj]['position'][2]
            # finding the color
            c = self._find_color(l1[obj]['F_HSV'])
            # finding the shape1    TO SIMULATE SHAPE FEATURE VECTOR
            s = self._find_shape(l1[obj]['F_SHAPE'])
            # finding the shape
            # if l1[obj]['F_SHAPE'] == 'cube': self._cube(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'prism': self._prism(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'cylinder': self._cylinder(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'sphere': self._sphere(x,y,z*.9,c)

            # inilizing the position vector to be saved later
            self.positions[obj] = {}
            self.positions[obj]['x'] = [float(x)]
            self.positions[obj]['y'] = [float(y)]
            self.positions[obj]['z'] = [float(z)]
            self.positions[obj]['F_HSV'] = c
            self.positions[obj]['F_SHAPE'] = s

        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['final']]   # initial layput
        for obj in l1:
            x = l1[obj]['position'][0]
            y = l1[obj]['position'][1]
            z = l1[obj]['position'][2]
            # finding the color
            c = self._find_color(l1[obj]['F_HSV'])
            # finding the shape1    TO SIMULATE SHAPE FEATURE VECTOR
            s = self._find_shape(l1[obj]['F_SHAPE'])
            # finding the shape
            # if l1[obj]['F_SHAPE'] == 'cube': self._cube(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'prism': self._prism(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'cylinder': self._cylinder(x,y,z*.9,c)
            # if l1[obj]['F_SHAPE'] == 'sphere': self._sphere(x,y,z*.9,c)

            # inilizing the position vector to be saved later
            self.positions_f[obj] = {}
            self.positions_f[obj]['x'] = [float(x)]
            self.positions_f[obj]['y'] = [float(y)]
            self.positions_f[obj]['z'] = [float(z)]
            self.positions_f[obj]['F_HSV'] = c
            self.positions_f[obj]['F_SHAPE'] = s

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _sum_of_all_hypotheses(self):
        self.all_valid_hypotheses['sum'] = {}
        for f in self.all_valid_hypotheses:
            if f != 'sum':
                sum1 = 0.0
                for k in self.all_valid_hypotheses[f]:
                    for j in self.all_valid_hypotheses[f][k]:
                        sum1 += self.all_valid_hypotheses[f][k][j]
                self.all_valid_hypotheses['sum'][f]=sum1

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _draw_the_arrow(self):
        I = self.Data['scenes'][self.scene]['I_move']
        F = self.Data['scenes'][self.scene]['F_move']
        x1 = self.positions[I]['x'][0]
        y1 = self.positions[I]['y'][0]
        z1 = self.positions[I]['z'][0]
        x2 = self.positions_f[F]['x'][0]
        y2 = self.positions_f[F]['y'][0]
        z2 = self.positions_f[F]['z'][0]
        dist = np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2  )
        A = arrow(pos=(x1+self.chess_shift_x-3.5,y1+self.chess_shift_y-3.5,z1+.25),axis=(x2-x1,y2-y1,z2-z1),length=dist,shaftwidth=.15,color=color.red)

    #-----------------------------------------------------------------------------------------------------#     add objects to scene
    def _get_unique_features(self):
        I = self.Data['scenes'][self.scene]['I_move']
        F = self.Data['scenes'][self.scene]['F_move']

        #initial scene
        # check to see if single feature is enough
        # self._single_feature(I,self.positions)
        # self._single_feature_f(F,self.positions_f)
        #
        # print self.u_pos
        # print self.u_hsv
        # print self.u_shp
        # print '------'
        # print self.u_pos_f
        # print self.u_hsv_f
        # print self.u_shp_f
        # print '------'


        self.feature_I = self._find_unique_feature(I,copy.deepcopy(self.positions))
        if self.motion == [0,1,0] or self.motion == [1,0]:
            self.feature_F = self._find_unique_feature_F(F,copy.deepcopy(self.positions_f))
            print self.feature_F
        print '-------------'
        # self.feature_I = [['color']]
        print self.feature_I

        # self.feature_I =

    #--------------------------------------------------------------------------------------------------------#
    def _find_unique_feature(self,I,positions):

        self.u_pos_f    = []
        self.u_hsv_name = []
        self.u_shp_name = []
        self.u_hsv_value = []
        self.u_shp_value = []
        self.u_hsvR_name = []
        self.u_shpR_name = []
        self.u_hsvR_value = []
        self.u_shpR_value = []

        x1 = positions[I]['x'][0]/7.0
        y1 = positions[I]['y'][0]/7.0
        z1 = positions[I]['z'][0]

        scene_layout = {}
        for i in positions:
            scene_layout[i] = {}
            scene_layout[i]['F_SHAPE'] = positions[i]['F_SHAPE']
            scene_layout[i]['F_HSV'] = positions[i]['F_HSV']
            scene_layout[i]['F_POS'] = [positions[i]['x'][0]/7.0,positions[i]['y'][0]/7.0,positions[i]['z'][0]]

        # Test which features can be used from the moving objects
        # 1) check pos first
        features = []
        relations = []
        use_pos = 0
        # print self.all_valid_hypotheses.keys()
        for p in self.all_valid_hypotheses['F_POS']:
            for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                m1 = np.asarray([x1,y1,z1])
                m2 = np.asarray(list(p1))
                print p,m1,m2
                if self._distance_test(m1,m2)<.10:
                    # print p

                    if 'F_POS' not in features:
                        features.append('F_POS')

        # I already now that color and shape are good to go
        # 2) Test color
        features.append('F_HSV')
        # 3) test Shape
        features.append('F_SHAPE')

        # self.u_hsv.append(np.asarray(list(positions[I]['F_HSV'])))
        for c in self.all_valid_hypotheses['F_HSV']:
            for c1 in self.all_valid_hypotheses['F_HSV'][c].keys():
                m1 = np.asarray(list(positions[I]['F_HSV']))
                m2 = np.asarray(list(c1))
                if self._distance_test(m1,m2)<.1:
                    self.u_hsv_name.append(c)
                    # print self.all_valid_hypotheses['F_HSV'].keys()
                    self.u_hsv_value.append(self.all_valid_hypotheses['F_HSV'][c][c1]/self.all_valid_hypotheses['sum']['F_HSV'])

        # self.u_shp.append(np.asarray(list([positions[I]['F_SHAPE']])))
        for s in self.all_valid_hypotheses['F_SHAPE']:
            for s1 in self.all_valid_hypotheses['F_SHAPE'][s].keys():
                m1 = np.asarray(list([positions[I]['F_SHAPE']]))
                m2 = np.asarray(list([s1]))
                if self._distance_test(m1,m2)<.1:
                    print '>>>>',m1,m2,s
                    self.u_shp_name.append(s)
                    self.u_shp_value.append(self.all_valid_hypotheses['F_SHAPE'][s][s1]/self.all_valid_hypotheses['sum']['F_SHAPE'])

        relations.append('F_DIR')

        unique_feature = []
        # this is designed to work with only on top so the hight of the object is how many objects under it
        for R in range(0,int(z1)+1):
            for L in range(1,len(features)+1):
                for subset in itertools.combinations(features,L):
                    objects = 0
                    objects = {}
                    for f in list(subset):
                        # print '>>>>',R,L,f
                        objects[f] = []
                        feature_v = scene_layout[I][f]
                        for o in scene_layout:
                            if o != I:
                                if feature_v == scene_layout[o][f]:
                                    objects[f].append(o)
                        if R>0:
                            objects_to_be_removed = []
                            for o in objects[f]:
                                if scene_layout[o]['F_POS'][2]<R:
                                    objects_to_be_removed.append(o)
                            # print 'TBR',objects_to_be_removed
                            # make sure the similar objects have at least one object below them
                            for o in objects_to_be_removed:
                                objects[f].remove(o)
                            # get the relation object
                            pos = scene_layout[I]['F_POS']
                            pos[2] -= R
                            for o in scene_layout:
                                if scene_layout[o]['F_POS'] == pos:
                                    I_R = o
                            # make sure that similar objects have different objects with this relation
                            objects_to_be_removed = []
                            feature_R_v = scene_layout[I_R][f]
                            for o in objects[f]:
                                # get the relation object for similar object
                                pos = scene_layout[o]['F_POS']
                                pos[2] -= R
                                for o2 in scene_layout:
                                    if scene_layout[o2]['F_POS'] == pos:
                                        I_R2 = o2
                                #print feature_R_v
                                #print scene_layout[I_R2][f]
                                #print '-................-'
                                if feature_R_v != scene_layout[I_R2][f]:
                                    objects_to_be_removed.append(o)
                            # remove all objects that their relational object does not match the actual object
                            for o in objects_to_be_removed:
                                objects[f].remove(o)
                            #print objects
                    #
                    # if R==0:
                    #     for f in list(subset):
                    #         objects[f] = []
                    #         feature_v = scene_layout[I][f]
                    #         for o in scene_layout:
                    #             if o != I:
                    #                 if feature_v == scene_layout[o][f]:
                    #                     objects[f].append(o)
                                #print f,feature_v,scene_layout[o][f]
                    # all the objects are saved in intersection, now try and see if features are good enough to get a unique object
                    intersection = scene_layout.keys()
                    intersection.remove(I)
                    # print '>>>>>',intersection
                    for f in objects:
                        intersection = [val for val in intersection if val in objects[f]]
                    if intersection == []:
                        unique_feature.append([list(subset),R])
                if unique_feature != []:        break
            if unique_feature != []:        break
        return unique_feature

#--------------------------------------------------------------------------------------------------------#
    def _find_unique_feature_F(self,I,positions):
        I_R = -1
        x1 = positions[I]['x'][0]/7.0
        y1 = positions[I]['y'][0]/7.0
        z1 = positions[I]['z'][0]

        scene_layout = {}
        for i in positions:
            scene_layout[i] = {}
            scene_layout[i]['F_SHAPE'] = positions[i]['F_SHAPE']
            scene_layout[i]['F_HSV'] = positions[i]['F_HSV']
            scene_layout[i]['F_POS'] = [positions[i]['x'][0]/7.0,positions[i]['y'][0]/7.0,positions[i]['z'][0]]

        # Test which features can be used from the moving objects
        # 1) check pos first
        features = []
        use_pos = 0
        for p in self.all_valid_hypotheses['F_POS']:
            for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                m1 = np.asarray([x1,y1,z1])
                m2 = np.asarray(list(p1))
                # print p,m1,m2
                if self._distance_test(m1,m2)<.1:
                    if 'F_POS' not in features:
                        features.append('F_POS')
        self.u_pos_f.append(m1)

        # print features

        unique_feature = []
        # for R in range(0,int(z1)+1):
        R = 0
        for L in range(1,len(features)+1):
            for subset in itertools.combinations(features,L):
                objects = {}
                for f in list(subset):
                    objects[f] = []
                    feature_v = scene_layout[I][f]
                    for o in scene_layout:
                        if o != I:
                            if feature_v == scene_layout[o][f]:
                                objects[f].append(o)

                intersection = scene_layout.keys()
                intersection.remove(I)
                # print '>>>>>',intersection
                for f in objects:
                    intersection = [val for val in intersection if val in objects[f]]
                if intersection == []:
                    unique_feature.append([list(subset),R])
            if unique_feature != []:        break

        if unique_feature == []:

            features.append('F_HSV')
            features.append('F_SHAPE')
            for R in range(1,int(z1)+1):
                for L in range(1,len(features)+1):
                    for subset in itertools.combinations(features,L):
                        objects = 0
                        objects = {}
                        for f in list(subset):
                            # print '>>>>',R,L,f
                            objects[f] = []
                            pos = scene_layout[I]['F_POS'][:]
                            pos[2] -= R
                            for o in scene_layout:
                                if scene_layout[o]['F_POS'] == pos:
                                    I_R = o
                                    print '>>>>>>>>>>',scene_layout[o]['F_POS'],pos,I_R
                            feature_v = scene_layout[I_R][f]
                            for o in scene_layout:
                                if o != I_R:
                                    if feature_v == scene_layout[o][f]:
                                        objects[f].append(o)
                            # if R>0:
                                # objects_to_be_removed = []
                                # for o in objects[f]:
                                #     if scene_layout[o]['F_POS'][2]<R:
                                #         objects_to_be_removed.append(o)
                                # print 'TBR',objects_to_be_removed
                                # make sure the similar objects have at least one object below them
                                # for o in objects_to_be_removed:
                                #     objects[f].remove(o)
                                # get the relation object

                                # make sure that similar objects have different objects with this relation
                                # objects_to_be_removed = []
                                # feature_R_v = scene_layout[I_R][f]
                                # for o in objects[f]:
                                #     # get the relation object for similar object
                                #     pos = scene_layout[o]['F_POS']
                                #     pos[2] -= R
                                #     for o2 in scene_layout:
                                #         if scene_layout[o2]['F_POS'] == pos:
                                #             I_R2 = o2
                                #     #print feature_R_v
                                #     #print scene_layout[I_R2][f]
                                #     #print '-................-'
                                #     if feature_R_v != scene_layout[I_R2][f]:
                                #         objects_to_be_removed.append(o)
                                # remove all objects that their relational object does not match the actual object
                                # for o in objects_to_be_removed:
                                #     objects[f].remove(o)

                                #print f,feature_v,scene_layout[o][f]
                    # all the objects are saved in intersection, now try and see if features are good enough to get a unique object
                    intersection = scene_layout.keys()
                    intersection.remove(I_R)
                    # print '>>>>>',intersection
                    for f in objects:
                        intersection = [val for val in intersection if val in objects[f]]
                    if intersection == []:
                        unique_feature.append([list(subset),R])
                if unique_feature != []:        break
        self.features_final = features

        # self.u_hsv.append(np.asarray(list(positions[I]['F_HSV'])))
        if I_R != -1:
            for c in self.all_valid_hypotheses['F_HSV']:
                for c1 in self.all_valid_hypotheses['F_HSV'][c].keys():
                    m1 = np.asarray(list(positions[I_R]['F_HSV']))
                    m2 = np.asarray(list(c1))
                    if self._distance_test(m1,m2)<.25:
                        self.u_hsvR_name.append(c)
                        # print self.all_valid_hypotheses['F_HSV'].keys()
                        self.u_hsvR_value.append(self.all_valid_hypotheses['F_HSV'][c][c1]/self.all_valid_hypotheses['sum']['F_HSV'])

            # self.u_shp.append(np.asarray(list([positions[I]['F_SHAPE']])))
            for s in self.all_valid_hypotheses['F_SHAPE']:
                for s1 in self.all_valid_hypotheses['F_SHAPE'][s].keys():
                    m1 = np.asarray(list([positions[I_R]['F_SHAPE']]))
                    m2 = np.asarray(list([s1]))
                    if self._distance_test(m1,m2)<.25:
                        print '>>>>>>>>>>>>>>-----',m1,m2,s
                        self.u_shpR_name.append(s)
                        self.u_shpR_value.append(self.all_valid_hypotheses['F_SHAPE'][s][s1]/self.all_valid_hypotheses['sum']['F_SHAPE'])


        return unique_feature

    #--------------------------------------------------------------------------------------------------------#
    def _single_feature(self,I,positions):
        x1 = positions[I]['x'][0]
        y1 = positions[I]['y'][0]
        z1 = positions[I]['z'][0]

        print '>>>>',x1,y1,z1

        self.u_pos = []
        self.u_hsv = []
        self.u_shp = []

        self.u_hsv_name = []
        self.u_shp_name = []

        self.u_hsv_value = []
        self.u_shp_value = []
        #check position
        use_pos = 0
        for p in self.all_valid_hypotheses['F_POS']:
            for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                m1 = np.asarray([x1/7.0,y1/7.0,z1])
                m2 = np.asarray(list(p1))
                if self._distance_test(m1,m2)<.25:
                    use_pos = 1

        if use_pos:
            self.u_pos.append(m1)

        for obj in positions:
            use_color = 1
            use_shape = 1
            if obj != I:
                #check color
                m1 = np.asarray(list(positions[I]['F_HSV']))
                m2 = np.asarray(list(positions[obj]['F_HSV']))
                if self._distance_test(m1,m2)<.25:
                    use_color = 0
                #check shape
                m1 = np.asarray(list([positions[I]['F_SHAPE']]))
                m2 = np.asarray(list([positions[obj]['F_SHAPE']]))
                if self._distance_test(m1,m2)<.25:
                    use_shape = 0

        if use_color:
            self.u_hsv.append(np.asarray(list(positions[I]['F_HSV'])))
            for c in self.all_valid_hypotheses['F_HSV']:
                for c1 in self.all_valid_hypotheses['F_HSV'][c].keys():
                    m1 = np.asarray(list(positions[I]['F_HSV']))
                    m2 = np.asarray(list(c1))
                    if self._distance_test(m1,m2)<.25:
                        self.u_hsv_name.append(c)
                        # print self.all_valid_hypotheses['F_HSV'].keys()
                        self.u_hsv_value.append(self.all_valid_hypotheses['F_HSV'][c][c1]/self.all_valid_hypotheses['sum']['F_HSV'])
        if use_shape:
            self.u_shp.append(np.asarray(list([positions[I]['F_SHAPE']])))
            for s in self.all_valid_hypotheses['F_SHAPE']:
                for s1 in self.all_valid_hypotheses['F_SHAPE'][s].keys():
                    m1 = np.asarray(list([positions[I]['F_SHAPE']]))
                    m2 = np.asarray(list([s1]))
                    if self._distance_test(m1,m2)<.25:
                        self.u_shp_name.append(s)
                        self.u_shp_value.append(self.all_valid_hypotheses['F_SHAPE'][s][s1]/self.all_valid_hypotheses['sum']['F_SHAPE'])


        # print self.u_pos,self.u_hsv,self.u_shp

    #--------------------------------------------------------------------------------------------------------#
    def _single_feature_f(self,I,positions):
        x1 = positions[I]['x'][0]
        y1 = positions[I]['y'][0]
        z1 = positions[I]['z'][0]
        self.u_pos_f = []
        self.u_hsv_f = []
        self.u_shp_f = []
        #check position
        use_pos = 0
        for p in self.all_valid_hypotheses['F_POS']:
            for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                m1 = np.asarray([x1/7.0,y1/7.0,z1])
                m2 = np.asarray(list(p1))
                # print m1,m2
                if self._distance_test(m1,m2)<.25:
                    use_pos = 1

        if use_pos:
            self.u_pos_f.append(m1)

        # print self.u_pos_f

    #--------------------------------------------------------------------------------------------------------#
    def _distance_test(self,m1,m2):
        return np.sqrt(np.sum((m1-m2)**2))/np.sqrt(len(m1))

    #--------------------------------------------------------------------------------------------------------#
    def _generate_all_sentences(self):
        # max_num_of_sentences = 500
        # max_num_in_each_sentences = 100
        print '--------------------------- Generating all sentences'
        all_sentences = {}
        # general sentence structure
        # self.motion = [0,1]
        for S in self.N['S']:
            if S != 'sum':
                # first kind of sentences
                if self.motion == [0,1,0]:
                    conditions = ['E2FV2','E1','FV1','E2c','FV2c']
                    if 'E2_FV2' in S.split(' '):
                        all_sentences[S] = self.N['S'][S]/self.N['S']['sum']
                    if '_S' in S.split(' '):
                        all_sentences['CH_POS_E1 E1 _S_connect CH_POS_FV1 FV1'] = self.N['S'][S]/self.N['S']['sum']

                # pick up kind of sentences
                if self.motion == [0,1]:
                    conditions = ['E1']
                    if 'E1' in S.split(' '):
                        all_sentences[S] = self.N['S'][S]/self.N['S']['sum']

                # put down kind of sentences
                if self.motion == [1,0]:
                    conditions = ['FV1']
                    if 'FV1' in S.split(' '):
                        all_sentences[S] = self.N['S'][S]/self.N['S']['sum']

        # Verbs
        print '--------------------------- Updating verbs'
        for condition in conditions:
            new_sentences = {}
            for S in all_sentences:
                V1 = all_sentences[S]
                S = S.split(' ')
                changed = 0
                for count,part in enumerate(S):
                    if 'CH_' in part and condition in part:
                        changed = 1
                        for i in self.T['features'][part]:
                            S[count] = i
                            new_sentences[' '.join(S[:])] = V1*self.T['features'][part][i]/self.T['sum'][part]
                if not changed:
                    new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

        # Conditions
        print '--------------------------- Updating conditions'
        for condition in ['E2_FV2','E2','FV2','E2c','FV2c','E1','FV1']:
            new_sentences = {}
            for S in all_sentences:
                V1 = all_sentences[S]
                S = S.split(' ')
                changed = 0
                for count,part in enumerate(S):
                    if condition == part:
                        changed = 1
                        for i in self.N[part]:
                            if i != 'sum':
                                # print '>>>>>>>',i
                                S[count] = i
                                # print '>>>>>>>',' '.join(S[:])
                                # print '>>>>>>>',V1
                                # print self.N[part][i]
                                # print self.N[part]

                                new_sentences[' '.join(S[:])] = V1*self.N[part][i]/self.N[part]['sum']
                                # print part,V1*self.N[part][i]/self.N[part]['sum']
                if not changed:
                    new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

        print '--------------------------- Updating connections'
        for condition in ['connect']:
            new_sentences = {}
            for S in all_sentences:
                V1 = all_sentences[S]
                S = S.split(' ')
                changed = 0
                for count,part in enumerate(S):
                    if condition in part:
                        changed = 1
                        for i in self.N[part]:
                            if i != 'sum':
                                S[count] = i
                                new_sentences[' '.join(S[:])] = V1*self.N[part][i]/self.N[part]['sum']
                if not changed:
                    new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

        # print all_sentences
        print '--------------------------- Updating non_terminals'
        new_sentences = {}
        for S in all_sentences:
            V1 = all_sentences[S]
            S = S.split(' ')
            changed = 0
            for count,part in enumerate(S):
                if part in self.N:
                    if part not in ['_entity','_relation','_F_POS','_Direction']:
                        for i in self.N[part]:
                            if i != 'sum':
                                S[count] = i
                                new_sentences[' '.join(S[:])] = V1*self.N[part][i]/self.N[part]['sum']
            if not changed:
                new_sentences[' '.join(S[:])] = V1
        all_sentences = new_sentences.copy()


        # print all_sentences
        # print self.u_hsv_name
        # if self.motion == [0,1,0]:
        # update the E
        print '--------------------------- Updating E-condition'
        new_sentences = {}
        for S in all_sentences:
            V1 = all_sentences[S]
            S = S.split(' ')
            changed = 0
            for count,part in enumerate(S):
                if '_' in part:
                    # if its single feature
                    if self.feature_I[0][1] == 0:
                        ok_features = []
                        for f1 in self.feature_I:
                            for f2 in f1[0]:
                                ok_features.append(f2)
                        if part == '_entity':
                            for i in self.N[part]:
                                if i != 'sum' and '_entity' not in i:
                                    # print '>>',i
                                    for j in ok_features:
                                        if j in i:
                                            # print '>>>',i
                                            for s,sv in zip(self.u_shp_name, self.u_shp_value):
                                                for c,cv in zip(self.u_hsv_name, self.u_hsv_value):
                                                    val = 1
                                                    # print '>>',s,c
                                                    # entity = ''
                                                    changed = 1
                                                    i2 = i[:]
                                                    i2 = i2.split(' ')
                                                    for ccc,k in enumerate(i2):
                                                        if k == 'F_HSV':
                                                            i2[ccc] = c
                                                            val *= cv
                                                        if k == 'F_SHAPE':
                                                            i2[ccc] = s
                                                            val *= sv
                                                        # if '_' not in (' ').join(i2):
                                                            # print '-----------',i2,self.N[part][i]/self.N[part]['sum']*val
                                                    S[count] = ' '.join(i2)
                                                    new_sentences[' '.join(S[:])] = V1*self.N[part][i]/self.N[part]['sum']*val

                    if self.feature_I[0][1] == 1:
                        ok_features = []
                        for f1 in self.feature_I:
                            for f2 in f1[0]:
                                ok_features.append(f2)
                        if part == '_entity':
                            for i in self.N[part]:
                                # print i
                                if i != 'sum' and '_entity' not in i:
                                    # print '>>',i
                                    for j in ok_features:
                                        if j in i:
                                            # print '>>>',i
                                            for s,sv in zip(self.u_shp_name, self.u_shp_value):
                                                for c,cv in zip(self.u_hsv_name, self.u_hsv_value):
                                                    val = 1
                                                    # print '>>',s,c
                                                    # entity = ''
                                                    changed = 1
                                                    i2 = i[:]
                                                    i2 = i2.split(' ')
                                                    for ccc,k in enumerate(i2):
                                                        if k == 'F_HSV':
                                                            i2[ccc] = c
                                                            val *= cv
                                                        if k == 'F_SHAPE':
                                                            i2[ccc] = s
                                                            val *= sv
                                                        # if '_' not in (' ').join(i2):
                                                            # print '-----------',i2,self.N[part][i]/self.N[part]['sum']*val
                                                    S[count] = ' '.join(i2)
                                                    new_sentences[' '.join(S[:])] = V1*self.N[part][i]/self.N[part]['sum']*val
                    break
            if not changed:
                new_sentences[' '.join(S[:])] = V1
        all_sentences = new_sentences.copy()

        # print all_sentences

        if self.motion == [0,1,0] or self.motion == [1,0]:
            # update the FV
            print '--------------------------- Updating FV-condition'
            # print '>>>>>>>>>>>',self.u_pos_f
            new_sentences = {}
            counter = 0
            for S in all_sentences:
                counter += 1
                # print counter
                V1 = all_sentences[S]
                S = S.split(' ')
                changed = 0
                for count,part in enumerate(S):
                    if '_' in part:
                        # if its a location
                        if self.feature_F[0][1] == 0:
                            if part == '_F_POS':
                                changed = 1
                                for p in self.all_valid_hypotheses['F_POS']:
                                    for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                                        m1 = self.u_pos_f[0]
                                        m2 = np.asarray(list(p1))
                                        if self._distance_test(m1,m2)<.25:
                                            S[count] = p
                                            new_sentences[' '.join(S[:])] = V1#/self._distance_test(m1,m2)
                        if self.feature_F[0][1] == 1:
                            # print 'YAAAAAAAAY'
                            if part == '_relation' :
                                for i in self.N[part]:
                                    if i != 'sum' and '_relation' in i:
                                        i2 = i[:]
                                        i2 = i2.split(' ')
                                        # make sure there are at least _relation and _entity
                                        countt = 0
                                        P1 = 1
                                        for j2 in i2:
                                            if '_' in j2:
                                                countt += 1
                                        if countt == 2:
                                            for count2,part2 in enumerate(i2):
                                                if part2 == '_relation':
                                                    for i3 in self.N[part2]:
                                                        if i3 != 'sum' and '_relation' not in i3:
                                                            i2[count2] = i3
                                                            P1 *= self.N[part2][i3]/self.N[part2]['sum']
                                                if part2 == '_entity':
                                                    for i3 in self.N[part2]:
                                                        if i3 != 'sum' and '_entity' not in i3:
                                                            i2[count2] = i3
                                                            # print '>>>>.',i3
                                                            # P1 *= self.N[part2][i3]/self.N[part2]['sum']
                                                            S[count] = ' '.join(i2)
                                                            new_sentences[' '.join(S[:])] = V1*self.N[part2][i3]/self.N[part2]['sum']
                                                            changed = 1
                                                # print '>>>',S

                if not changed:
                    new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

            # print len(all_sentences.keys())

            if self.feature_F[0][1] == 1:
                new_sentences = {}
                for S in all_sentences:
                    # if 'F_POS' not in S:
                        # print '>>>>>>',S

                    # print S
                    V1 = all_sentences[S]
                    S = S.split(' ')
                    changed = 0
                    for count,part in enumerate(S):
                        # print '>>>>>>>',part
                        if '_' in part:
                            # print '>>>>>>>',part
                            if part == 'F_DIR':
                                for i in self.T['features'][part]:
                                    if i != 'sum' and 'F_DIR' not in i:
                                        changed = 1

                                        if '_' in i:
                                            i2 = i.split('_')
                                            # print i2
                                            i2 = ' '.join(i2[1:])
                                            # print i2
                                            # print '-----'
                                        else:
                                            i2 = i
                                        S[count] = i2
                                        # print i2,self.T['features'][part][i]/self.T['sum'][part]
                                        new_sentences[' '.join(S[:])] = V1*self.T['features'][part][i]/self.T['sum'][part]
                                        # print S,V1*self.T['features'][part][i]/self.T['sum'][part]
                            break
                        # if its a relation
                    if not changed:
                        new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

            if 'F_HSV' not in self.features_final:  self.features_final.append('F_HSV')
            if 'F_SHAPE' not in self.features_final:  self.features_final.append('F_SHAPE')

            if self.feature_F[0][1] == 1:
                new_sentences = {}
                for S in all_sentences:
                    V1 = all_sentences[S]
                    S = S.split(' ')
                    changed = 0
                    for count,part in enumerate(S):
                        if '_' in part and part in self.features_final:
                            if part == 'F_HSV':
                                for c,cv in zip(self.u_hsvR_name, self.u_hsvR_value):
                                    changed = 1
                                    S[count] = c
                                    new_sentences[' '.join(S[:])] = V1*cv
                    if not changed:
                        new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

            if self.feature_F[0][1] == 1:
                new_sentences = {}
                for S in all_sentences:
                    V1 = all_sentences[S]
                    S = S.split(' ')
                    changed = 0
                    for count,part in enumerate(S):
                        if '_' in part and part in self.features_final:
                            if part == 'F_SHAPE':
                                for c,cv in zip(self.u_shpR_name, self.u_shpR_value):
                                    changed = 1
                                    S[count] = c
                                    new_sentences[' '.join(S[:])] = V1*cv
                    if not changed:
                        new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

            if self.feature_F[0][1] == 1:
                new_sentences = {}
                for S in all_sentences:
                    V1 = all_sentences[S]
                    S = S.split(' ')
                    changed = 0
                    for count,part in enumerate(S):
                        if '_' in part and part in self.features_final:
                            if part == 'F_POS':
                                for p in self.all_valid_hypotheses['F_POS']:
                                    for p1 in self.all_valid_hypotheses['F_POS'][p].keys():
                                        m1 = self.u_pos_f[0]
                                        m2 = np.asarray(list(p1))
                                        if self._distance_test(m1,m2)<.25:
                                            S[count] = p
                                            new_sentences[' '.join(S[:])] = V1#/self._distance_test(m1,m2)
                    if not changed:
                        new_sentences[' '.join(S[:])] = V1
            all_sentences = new_sentences.copy()

            # # including words order
            # print '--------------------------- Updating Word Order'
            # for i in all_sentences:
            #     words = i.split(' ')
            #     words_order_val = [10]
            #     for j in range(len(words)-1):
            #         if words[j] in self.words_order:
            #             if words[j+1] in self.words_order[words[j]]:
            #                 words_order_val.append(1+float(self.words_order[words[j]][words[j+1]])/float(self.words_order[words[j]]['count']))
            #     all_sentences[i] *= np.min(words_order_val)
            #     # print '>>>>..',i,np.min(words_order_val)


        print
        print '==----------------- All sentences ---------------=='
        print
        sentences = ''
        sorted_x = sorted(all_sentences.items(), key=operator.itemgetter(1))
        counter = 0
        for count,i in enumerate(reversed(sorted_x)):
            # print count
            # if '_' not in i[0]:
            #     print '>>>>>>>>',i

            # print count
            words =  i[0].split(' ')
            ok = 1
            for j in range(len(words)-1):
                if words[j] == words[j+1] or '_' in words[j] or '_' in words[j+1] or words[j] == '' or words[j+1] == '':
                    ok = 0
            if ok:
                counter += 1
                # print test
                print ' '.join(words),':',i[1]
                sentences += str(counter)+'-'+' '.join(words)+'\n'
            if counter==30: break

        sentences
        file1 = open("/home/omari/Dropbox/robot_modified/IT/generation/sentences-"+str(self.scene)+".txt", "w")
        file1.write(sentences)
        file1.close()


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
        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['initial']]   # initial layout
        l2 = self.Data['layouts'][self.Data['scenes'][self.scene]['final']]     # final layout
        I = self.Data['scenes'][self.scene]['I_move']
        F = self.Data['scenes'][self.scene]['F_move']
        self.motion = []
        if I != []:
            # move the robot alone
            initial_position = self.Data['gripper'][self.Data['scenes'][self.scene]['initial']]
            final_position = l1[I]['position']
            if final_position != initial_position:
                self.motion.append(0)
            #     a1,a2,a3 = self._inverse_kinematics(final_position[0],final_position[1],final_position[2])
            #     self.rotate_robot(-a1,-a2,-a3,save)
            # if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'cube':
            #     z_offset = +.22
            # if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'sphere':
            #     z_offset = +.22
            # if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'cylinder':
            #     z_offset = -.18
            # if self.object_shape[(final_position[0],final_position[1],final_position[2])] == 'prism':
            #     z_offset = -.18
            # move the robot and the object
            initial_position = final_position
            final_position = l2[F]['position']
            if final_position != initial_position:
                self.motion.append(1)
            #     self.rotate_robot_with_object(initial_position, final_position, z_offset, save)
            # move the robot and the object down
            initial_position = final_position
            final_position = self.Data['gripper'][self.Data['scenes'][self.scene]['final']]
            if final_position != initial_position:
                self.motion.append(0)
            #     a1,a2,a3 = self._inverse_kinematics(final_position[0],final_position[1],final_position[2])
            #     self.rotate_robot(-a1,-a2,-a3,save)

    #-----------------------------------------------------------------------------------------------------#     save motion
    def _save_motion(self):
        target = open('/home/omari/Datasets/robot_modified/motion/scene'+str(self.scene)+'.txt', 'w')
        target2 = open('/home/omari/Dropbox/robot_modified/motion/scene'+str(self.scene)+'.txt', 'w')
        for F in [target,target2]:
            for i in self.sentences:
                F.write('sentence:'+str(i)+'\n')
                F.write(self.sentences[i][0]+':'+self.sentences[i][1]+'\n')

            for key in self.positions:
                F.write('object:'+str(key)+'\n')
                F.write('x:')
                for i in self.positions[key]['x']:
                    F.write("{:3.2f}".format(i)+',')
                F.write("\n")
                F.write('y:')
                for i in self.positions[key]['y']:
                    F.write("{:3.2f}".format(i)+',')
                F.write("\n")
                F.write('z:')
                for i in self.positions[key]['z']:
                    F.write("{:3.2f}".format(i)+',')
                F.write("\n")
                if key != 'gripper':
                    c = self.positions[key]['F_HSV']
                    s = self.positions[key]['F_SHAPE']
                    F.write('F_HSV:'+str(c[0])+','+str(c[1])+','+str(c[2]))
                    F.write("\n")
                    F.write('F_SHAPE:'+str(s))
                    F.write("\n")
            F.close()

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
            x=0, y=0, width=2000, height=2000,
            center=(self.chess_shift_x,self.chess_shift_y,0),
            forward=(self.chess_shift_x-10,self.chess_shift_y-3,-7),
            background=(1,1,1))
        distant_light(direction=(-150,-150,1), color=color.white)
        self.label = label(pos=(10,10,10), text='Scene number : ',height=20,color=(0,0,0))
        checkerboard = ( (.9,1,.9,1,.9,1,.9,1),
				 (1,.9,1,.9,1,.9,1,.9),
				 (.9,1,.9,1,.9,1,.9,1),
				 (1,.9,1,.9,1,.9,1,.9),
		 		 (.9,1,.9,1,.9,1,.9,1),
				 (1,.9,1,.9,1,.9,1,.9),
				 (.9,1,.9,1,.9,1,.9,1),
				 (1,.9,1,.9,1,.9,1,.9) )
        tex = materials.texture(data=checkerboard,
            mapping="sign",
            interpolate=False)
        chess1 = box(pos=(self.chess_shift_x,self.chess_shift_y,-.3),axis=(0,0,1),size=(.4,9,9),
            color=color.orange,material=materials.wood)
        chess2 = box(pos=(self.chess_shift_x,self.chess_shift_y,-.25),axis=(0,0,1), size=(.5,8,8),
            color=color.orange, material=tex)
        # x = arrow(pos=(1,6,0),axis=(1,0,0),length=2,shaftwidth=.2,color=color.red)
        # y = arrow(pos=(1,6,0),axis=(0,1,0),length=2,shaftwidth=.2,color=color.green)
        # z = arrow(pos=(1,6,0),axis=(0,0,1),length=2,shaftwidth=.2,color=color.blue)

    #-----------------------------------------------------------------------------------------------------#     initial draw robot
    def draw_robot(self):
		base_1 = box(pos=(0,self.chess_shift_y,-.25),axis=(0,0,1), size=(.5,2,2),color=color.black,
		    material=materials.plastic)
		c1 = .35
		c2 = .15
		base_2 = Polygon( [(-1,0), (-.75,self.len_base), (.75,self.len_base), (1,0)] )
		base_3 = shapes.circle(pos=(0,self.len_base), radius=.75)
		base_4 = shapes.circle(pos=(0,self.len_base), radius=0.2)
		base_s = [(0,self.chess_shift_y-.5,0),(0,self.chess_shift_y+.5,0)]
		self.base = extrusion(pos=base_s, shape=base_2+base_3-base_4, color=(c1,c1,c1))

		arm1_1 = Polygon( [(0,.75), (self.len_arm1,.5), (self.len_arm1,-.5), (0,-.75)] )
		arm1_2 = shapes.circle(pos=(0,0), radius=.75)
		arm1_3 = shapes.circle(pos=(0,0), radius=0.2)
		arm1_4 = shapes.circle(pos=(self.len_arm1,0), radius=.5)
		arm1_5 = shapes.circle(pos=(self.len_arm1,0), radius=0.2)
		arm1_s = [(0,self.chess_shift_y+.5,self.len_base),(0,self.chess_shift_y+1.5,self.len_base)]
		self.arm1 = extrusion(pos=arm1_s, shape=arm1_1+arm1_2-arm1_3+arm1_4-arm1_5, color=(c2,c2,c2))

		arm2_1 = Polygon( [(0,.5), (self.len_arm2,.4), (self.len_arm2,-.4), (0,-.5)] )
		arm2_2 = shapes.circle(pos=(0,0), radius=.5)
		arm2_3 = shapes.circle(pos=(0,0), radius=0.2)
		arm2_4 = shapes.circle(pos=(self.len_arm2,0), radius=.4)
		arm2_5 = shapes.circle(pos=(self.len_arm2,0), radius=0.2)
		arm2_s = [(self.len_arm1,self.chess_shift_y-.5,self.len_base),(self.len_arm1,self.chess_shift_y+.5,self.len_base)]
		self.arm2 = extrusion(pos=arm2_s, shape=arm2_1+arm2_2-arm2_3+arm2_4-arm2_5, color=(c1,c1,c1))

		gripper_1 = Polygon( [(0,.4), (self.len_gripper,.3), (self.len_gripper,-.3), (0,-.4)] )
		gripper_2 = shapes.circle(pos=(0,0), radius=.4)
		gripper_4 = shapes.circle(pos=(self.len_gripper,0), radius=.3)
		gripper1_s = [(self.len_arm1+self.len_arm2,self.chess_shift_y-.6,self.len_base),(self.len_arm1+self.len_arm2,self.chess_shift_y-.5,self.len_base)]
		gripper2_s = [(self.len_arm1+self.len_arm2,self.chess_shift_y+.5,self.len_base),(self.len_arm1+self.len_arm2,self.chess_shift_y+.6,self.len_base)]

		self.gripper1 = extrusion(pos=gripper1_s, shape=gripper_1+gripper_2+gripper_4, color=(c2,c2,c2))
		self.gripper2 = extrusion(pos=gripper2_s, shape=gripper_1+gripper_2+gripper_4, color=(c2,c2,c2))

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
        # if self.frame_number<10:        j = '000'+str(self.frame_number)
        # elif self.frame_number<100:     j = '00'+str(self.frame_number)
        # elif self.frame_number<1000:    j = '0'+str(self.frame_number)
        # if self.scene<10:      k = '000'+str(self.scene)
        # elif self.scene<100:    k = '00'+str(self.scene)
        # elif self.scene<1000:   k = '0'+str(self.scene)
        img.SaveFile('/home/omari/Dropbox/robot_modified/IT/generation/'+str(self.scene)+'-scene.png', wx.BITMAP_TYPE_PNG)
        # img.SaveFile(self.image_dir2+str(self.scene)+'/scene_'+k+'_frame_'+j+'.png', wx.BITMAP_TYPE_PNG)

        print self.frame_number,'image saved..'
        self.frame_number+=1

    def forward_arms(self,a1,a2,a3):
		z = self.l2*np.sin(a2)+self.l3*np.sin(a3+a2)
		r = self.l2*np.cos(a2)+self.l3*np.cos(a3+a2)
		x = r*np.cos(a1)
		y = r*np.sin(a1)
		return (x,y+self.chess_shift_y,1.8-z)
