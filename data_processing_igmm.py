import numpy as np
import networkx as nx
import itertools
from nltk import PCFG
import operator
#from sklearn import mixture
from GMM_functions import *
from GMM_BIC import *
from IGMM import *
import copy
import colorsys

import multiprocessing

"""
Focus 7bibi
el pcfg, the words that you don't know yet, keep them as a terminal. and build relations between all the word in the sentence. Once you figure out a word that belong to a feature.
Convert all the probabilities of that terminal the feature terminal .. ! this should work !
"""


#--------------------------------------------------------------------------------------------------------#
def _distance_test(m1,m2):
    return np.sqrt(np.sum((m1-m2)**2))/np.sqrt(len(m1))

#--------------------------------------------------------------------------------------------------------#
def calc(data):
    subset = data[0]
    scene = data[1]
    indices = data[2]
    hyp_language_pass = data[3]
    total_motion = data[4]
    S = data[5]
    parsed_sentence = ''
    value_sentence = ''
    # no 2 phrases are allowed to intersect in the same sentence
    no_intersection = 1
    all_indeces = []
    for w in subset:
        #print w
        for w1 in indices[scene][w]:
            for w2 in w1:
                if w2 not in all_indeces:
                    all_indeces.append(w2)
                else:
                    no_intersection = 0
    #print no_intersection
    #print '------'

    if no_intersection:
        # this function tests one susbet of words at a time
        all_possibilities = []      # all the possibilities gathered in one list
        for word in subset:
            all_possibilities.append(hyp_language_pass[word]['all'])

        #print all_possibilities
        #print '---------------',len(all_possibilities)
        # find the actual possibilities for every word in the subset
        #c = 1
        for element in itertools.product(*all_possibilities):
            #print c
            #c+=1

            # no 2 words are allowed to mean the same thing
            not_same = 1
            features = {}
            for i in element:
                if i[0] not in features: features[i[0]] = []
                features[i[0]].append(i[1])

            for f in features:
                if len(features[f])>1:
                    for f1 in range(len(features[f])-1):
                        for f2 in range(f1+1,len(features[f])):
                            m1 = np.asarray(list(features[f][f1]))
                            m2 = np.asarray(list(features[f][f2]))
                            if len(m1) != len(m2):          continue        # motions !
                            if _distance_test(m1,m2)<.25:
                                not_same = 0
                                continue
            if not_same:
                # 1) does actions match ?   it should match 100%
                hyp_motion = {}
                motion_pass = 0

                for k,word in enumerate(subset):
                    if element[k][0] == 'motion':
                        a = element[k][1]
                        if a not in hyp_motion:     hyp_motion[a] = len(indices[scene][word])
                        else:                       hyp_motion[a] += len(indices[scene][word])

                for i in total_motion:
                    if total_motion[i] == hyp_motion:
                        #print '+++',total_motion[i],hyp_motion
                        motion_pass = 1

                # 2) parse the sentence
                if motion_pass:
                    parsed_sentence = []
                    value_sentence = []
                    for i in S[scene].split(' '):
                        parsed_sentence.append('_')
                        value_sentence.append('_')
                    for word1 in subset:
                        for i1 in indices[scene][word1]:
                            for j1 in i1:
                                #print word1,j1
                                k = subset.index(word1)
                                parsed_sentence[j1] = element[k][0]
                                value_sentence[j1] = map(prettyfloat, element[k][1])

                    for k,word in enumerate(subset):
                        print word,map(prettyfloat, element[k][1])
                        #print subset
                        #print S[scene].split(' ')
                        #print parsed_sentence
                        #print value_sentence
                    print '----'
    return ([S[scene],parsed_sentence,value_sentence],)


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

class process_data():
    def __init__(self):
        self.dir1 = '/home/omari/Datasets/robot_modified/motion/scene'
        self.dir2 = '/home/omari/Datasets/robot_modified/scenes/'
        self.dir3 = '/home/omari/Datasets/robot_modified/graphs/scene'

        # initial language
        self.N                      = {}                    # non-terminals
        self.N['e']                 = {}                    # non-terminals entity
        self.N['e']['sum']          = 0.0                   # non-terminals entity counter

        # connecting language to vision hypotheses
        self.hyp_language           = {}
        self.hyp_relation           = {}
        self.hyp_motion             = {}
        self.hyp_language_pass      = {}
        self.hyp_obj_pass           = {}
        self.hyp_relation_pass      = {}
        self.hyp_motion_pass        = {}
        self.feature_pass           = {}
        self.all_total_motion       = {}
        self.all_scene_features     = {}

        self.n_word                 = 3
        self.step                   = 3
        self.all_words              = []

        #gmm
        self.gmm_obj                = {}
        self.gmm_M                  = {}
        self.cv_types = ['spherical', 'tied', 'diag', 'full']
        # self.cv_types = ['full']

        # habituation parameter
        self.p_parameter            = 3.0                       # probability parameter for exp function in habituation
        self.pass_distance          = .25                       # distance test for how much igmms match
        self.pass_distance_phrases  = .25                       # distance test for how much phrases match
        self.p_obj_pass             = .75                        # for object
        self.p_relation_pass        = .8                        # for both relation and motion
        self.pool = multiprocessing.Pool(32)

    #--------------------------------------------------------------------------------------------------------#
    # read the sentences and data from file
    def _read(self,scene):
        self.scene = scene
        print 'reading scene number:',self.scene
        f = open(self.dir1+str(scene)+'.txt', 'r')
        data = f.read().split('\n')
        self.G = nx.Graph()
        sentences = []
        objects = []
        gripper = []
        for count,line in enumerate(data):
            if line.split(':')[0] == 'sentence':    sentences.append(count+1)
            if line.split(':')[0] == 'object' and line.split(':')[1]!='gripper':    objects.append(count+1)
            if line.split(':')[0] == 'object' and line.split(':')[1]=='gripper':    gripper.append(count+1)
        # reading sentences
        self.S = {}
        for count,s in enumerate(sentences):
            if data[s].split(':')[0] == 'GOOD': self.S[count] = (data[s].split(':')[1]).lower()
        # reading Data of objects
        self.Data = {}
        for count,o in enumerate(objects):
            self.G.add_node(str(count),type1='obj')
            self.Data[count] = {}
            self.Data[count][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))          #x
            self.Data[count][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))      #y
            self.Data[count][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))      #z
            self.Data[count][data[o+3].split(':')[0]] = np.asarray(map(float,data[o+3].split(':')[1].split(',')))           #color
            self.Data[count][data[o+4].split(':')[0]] = float(data[o+4].split(':')[1])                                      #shape
        # reading Data of robot
        for o in gripper:
            self.G.add_node('G',type1='G')
            self.Data['G'] = {}
            self.Data['G'][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))      #x
            self.Data['G'][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))  #y
            self.Data['G'][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))  #z

    #--------------------------------------------------------------------------------------------------------#
    # print sentences on terminal
    def _print_scentenses(self):
        for count,i in enumerate(self.S):
            print count,'-',self.S[i]
        print '====-----------------------------------------------------------------===='

    #--------------------------------------------------------------------------------------------------------#
    # correcting motion planning simulation
    def _fix_data(self):
        # correction to Data removing the step
        for i in self.Data:
            self.Data[i]['x'] = np.delete(self.Data[i]['x'],[self.step,2*self.step])
            self.Data[i]['y'] = np.delete(self.Data[i]['y'],[self.step,2*self.step])
            self.Data[i]['z'] = np.delete(self.Data[i]['z'],[self.step,2*self.step])

    #--------------------------------------------------------------------------------------------------------#
    # find phrases and words for each sentence
    def _find_unique_words(self):
        self.phrases = {}                       # hold the entire phrase up to a certain number of words
        self.words = {}                         # hold the list of independent words in a sentence
        # read the sentence
        for s in self.S:
            self.phrases[s] = []
            self.words[s] = []
            sentence = self.S[s]
            w = sentence.split(' ')
            phrases = []
            for i in range(len(w)):
                if w[i]not in self.words[s]: self.words[s].append(w[i])
                for j in range(i+1,np.min([i+1+self.n_word,len(w)+1])):
                    phrases.append(' '.join(w[i:j]))
            for i in phrases:
                if i not in self.phrases[s]:    self.phrases[s].append(i)

    #--------------------------------------------------------------------------------------------------------#
    def _compute_features_for_all(self):
        # this function comutes the following
        # self.touch_all
        # self.motion_all

        # initial parameter
        self.frames = len(self.Data['G']['x'])-1     # we remove the first one to compute speed
        self.keys = self.Data.keys()
        self.n = np.sum(range(len(self.keys)))
        # computing distance and touch   (between all objects including robot)
        #self.dis_all = np.zeros((self.n,self.frames),dtype=np.float)
        self.touch_all = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)-1):
            for j in range(i+1,len(self.keys)):
                k1 = self.keys[i]
                k2 = self.keys[j]
                dx = np.abs(self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:])
                dy = np.abs(self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:])
                dz = np.abs(self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:])
                A = dx+dy+dz
                #self.dis_all[counter,:] = A
                A[A<=1.0] = 1
                A[A>1.0] = 0
                self.touch_all[counter,:] = A
                counter += 1

        """
        # computing direction   (between all objects not robot)
        self.dirx_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirx_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.diry_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.diry_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirz_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirz_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        for i in self.keys:
            for j in self.keys:
                if i!=j and i!='G' and j!='G':
                    k1 = self.keys[i]
                    k2 = self.keys[j]
                    dxi = self.Data[k1]['x'][0]-self.Data[k2]['x'][0]
                    dxf = self.Data[k1]['x'][-1]-self.Data[k2]['x'][-1]
                    dyi = self.Data[k1]['y'][0]-self.Data[k2]['y'][0]
                    dyf = self.Data[k1]['y'][-1]-self.Data[k2]['y'][-1]
                    dzi = self.Data[k1]['z'][0]-self.Data[k2]['z'][0]
                    dzf = self.Data[k1]['z'][-1]-self.Data[k2]['z'][-1]
                    self.dirx_all_i[i,j] = np.sign(dxi)
                    self.dirx_all_f[i,j] = np.sign(dxf)
                    self.diry_all_i[i,j] = np.sign(dyi)
                    self.diry_all_f[i,j] = np.sign(dyf)
                    self.dirz_all_i[i,j] = np.sign(dzi)
                    self.dirz_all_f[i,j] = np.sign(dzf)
        """

        # computing motion      (for all objects)
        for i in self.Data:
            dx = (self.Data[i]['x'][:-1]-self.Data[i]['x'][1:])**2
            dy = (self.Data[i]['y'][:-1]-self.Data[i]['y'][1:])**2
            dz = (self.Data[i]['z'][:-1]-self.Data[i]['z'][1:])**2
            self.Data[i]['motion'] = (np.round(np.sqrt(dx+dy+dz)*10000)).astype(int)
            self.Data[i]['motion'][self.Data[i]['motion']!=0] = 1

        # compute relative motion
        # similar motion == 1 different == 0
        self.motion_all = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)-1):
            for j in range(i+1,len(self.keys)):
                k1 = self.keys[i]
                k2 = self.keys[j]
                A = np.abs(self.Data[k1]['motion'] - self.Data[k2]['motion'])
                A[A==0] = 2
                A[A!=2] = 0
                A[A!=0] = 1     # a little trick to make 0=1 and 1=0
                self.motion_all[counter,:] = A
                counter += 1

    #--------------------------------------------------------------------------------------------------------#
    def _compute_features_for_moving_object(self):
        # finding the moving object ! fix this
        # self.m_obj
        # self.touch_m
        # self.touch_m_i        # initially touching the moving obj
        # self.touch_m_f        # finally touching the moving obj

        self.m_obj = []
        for i in self.Data:
            if i != 'G':
                x = np.abs(self.Data[i]['x'][-1]-self.Data[i]['x'][0])
                y = np.abs(self.Data[i]['y'][-1]-self.Data[i]['y'][0])
                z = np.abs(self.Data[i]['z'][-1]-self.Data[i]['z'][0])
                if (x+y+z) > 0:
                    self.m_obj = i

        # computing distance BINARY Distance (touch or no touch all obj no robot)
        n = len(self.Data)-2
        #self.dis_m = np.zeros((n,self.frames),dtype=np.float)
        self.touch_m = np.zeros((n,self.frames),dtype=np.uint8)
        counter = 0
        k1 = self.m_obj

        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                dx = np.abs(self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:])
                dy = np.abs(self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:])
                dz = np.abs(self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:])
                A = dx+dy+dz
                #self.dis_m[counter,:] = A
                A[A<=1.5] = 1
                A[A>1] = 0
                self.touch_m[counter,:] = A
                counter += 1

        counter = 0
        self.touch_m_i = []                 #which objects were in touch with the moving objects initially
        self.touch_m_f = []                 #which objects were in touch with the moving objects finally
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                if self.touch_m[counter,0]: self.touch_m_i.append(i)
                if self.touch_m[counter,-1]: self.touch_m_f.append(i)
                counter += 1

        # computing direction
        self.dirx_m = np.zeros((n,self.frames),dtype=np.int8)
        self.diry_m = np.zeros((n,self.frames),dtype=np.int8)
        self.dirz_m = np.zeros((n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                dx = self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:]
                dy = self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:]
                dz = self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:]
                self.dirx_m[counter,:] = np.sign((dx).astype(int))
                self.diry_m[counter,:] = np.sign((dy).astype(int))
                self.dirz_m[counter,:] = np.sign(np.round(dz))
                counter += 1

        self.dir_touch_m_i = []
        self.dir_touch_m_f = []
        for i in self.touch_m_i:
            if i > int(k1): val = i-1
            else:           val = i
            a = self.dirx_m[val,0]
            b = self.diry_m[val,0]
            c = self.dirz_m[val,0]
            d = (a,b,c)
            if d not in self.dir_touch_m_i:    self.dir_touch_m_i.append(d)
        for i in self.touch_m_f:
            if i > int(k1): val = i-1
            else:           val = i
            a = self.dirx_m[val,-1]
            b = self.diry_m[val,-1]
            c = self.dirz_m[val,-1]
            d = (a,b,c)
            if d not in self.dir_touch_m_f:    self.dir_touch_m_f.append(d)

        # finding locations of moving object
        self.locations_m_i = []
        self.locations_m_f = []
        self.locations_m_i.append((self.Data[self.m_obj]['x'][0],self.Data[self.m_obj]['y'][0]))
        self.locations_m_f.append((self.Data[self.m_obj]['x'][-1],self.Data[self.m_obj]['y'][-1]))
        #self.locations.append(di)
        #if df not in self.locations: self.locations.append(df)

    #--------------------------------------------------------------------------------------------------------#
    # find the transitions for touch and motion
    def _transition(self):
        # comput the transition intervals for motion
        self.motion = [self.Data[self.m_obj]['motion'][0]]
        col = self.motion_all[:,0]
        self.transition = {}
        self.transition['motion'] = [0]
        self.transition['all'] = [0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.motion_all[:,i]))!=0:
                col = self.motion_all[:,i]
                self.transition['motion'].append(i)
                self.transition['all'].append(i)
                self.motion.append(self.Data[self.m_obj]['motion'][i])
        # comput the transition intervals for touch
        self.transition['touch'] = [0]
        col = self.touch_all[:,0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.touch_all[:,i]))!=0:
                col = self.touch_all[:,i]
                self.transition['touch'].append(i)
                if i not in self.transition['all']: self.transition['all'].append(i)
        self.transition['all'] = sorted(self.transition['all'])

    #--------------------------------------------------------------------------------------------------------#
    # group touch and motion objects
    def _grouping(self):
        self.G_motion = self._grouping_template(self.transition['motion'],self.motion_all)
        self.G_touch = self._grouping_template(self.transition['touch'],self.touch_all)

    #-------------------------------------------------#
    # sub funtion
    def _grouping_template(self,transition,feature):
        G_all = {}
        for T in transition:
            G = self.G.copy()
            counter = 0
            for i in range(len(self.keys)-1):
                for j in range(i+1,len(self.keys)):
                    k1 = self.keys[i]
                    k2 = self.keys[j]
                    a = feature[counter,T]
                    counter += 1
                    if a == 1:    G.add_edge(str(k1),str(k2),value=1)
            G_all[T] = {}
            G_all[T]['graph'] = G
            G_all[T]['groups'] = []
            C=nx.connected_component_subgraphs(G)
            for g in C:
                G_all[T]['groups'].append(g.nodes())
        return G_all

    #--------------------------------------------------------------------------------------------------------#
    # compute unique color and shape for every scene
    def _compute_unique_color_shape(self):
        self.unique_colors = []
        self.unique_shapes = []
        for i in self.Data:
            if i != 'G':
                c = self.Data[i]['color']
                C = (c[0],c[1],c[2])
                s = self.Data[i]['shape']
                if C not in self.unique_colors: self.unique_colors.append(C)
                if s not in self.unique_shapes: self.unique_shapes.append(s)

    #--------------------------------------------------------------------------------------------------------#
    # compute unique motion for every scene
    def _compute_unique_motion(self):
        self.unique_motions = []
        self.total_motion = {}
        for i in range(2, len(self.motion)+1):  #possible windows
            self.total_motion[i-1] = {}
            for j in range(len(self.motion)+1-i):
                c = self.motion[j:j+i]
                if i == 2:  C = (c[0],c[1])
                if i == 3:  C = (c[0],c[1],c[2])
                if i == 4:  C = (c[0],c[1],c[2],c[3])
                if i == 5:  C = (c[0],c[1],c[2],c[3],c[4])
                if i == 6:  C = (c[0],c[1],c[2],c[3],c[4],c[5])

                if C not in self.unique_motions:    self.unique_motions.append(C)
                if C not in self.total_motion[i-1]:   self.total_motion[i-1][C] = 1
                else:                               self.total_motion[i-1][C] += 1
        self.all_total_motion[self.scene] = self.total_motion

    #--------------------------------------------------------------------------------------------------------#
    # compute uniqe direction for every scene
    def _compute_unique_direction(self):
        self.unique_direction = []
        for i in self.dir_touch_m_i:
            if i not in self.unique_direction:  self.unique_direction.append(i)
        for i in self.dir_touch_m_f:
            if i not in self.unique_direction:  self.unique_direction.append(i)

    #--------------------------------------------------------------------------------------------------------#
    # compute unique location for the moving object
    def _compute_unique_location(self):
        self.unique_locations = []
        for i in self.locations_m_i:
            if i not in self.unique_locations:  self.unique_locations.append(i)
        for i in self.locations_m_f:
            if i not in self.unique_locations:  self.unique_locations.append(i)

    #--------------------------------------------------------------------------------------------------------#
    # making simulation as real world
    def _convert_color_shape_location_to_gmm(self,plot):
        self.real = 0
        unique_colors = []
        unique_shapes = []
        unique_locations = []
        self.gmm_M = {}

        # more real simulation but more time !
        if self.real:
            for i in self.Data:
                if i != 'G':
                    for k in range(50):
                        c = self.Data[i]['color']
                        r = np.random.normal(0, .005, 3)
                        #c += r
                        hsv = colorsys.rgb_to_hsv(c[0], c[1], c[2])
                        r1 = np.random.normal(0, .06, 1)
                        r2 = np.random.normal(0, .05, 1)
                        r3 = np.random.normal(0, .04, 1)
                        h = hsv[0]*2*np.pi  + r1[0]
                        s = hsv[1]          + r2[0]
                        v = hsv[2]          + r3[0]
                        if s<0:      s+=2*np.abs(r2[0])
                        if s>1:      s-=2*np.abs(r2[0])
                        if v<0:      v+=2*np.abs(r3[0])
                        if v>1:      v-=2*np.abs(r3[0])
                        x,y,z = self.hsv2xyz(h,s,v)
                        C = (x[0],y[0],z[0])
                        #print c,[h,s,v],C
                        s = self.Data[i]['shape']
                        r = np.random.normal(0, .03, 1)
                        s += r[0]
                        if s<0:      s+=2*np.abs(r[0])
                        if s>1:      s-=2*np.abs(r[0])
                        unique_colors.append(C)
                        unique_shapes.append(s)
        else:
            # less real simulation but more quick
            for color in self.unique_colors:
                for k in range(10):
                    r = np.random.normal(0, .005, 3)
                    c = color+r
                    if c[0]<0:      c[0]=0
                    if c[1]<0:      c[1]=0
                    if c[2]<0:      c[2]=0
                    if c[0]>1:      c[0]=1
                    if c[1]>1:      c[1]=1
                    if c[2]>1:      c[2]=1
                    C = (c[0],c[1],c[2])
                    """
                    hsv = colorsys.rgb_to_hsv(c[0], c[1], c[2])
                    r1 = np.random.normal(0, .06, 1)
                    r2 = np.random.normal(0, .05, 1)
                    r3 = np.random.normal(0, .04, 1)
                    h = hsv[0]*2*np.pi  + r1[0]
                    s = hsv[1]          + r2[0]
                    v = hsv[2]          + r3[0]
                    if s<0:      s+=2*np.abs(r2[0])
                    if s>1:      s-=2*np.abs(r2[0])
                    if v<0:      v+=2*np.abs(r3[0])
                    if v>1:      v-=2*np.abs(r3[0])
                    x,y,z = self.hsv2xyz(h,s,v)
                    C = (x[0],y[0],z[0])
                    """
                    unique_colors.append(C)
            for shape in self.unique_shapes:
                for k in range(10):
                    r = np.random.normal(0, .005, 1)
                    s = r[0]+shape
                    if s<0:      s+=2*np.abs(r[0])
                    if s>1:      s-=2*np.abs(r[0])
                    unique_shapes.append(s)
            # just a note to should reverce x and y in testing
            for i,l in enumerate(self.unique_locations):
                    l = [l[0],l[1]]
                    for k in range(10):
                        r = np.random.normal(0, .005, 2)
                        l += r
                        for j,l1 in enumerate(l):
                            if l1<0:      l[j]+=2*np.abs(r[j])
                            if l1>7:      l[j]-=2*np.abs(r[j])
                        L = (l[0]/7.0,l[1]/7.0)
                        unique_locations.append(L)

        gmm_c = self._bic_gmm(unique_colors,plot)
        gmm_s = self._bic_gmm(unique_shapes,plot)
        gmm_l = self._bic_gmm(unique_locations,plot)
        self.gmm_M['color'] = copy.deepcopy(gmm_c)
        self.gmm_M['shape'] = copy.deepcopy(gmm_s)
        self.gmm_M['location'] = copy.deepcopy(gmm_l)

    #-------------------------------------------------------------------------------------#
    def hsv2xyz(self,H, S, V):
    	x = [S*np.cos(H)]
    	y = [S*np.sin(H)]
    	z = [V]
    	return x,y,z

    #--------------------------------------------------------------------------------------------------------#
    # To Do !
    def _convert_direction_to_gmm(self,plot):
        print self.unique_direction

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
            plot_data(X, best_gmm, bic, k, self.cv_types,0, 1)
            plt.show()
            plt.savefig("/home/omari/Datasets/robot_modified/clusters/" + str(len(X[0])) + "-" + str(self.scene) +".png")

        for i, (mean, covar) in enumerate(zip(best_gmm.means_, best_gmm._get_covars())):
            gmm1[i] = {}
            gmm1[i]['mean'] = mean
            gmm1[i]['covar'] = covar
            gmm1[i]['N'] = 1.0
        return gmm1

    #--------------------------------------------------------------------------------------------------------#
    def _build_obj_hyp_igmm(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                #if word == 'green':
                    if word not in self.gmm_obj:
                        self.gmm_obj[word] = {}
                        for f in self.gmm_M:
                            self.gmm_obj[word][f] = {}
                            self.gmm_obj[word][f]['N']    = 1.0
                            self.gmm_obj[word][f]['gmm']  = copy.deepcopy(self.gmm_M[f])
                    else:
                        for f in self.gmm_M:
                            gmm_N   = dict(self.gmm_obj[word][f]['gmm'])
                            gmm_M   = dict(self.gmm_M[f])
                            N       = self.gmm_obj[word][f]['N']
                            M       = 1.0
                            #============================#
                            # with compition between gmms
                            accepted_ind = []
                            smallest_matrix = np.zeros((len(gmm_N),len(gmm_M)),dtype=np.float)
                            for i in gmm_N:
                                mean_i = gmm_N[i]['mean']
                                covar_i = gmm_N[i]['covar']
                                m_min = np.inf
                                m_ind = []
                                for j in gmm_M:
                                    mean_j = gmm_M[j]['mean']
                                    dis = self._distance_test(mean_i, mean_j)
                                    smallest_matrix[i,j] = dis
                                    #dis1 = self.Mean_Test(mean_j,mean_i,covar_i)
                                    #smallest_matrix[i,j] = dis1
                            for k in range(np.min((len(gmm_N),len(gmm_M)))):
                                #print smallest_matrix
                                i,j = np.unravel_index(smallest_matrix.argmin(), smallest_matrix.shape)
                                m_max = smallest_matrix[i,j]
                                #print m_max
                                if m_max < self.pass_distance:               # == 75% matching in the t-test
                                    # - .04*(1-(1.0/np.exp(15.0/N)))
                                    gmm_N = self._update_gmm(gmm_N, gmm_M, i, j, N, M, 1)
                                    if j not in accepted_ind:   accepted_ind.append(j)
                                    gmm_N[i]['N'] += 1.0
                                    smallest_matrix[i,:] = +np.inf  # row
                                    smallest_matrix[:,j] = +np.inf  # col
                            #print accepted_ind
                            """
                            #============================#
                            # no competition between the different gmms
                            accepted_ind = []
                            for i in gmm_N:
                                mean_i = gmm_N[i]['mean']
                                m_min = np.inf
                                m_ind = []
                                for j in gmm_M:
                                    mean_j = gmm_M[j]['mean']
                                    dis = np.sqrt(np.sum((mean_i-mean_j)**2))
                                    #print f,i,j,dis
                                    if dis < m_min:
                                        m_min = dis
                                        m_ind = j
                                if m_min < .12:               # == 88% matching in the t-test
                                    gmm_N = self._update_gmm(gmm_N, gmm_M, i, m_ind, N, M, 1)
                                    if m_ind not in accepted_ind:   accepted_ind.append(m_ind)
                                    gmm_N[i]['N'] += 1.0
                            """
                            for key in gmm_M.keys():
                                if key not in accepted_ind:
                                    # we have to add this gmm to the hypotheses list with N = 1
                                    gmm_N = self._add_gmm(gmm_N, gmm_M, key)
                            #print 'new clusters',len(accepted_ind)-len(gmm_M.means_)
                            self.gmm_obj[word][f]['gmm'] = copy.deepcopy(gmm_N)
                            self.gmm_obj[word][f]['N'] += M

    #--------------------------------------------------------------------------------------------------------#
    # 3.2 Testing for equality to a mean vector
    # To Do sub function for _build_obj_hyp_igmm
    def Mean_Test(self, x_mean, mean, S):
        # compute sample mean
        d = len(x_mean)
        n = 10
        # compute the sample covariance
        S_inv = np.linalg.inv(S)
        # computing the T squared test
        c1 = np.transpose([mean - x_mean])
        T = n*np.dot(np.dot(np.transpose(c1),S_inv),c1)
        F = T[0][0]*float(n-d)/float(d*(n-1))
        #alpha = .005 #Or whatever you want your alpha to be.
        p_value = scipy.stats.f.pdf(F, d, n-d)
        #result = 0
        #if p_value>alpha:
            #result = 1.0
    	return p_value

    #--------------------------------------------------------------------------------------------------------#
    # sub funtion to compute counting probability
    def _probability(self,count,value):
        P = (value/count)*(1.0/np.exp(self.p_parameter/count))
        return P

    #--------------------------------------------------------------------------------------------------------#
    # 3.3.1 Merging Components
    def _update_gmm(self,gmm1, gmm2, j, k, N, M, Mk):
        mu_j = gmm1[j]['mean']
        S_j = gmm1[j]['covar']
        pi_j = 1
        mu_k = gmm2[k]['mean']
        S_k = gmm2[k]['covar']
        pi_k = 1
        # update the mean
        mu = ( N*pi_j*mu_j + Mk*mu_k )/( N*pi_j + Mk )
        # update the covariance matrix
        mu = np.array(mu)[np.newaxis]
        mu_j = np.array(mu_j)[np.newaxis]
        mu_k = np.array(mu_k)[np.newaxis]
        S = ((N*pi_j*S_j + Mk*S_k)/(N*pi_j+Mk)) + ((N*pi_j*np.dot(mu_j.T,mu_j)+Mk*np.dot(mu_k.T,mu_k))/(N*pi_j+Mk)) - np.dot(mu.T,mu)
        # update the weight
        pi = 1
        # update gmm1
        gmm1[j]['mean'] = mu[0]
        gmm1[j]['covar'] = S
        return gmm1

    #--------------------------------------------------------------------------------------------------------#
    # 3.3.1 Adding Components
    def _add_gmm(self, gmm1, gmm2, i):
            new_key = np.max(gmm1.keys())+1
            gmm1[new_key] = gmm2[i]
            return gmm1

    #--------------------------------------------------------------------------------------------------------#
    def _build_relation_hyp(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                not_ok = 1
                for w in word.split(' '):
                    if w in ['top','over','on','above','placed','front','sitting','onto','underneath','infront','besides','resting','sit','sits','topmost','ontop','sat','stood','higher','downwords']:
                        not_ok = 0
                if word not in self.hyp_relation:
                    self.hyp_relation[word] = {}
                    self.hyp_relation[word]['count'] = 0
                    self.hyp_relation[word]['direction'] = {}
                if self.unique_direction != []:         self.hyp_relation[word]['count'] += 1
                for direction in self.unique_direction:
                    if direction not in self.hyp_relation[word]['direction']:   self.hyp_relation[word]['direction'][direction] = 1
                    else: self.hyp_relation[word]['direction'][direction] += 1

    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this one has a hack, please make sure to clear it.
    def _build_motion_hyp(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                not_ok = 1
                for w in word.split(' '):
                    if w in ['pick','place','put','up','down','move','put','shift','drop','take','remove']:
                        not_ok = 0
                if word not in self.hyp_motion:
                    self.hyp_motion[word] = {}
                    self.hyp_motion[word]['count'] = 0
                    self.hyp_motion[word]['motion'] = {}
                if not_ok:              continue
                if self.unique_motions != []:         self.hyp_motion[word]['count'] += 1
                for motion in self.unique_motions:
                    if motion not in self.hyp_motion[word]['motion']:
                        self.hyp_motion[word]['motion'][motion] = 1
                    else: self.hyp_motion[word]['motion'][motion] += 1

    #--------------------------------------------------------------------------------------------------------#
    def _save_all_features(self):
        self.all_scene_features[self.scene] = {}
        self.all_scene_features[self.scene]['color'] = self.unique_colors
        self.all_scene_features[self.scene]['shapes'] = self.unique_shapes
        self.all_scene_features[self.scene]['direction'] = self.unique_direction

    #--------------------------------------------------------------------------------------------------------#
    def _test_relation_hyp(self):
        self.hyp_relation_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    count = float(self.hyp_relation[word]['count'])
                    for j in self.hyp_relation[word]:
                        if j != 'count':
                            for k in self.hyp_relation[word][j]:
                                prob = self._probability(count,self.hyp_relation[word][j][k])
                                if prob>self.p_relation_pass:
                                    if word not in self.hyp_relation_pass:
                                        self.hyp_relation_pass[word] = {}
                                        self.hyp_relation_pass[word]['possibilities'] = 0
                                    if j not in self.hyp_relation_pass[word]:
                                        self.hyp_relation_pass[word][j] = {}
                                    self.hyp_relation_pass[word]['possibilities'] += 1
                                    self.hyp_relation_pass[word][j][k] = prob

    #--------------------------------------------------------------------------------------------------------#
    def _test_motion_hyp(self):
        self.hyp_motion_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    count = float(self.hyp_motion[word]['count'])
                    for j in self.hyp_motion[word]:
                        if j != 'count':
                            for k in self.hyp_motion[word][j]:
                                prob = self._probability(count,self.hyp_motion[word][j][k])
                                if prob>self.p_relation_pass:
                                    if word not in self.hyp_motion_pass:
                                        self.hyp_motion_pass[word] = {}
                                        self.hyp_motion_pass[word]['possibilities'] = 0
                                    if j not in self.hyp_motion_pass[word]:
                                        self.hyp_motion_pass[word][j] = {}
                                    self.hyp_motion_pass[word]['possibilities'] += 1
                                    self.hyp_motion_pass[word][j][k] = prob

    #--------------------------------------------------------------------------------------------------------#
    def _test_obj_hyp(self):
        hyp_language_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    for f in self.gmm_obj[word]:
                        N = self.gmm_obj[word][f]['N']
                        for i in self.gmm_obj[word][f]['gmm']:
                            value = self.gmm_obj[word][f]['gmm'][i]['mean']
                            count = self.gmm_obj[word][f]['gmm'][i]['N']
                            p = self._probability(N,count)
                            if p >self.p_obj_pass:
                                #print f,'>>>',word,value,p
                                if word not in hyp_language_pass:
                                    hyp_language_pass[word] = {}
                                    hyp_language_pass[word]['possibilities'] = 1
                                else:
                                    hyp_language_pass[word]['possibilities'] += 1
                                if f not in hyp_language_pass[word]:
                                    hyp_language_pass[word][f] = {}
                                hyp_language_pass[word][f][tuple(value)] = p
        #for word in hyp_language_pass:
        self.hyp_obj_pass = copy.deepcopy(hyp_language_pass)

    #--------------------------------------------------------------------------------------------------------#
    def _combine_language_hyp(self):
        ### this is just a test
        self.hyp_language_pass = {}
        self.hyp_all_features = []
        self._combine_hyp(self.hyp_language_pass,self.hyp_obj_pass)
        self._combine_hyp(self.hyp_language_pass,self.hyp_relation_pass)
        self._combine_hyp(self.hyp_language_pass,self.hyp_motion_pass)


    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this will pass all hypotheses that are .9 of maximum value
    def _combine_hyp(self,A,B):
        ### adding two hypotheses A = A+B

        for word in B:
            if word not in A:
                A[word] = {}
                A[word]['possibilities'] = 0
            if word in A:
                for f in B[word]:
                    if f != 'possibilities':
                        if f not in self.hyp_all_features:      self.hyp_all_features.append(f)
                        if len(B[word][f]) > 1:
                            C = copy.deepcopy(B[word][f])
                            maxval = max(C.iteritems(), key=operator.itemgetter(1))[1]
                            keys = [k for k,v in C.items() if v>.9*maxval]
                            A[word][f] = {}
                            for key in keys:
                                A[word]['possibilities'] += 1
                                A[word][f][key] = C[key]
                        else:
                            A[word][f] = copy.deepcopy(B[word][f])
                            A[word]['possibilities'] += 1

    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this will pass all hypotheses that are .9 of maximum value expect location
    def _filter_hyp(self):
        for word in self.hyp_language_pass:
            max_hyp = 0
            keys_remove = []
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities':
                    # find the max hypotheses
                    for key in self.hyp_language_pass[word][f].keys():
                        if self.hyp_language_pass[word][f][key]>max_hyp:            max_hyp = self.hyp_language_pass[word][f][key]

            # find all hypotheses that are within .9 of maximum hyp
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities' and f!= 'location':
                    for key in self.hyp_language_pass[word][f].keys():
                        if self.hyp_language_pass[word][f][key]<.9*max_hyp:         keys_remove.append([f,key])
            for A in keys_remove:
                f=A[0]
                key=A[1]
                self._remove_phrase(word, f, key)


    #--------------------------------------------------------------------------------------------------------#
    # remove sub phrases and bigger phrases based on their meanings
    # NOTE: this will remove sub phrases
    def _filter_phrases(self):
        phrases_to_remove = {}                                                     # this contains the list of phrases that are already described in smaller phrases
        hyp = self.hyp_language_pass
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    if word in hyp:
                        words = self._get_phrases(word)
                        for i in words:
                            for feature in hyp[word]:
                                if feature == 'possibilities': continue
                                for p1 in hyp[word][feature]:
                                    score = []
                                    matching = {}
                                    for sub_phrase in words[i]:
                                        score.append(0)
                                        if sub_phrase in hyp:
                                            if feature in hyp[sub_phrase]:
                                                for p2 in hyp[sub_phrase][feature]:
                                                    m1 = np.asarray(list(p1))
                                                    m2 = np.asarray(list(p2))
                                                    if len(m1) != len(m2):          continue        # motions !
                                                    if self._distance_test(m1,m2)<self.pass_distance_phrases:
                                                        score[-1] = 1
                                                        matching[sub_phrase] = p2
                                    N = float(np.sum(score))/float(len(words[i]))
                                    #print N
                                    # case 1 if N == 1 it means I should remove sub phrases
                                    # case 2 if N == 0 I should keep everything
                                    # case 3 if N < 1  I should remove phrase
                                    if N == 1:
                                        for key in matching:
                                            pass
                                             #if key not in phrases_to_remove:            phrases_to_remove[key] = {}
                                             #if feature not in phrases_to_remove[key]:   phrases_to_remove[key][feature] = []
                                             #if matching[key] not in phrases_to_remove[key][feature]:
                                            #    phrases_to_remove[key][feature].append(matching[key])
                                    elif N > 0:
                                        #print '##########################################################################',word,p1
                                        if word not in phrases_to_remove:               phrases_to_remove[word] = {}
                                        if feature not in phrases_to_remove[word]:      phrases_to_remove[word][feature] = []
                                        if p1 not in phrases_to_remove[word][feature]:
                                            phrases_to_remove[word][feature].append(p1)
        #print phrases_to_remove.keys()
        for word in phrases_to_remove:
            for feature in phrases_to_remove[word]:
                for key in phrases_to_remove[word][feature]:
                    self._remove_phrase(word, feature, key)

    #--------------------------------------------------------------------------------------------------------#
    # sub function
    def _remove_phrase(self, p, f, p_remove):
        # p is the word
        # f is the feature
        # p_remove is the key in the feature in the word
        # remove a phrase from self.hyp_language_pass
        self.hyp_language_pass[p][f].pop(p_remove, None)
        if len(self.hyp_language_pass[p][f].keys()) == 0:
            self.hyp_language_pass[p].pop(f,None)
        self.hyp_language_pass[p]['possibilities'] -= 1
        if self.hyp_language_pass[p]['possibilities'] == 0:
            self.hyp_language_pass.pop(p,None)

    #--------------------------------------------------------------------------------------------------------#
    def _distance_test(self,m1,m2):
        return np.sqrt(np.sum((m1-m2)**2))/np.sqrt(len(m1))

    #--------------------------------------------------------------------------------------------------------#
    # sub function to get all possible phrases that can make a certain phrase
    def _get_phrases(self,word):
        # get all possible combination of sub phrases for a phrase
        words = {}
        w = word.split(' ')
        n = len(w)
        if n == 2:
            words[0] = [w[0],w[1]]
        elif n == 3:
            words[0] = [w[0],w[1],w[2]]
            words[1] = [' '.join(w[0:2]),w[2]]
            words[2] = [w[0],' '.join(w[1:3])]
        return words

    #--------------------------------------------------------------------------------------------------------#
    # generate all possible combinitaiton for each phrase in a single list
    def _get_all_valid_combinations(self):
        for word in self.hyp_language_pass:
            self.hyp_language_pass[word]['all'] = []
            for f in self.hyp_language_pass[word]:
                if f != 'all' and f != 'possibilities':
                    for k in self.hyp_language_pass[word][f]:
                        self.hyp_language_pass[word]['all'].append((f,k))

    #--------------------------------------------------------------------------------------------------------#
    # generate all possible sentences that have hypotheses in language hypotheses pass
    def _test_all_valid_combinations(self):
        # test the hypotheses sentence by sentence
        if 'motion' in self.hyp_all_features:
            self._get_indices()
            for scene in self.phrases:
                #print scene
                # get the words that have hypotheses and are in the sentence
                phrases_with_hyp = list(set(self.hyp_language_pass.keys()).intersection(self.phrases[scene]))

                # generate all subsets (pick from 1 word to n words) with no repatetion in phrases
                for L in range(2, len(phrases_with_hyp)+1):
                    out1 = zip(*self.pool.map(calc, [[subset,scene,self.indices,self.hyp_language_pass,self.all_total_motion[self.scene],self.S] for subset in itertools.combinations(phrases_with_hyp, L)]))


    #------------------------------------------------------------------#
    # get the index of every phrase in every sentence
    def _get_indices(self):
        self.indices = {}
        for scene in self.phrases:
            self.indices[scene] = {}
            sentence = self.S[scene].split(' ')
            # build the indices
            phrases_with_hyp = list(set(self.hyp_language_pass.keys()).intersection(self.phrases[scene]))
            for word in phrases_with_hyp:
                w = word.split(' ')
                if len(w) == 1:
                    A = [[i] for i, x in enumerate(sentence) if x == word]
                    self.indices[scene][word] = A
                if len(w) == 2:
                    ind = {}
                    for wc,w1 in enumerate(w):
                        ind[wc] = []
                        [ind[wc].append(i) for i, x in enumerate(sentence) if x == w1]
                    for i1 in ind[0]:
                        for i2 in ind[1]:
                            if i2-i1 == 1:
                                if word not in self.indices[scene]:
                                    self.indices[scene][word] = []
                                self.indices[scene][word].append([i1,i2])
                if len(w) == 3:
                    ind = {}
                    for wc,w1 in enumerate(w):
                        ind[wc] = []
                        [ind[wc].append(i) for i, x in enumerate(sentence) if x == w1]
                    for i1 in ind[0]:
                        for i2 in ind[1]:
                            for i3 in ind[2]:
                                if i2-i1 == 1 and i3-i2 == 1:
                                    if word not in self.indices[scene]:
                                        self.indices[scene][word] = []
                                    self.indices[scene][word].append([i1,i2,i3])
        #for s in self.indices:
        #    print s,self.indices[s]

    #------------------------------------------------------------------#
    def _test(self,subset,scene):

        # no 2 phrases are allowed to intersect in the same sentence
        no_intersection = 1
        all_indeces = []
        for w in subset:
            #print w
            for w1 in self.indices[scene][w]:
                for w2 in w1:
                    if w2 not in all_indeces:
                        all_indeces.append(w2)
                    else:
                        no_intersection = 0
        #print no_intersection
        #print '------'

        if no_intersection:
            # this function tests one susbet of words at a time
            all_possibilities = []      # all the possibilities gathered in one list
            for word in subset:
                all_possibilities.append(self.hyp_language_pass[word]['all'])

            #print all_possibilities
            #print '---------------',len(all_possibilities)
            # find the actual possibilities for every word in the subset
            #c = 1
            for element in itertools.product(*all_possibilities):
                #print c
                #c+=1
                hyp_motion = {}
                motion_pass = 0

                # no 2 words are allowed to mean the same thing
                not_same = 1
                features = {}
                for i in element:
                    if i[0] not in features: features[i[0]] = []
                    features[i[0]].append(i[1])

                for f in features:
                    if len(features[f])>1:
                        for f1 in range(len(features[f])-1):
                            for f2 in range(f1+1,len(features[f])):
                                m1 = np.asarray(list(features[f][f1]))
                                m2 = np.asarray(list(features[f][f2]))
                                if len(m1) != len(m2):          continue        # motions !
                                if self._distance_test(m1,m2)<self.pass_distance:
                                    not_same = 0
                                    continue
                if not_same:
                    # 1) does actions match ?   it should match 100%
                    for k,word in enumerate(subset):
                        if element[k][0] == 'motion':
                            a = element[k][1]
                            if a not in hyp_motion:     hyp_motion[a] = len(self.indices[scene][word])
                            else:                       hyp_motion[a] += len(self.indices[scene][word])
                    for i in self.total_motion:
                        if self.total_motion[i] == hyp_motion:
                            motion_pass = 1
                        #else: print self.scene,'fail'

                    # 2) parse the sentence
                    if motion_pass:
                        parsed_sentence = []
                        value_sentence = []
                        for i in self.S[scene].split(' '):
                            parsed_sentence.append('_')
                            value_sentence.append('_')
                        for word1 in subset:
                            for i1 in self.indices[scene][word1]:
                                for j1 in i1:
                                    #print word1,j1
                                    k = subset.index(word1)
                                    parsed_sentence[j1] = element[k][0]
                                    value_sentence[j1] = map(prettyfloat, element[k][1])
                        print subset
                        print self.S[scene].split(' ')
                        print parsed_sentence
                        print value_sentence
                        print '----'

    #--------------------------------------------------------------------------------------------------------#
    def _build_parser(self):
        self._update_terminals()
        self._update_nonterminals()
        self._build_PCFG()

    #--------------------------------------------------------------------------------------------------------#
    def _update_terminals(self):
        hypotheses = self.hyp_language_pass
        self.grammar = ''
        self.T = {}                                             # terminals
        self.T['features'] = {}
        self.T['sum'] = {}
        for word in hypotheses:
            for feature in hypotheses[word]:
                if feature not in ['possibilities','all']:
                    if feature not in self.T['features']:
                        self.T['features'][feature] = []
                        self.T['sum'][feature] = 0
                    for hyp in hypotheses[word][feature]:
                        self.T['features'][feature].append((word,hyp[1]))
                        self.T['sum'][feature] += hyp[1]

        for feature in self.T['features']:
            l = len(self.T['features'][feature])
            for hyp in self.T['features'][feature]:
                self.grammar += feature+" -> '"+hyp[0]+"' ["+str(hyp[1]/self.T['sum'][feature])+"]"+'\n'

    #--------------------------------------------------------------------------------------------------------#
    def _update_nonterminals(self,):
        # there is a threshold in NLTK to drop a hypotheses 9.99500249875e-05 I think it;s 1e-4

        entity_features = ['color','shape','location']

        def rotate(l,n):
            return l[n:] + l[:n]

        def _find_entities(A):
            all_e = []
            e = []
            k_1 = 0
            for k in A:
                if e == []:
                    e.append(k)
                    k_1 = k
                elif k-k_1 == 1:
                    e.append(k)
                    k_1 = k
                else:
                    all_e.append(e)
                    e = [k]
                    k_1 = k
            all_e.append(e)
            return all_e

        # loop through all sentences
        features = self.T['features'].keys()
        for s in self.S:
            sentence = self.S[s].split(' ')
            indices = {}
            for feature in features:
                indices[feature] = []
                for hyp in self.T['features'][feature]:
                    A = [i for i, x in enumerate(sentence) if x == hyp[0]]
                    for i in A:
                        indices[feature].append(i)

            # plug in the hypotheses of each word
            parsed_sentence = []
            entity_sentence = []                        #to check connectivity of an entity
            for ind,word in enumerate(sentence):
                parsed_sentence.append('_')
                entity_sentence.append(0)
                for feature in indices:
                    if ind in indices[feature]:
                        parsed_sentence[ind] = feature
                        if feature in entity_features:
                            entity_sentence[ind] = 1

            # find the number of entity based on connectivity of features in a sentence
            A = [i for i, x in enumerate(entity_sentence) if x == 1]
            all_entities = _find_entities(A)

            #print sentence
            #print self.scene,parsed_sentence
            #print self.scene,entity_sentence

            # update the non-terminal counter
            for entity in all_entities:
                if entity != []:
                    h = ()
                    for j in entity:
                        h += (parsed_sentence[j],)
                    if h not in self.N['e']:        self.N['e'][h] =  1.0
                    else:        self.N['e'][h] += 1.0
                    self.N['e']['sum'] += 1.0

        # add entities, ++ to grammer
        for feature in self.N:
            sorted_f = sorted(self.N[feature].items(), key=operator.itemgetter(1))
            for l in range(len(sorted_f),0,-1):
                hyp = sorted_f[l-1]
                if hyp[0] != 'sum':
                    val = hyp[1]/self.N[feature]['sum']
                    if val > 1e-4:
                        small_msg = ''
                        if len(hyp[0]) == 1:
                            small_msg += hyp[0][0]
                        else:
                            for j in range(len(hyp[0])-1):
                                small_msg += hyp[0][j]+' '
                            small_msg += hyp[0][j+1]
                        self.grammar += feature+" -> '"+small_msg+"' ["+str(val)+"]"+'\n'

    #--------------------------------------------------------------------------------------------------------#
    def _build_PCFG(self):
        if self.grammar != '':
            self.pcfg1 = PCFG.fromstring(self.grammar)
            print self.pcfg1

    #--------------------------------------------------------------------------------------------------------#
    def _print_results(self):
        print '====-----------------------------------------------------------------===='
        for word in self.hyp_language_pass:
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities' and f != 'all':
                    for value in self.hyp_language_pass[word][f]:
                        if len(f) < 7:
                            print f,'\t\t>>>\t',
                        elif len(f) < 15:
                            print f,'\t>>>\t',
                        if len(word) < 7:
                            print word,'\t\t\t>>>\t',
                        elif len(word) < 15:
                            print word,'\t\t>>>\t',
                        elif len(word) < 23:
                            print word,'\t>>>\t',
                        print map(prettyfloat, value),
                        print("{0:.3f}".format( self.hyp_language_pass[word][f][value]))




    #--------------------------------------------------------------------------------------------------------#
    def _plot_graphs(self):
        self.f,self.ax = plt.subplots(len(self.transition['all']),4,figsize=(14,10)) # first col motion , second distance
        self.f.suptitle('Scene : '+str(self.scene), fontsize=20)
        for feature in [0,2]:
            # plot the different graphs of motion and distance
            for sub,T in enumerate(self.transition['all']):
                plt.sca(self.ax[sub,feature])
                print 'plotting graph : '+str(sub+1)+' from '+str(len(self.transition['all']))
                if feature == 0:
                    if T not in self.transition['motion']:
                        for i in self.transition['motion']:
                            if i<T: t=i
                    else: t=T
                    G=self.G_motion[t]['graph']
                elif feature == 2:
                    if T not in self.transition['touch']:
                        for i in self.transition['touch']:
                            if i<T: t=i
                    else: t=T
                    G=self.G_touch[t]['graph']
                # layout graphs with positions using graphviz neato
                pos=nx.graphviz_layout(G,prog="neato")
                # color nodes the same in each connected subgraph
                C=nx.connected_component_subgraphs(G)
                cK = 0
                for i in C:  cK += 1
                C=nx.connected_component_subgraphs(G)
                colors = np.linspace(.2,.6,cK)
                for count,g in enumerate(C):
                    c=[colors[count]]*nx.number_of_nodes(g) # same color...
                    nx.draw(g,pos,node_size=80,node_color=c,vmin=0.0,vmax=1.0,with_labels=False)
                    #nx.draw_networkx_edges(g,pos, with_labels=False, edge_color=c[0], width=6.0, alpha=0.5)
                nx.draw_networkx_nodes(self.G,pos, node_color='b', node_size=100, alpha=1)
                nx.draw_networkx_nodes(self.G,pos, nodelist=['G'], node_color='r', node_size=100, alpha=1)
                nx.draw_networkx_nodes(self.G,pos, nodelist=[str(self.m_obj)], node_color='c', node_size=100, alpha=1)
                nx.draw_networkx_edges(G,pos, alpha=0.8)
                #nx.draw(G)  # networkx draw()
                self.ax[sub,feature].axis('on')
                self.ax[sub,feature].axis('equal')
                plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
                if feature == 0:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('motion')
                if feature == 2:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('connectivity')

        #plt.draw() # display
        #plt.pause(.00001)
        #plt.show()

    #--------------------------------------------------------------------------------------------------------#
    def _create_moving_obj_graph(self,T):
            # Creating the graph structure
            G = nx.Graph()
            # creating the object layer
            m_count = 2.0                     # moving object location
            r_count = m_count+.5                 # 1 is reserved for the moving object
            obj_count = 3.0                 # 1 is reserved for the moving object
            G_count = 1.0
            for key in self.keys:
                if key == self.m_obj:
                    G.add_node(str(key),type1='mo',position=(m_count,3))
                    G.add_node(str(key)+'_c',type1='of',position=(m_count-.25,1));         #color
                    G.add_node(str(key)+'_s',type1='of',position=(m_count,1));         #shape
                    G.add_node(str(key)+'_l',type1='of',position=(m_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_c')
                    G.add_edge(str(key),str(key)+'_s')
                    G.add_edge(str(key),str(key)+'_l')
                elif key == 'G':
                    G.add_node(str(key),type1='G',position=(G_count,3))
                else:
                    G.add_node(str(key),type1='o',position=(obj_count,3))
                    G.add_node(str(key)+'_c',type1='of',position=(obj_count-.25,1));         #color
                    G.add_node(str(key)+'_s',type1='of',position=(obj_count,1));         #shape
                    G.add_node(str(key)+'_l',type1='of',position=(obj_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_c')
                    G.add_edge(str(key),str(key)+'_s')
                    G.add_edge(str(key),str(key)+'_l')
                    obj_count+=1

            # creating the relation layer

            k1 = self.m_obj
            counter = 0
            for k2 in self.keys:
                if k2 != k1 and k2 != 'G':
                    G.add_node(str(k1)+'_'+str(k2),type1='r',position=(r_count,7.0))     # it's a directed node from k1 to k2
                    G.add_edge(str(k1)+'_'+str(k2),str(k1))
                    G.add_edge(str(k1)+'_'+str(k2),str(k2))
                    G.add_node(str(k1)+'_'+str(k2)+'_dist',type1='rf',position=(r_count,5));         #distance
                    #direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                    G.add_node(str(k1)+'_'+str(k2)+'_dir',type1='rf',position=(r_count-.15,5));                     #direction
                    G.add_node(str(k1)+'_'+str(k2)+'_mot',type1='rf',position=(r_count+.15,5));                     #motion
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
                    counter += 1
                    r_count += 1

                if k2 != k1 and k2 == 'G':
                    G.add_node(str(k1)+'_'+str(k2),type1='r',position=(G_count+.5,7.0))     # it's a directed node from k1 to k2
                    G.add_edge(str(k1)+'_'+str(k2),str(k1))
                    G.add_edge(str(k1)+'_'+str(k2),str(k2))
                    G.add_node(str(k1)+'_'+str(k2)+'_dist',type1='rf',position=(G_count+.5,5));         #distance
                    #direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                    G.add_node(str(k1)+'_'+str(k2)+'_dir',type1='rf',position=(G_count+.5-.15,5));                     #direction
                    G.add_node(str(k1)+'_'+str(k2)+'_mot',type1='rf',position=(G_count+.5+.15,5));                     #motion
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                    G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
            return G

    #--------------------------------------------------------------------------------------------------------#
    def _plot_final_graph(self):
        for feature in [1,3]:
            for sub,T in enumerate(self.transition['all']):
                plt.sca(self.ax[sub,feature])

                G = self._create_moving_obj_graph(T)
                # Creating the group effect
                if feature == 1:
                    if T not in self.transition['motion']:
                        for i in self.transition['motion']:
                            if i<T: t=i
                    else: t=T
                    G_group = self.G_motion[t]['groups']
                if feature == 3:
                    if T not in self.transition['touch']:
                        for i in self.transition['touch']:
                            if i<T: t=i
                    else: t=T
                    G_group = self.G_touch[t]['groups']

                N = len(G_group)
                HSV = [(x*.9/N, 0.9, 0.9) for x in range(N)]
                RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV)
                for c,group in enumerate(G_group):
                    for node in group:
                        p = G.node[node]['position']
                        rect1 = mat.patches.Rectangle((p[0]-.4,p[1]-3), .8, 4, color=RGB[c],alpha=.5)
                        self.ax[sub,feature].add_patch(rect1)

                agents = list((n for n in G if G.node[n]['type1']=='G'))
                m_objects = list((n for n in G if G.node[n]['type1']=='mo'))
                objects = list((n for n in G if G.node[n]['type1']=='o'))
                objects_f = list((n for n in G if G.node[n]['type1']=='of'))
                relations = list((n for n in G if G.node[n]['type1']=='r'))
                relations_f = list((n for n in G if G.node[n]['type1']=='rf'))

                #pos=nx.graphviz_layout(G,prog="neato")
                pos = nx.get_node_attributes(G,'position')

                nx.draw_networkx_nodes(G,pos, nodelist=agents, node_color='r', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=m_objects, node_color='c', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=objects, node_color='b', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=objects_f, node_color='y', node_size=100, alpha=0.8)
                nx.draw_networkx_nodes(G,pos, nodelist=relations, node_color='g', node_size=100, alpha=0.8)
                nx.draw_networkx_nodes(G,pos, nodelist=relations_f, node_color='y', node_size=100, alpha=0.8)
                #nx.draw_networkx_edges(G,pos, with_labels=False, edge_color='r', width=6.0, alpha=0.5)
                nx.draw_networkx_edges(G,pos, alpha=0.8)

                if feature == 1:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('motion')

                if feature == 3:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('connectivity')

                self.ax[sub,2].axis('on')
                plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')



        plt.savefig(self.dir3+'_'+str(self.scene)+'.png',dpi=200) # save as png
        plt.show() # display

    #--------------------------------------------------------------------------------------------------------#
    def _plot_scene(self):
        import sys, select, os
        all_files = sorted(listdir(self.dir2+str(self.scene)+'/'))
        for i in all_files:
            img = cv2.imread(self.dir2+str(self.scene)+'/'+i)
            cv2.imshow('scene',img)
            cv2.waitKey(50) & 0xff
        plt.close(self.f)
