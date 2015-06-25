import numpy as np
import networkx as nx
import itertools
from nltk import PCFG
import operator

class process_data():
    def __init__(self):
        self.dir1 = '/home/omari/Datasets/robot_modified/motion/scene'
        self.dir2 = '/home/omari/Datasets/robot_modified/scenes/'
        self.dir3 = '/home/omari/Datasets/robot_modified/graphs/scene'
        self.hyp_final = {}
        self.hyp_final['action'] = {}
        self.hyp_final['color'] = {}
        self.hyp_final['shape'] = {}
        self.hyp_final['location'] = {}
        self.hyp_final['direction'] = {}
        self.hyp_final['?'] = {}
        
        self.hyp = {}
        self.hyp['action'] = {}
        self.hyp['color'] = {}
        self.hyp['shape'] = {}
        self.hyp['location'] = {}
        self.hyp['direction'] = {}
        self.hyp['?'] = {}
        
        # initial language
        self.N = {}                                             # non-terminals
        self.N['e'] = {}                                        # non-terminals entity
        self.N['e']['sum'] = 0.0                                # non-terminals entity counter
        
        self.hyp_language = {}
        
        self.n_word = 1
        self.step = 3
        self.all_words = []
        #plt.ion()
        
    #--------------------------------------------------------------------------------------------------------#
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
    def _fix_sentences(self):
        for i in self.S:
            self.S[i] = self.S[i].replace("  ", " ")            
            self.S[i] = self.S[i].replace(".", "")
            
    #--------------------------------------------------------------------------------------------------------#
    def _more_fix_sentences(self):
        for i in self.S:
            self.S[i] = self.S[i].replace("-", " ") 
            self.S[i] = self.S[i].replace("/", " ") 
            self.S[i] = self.S[i].replace("!", "")  
            self.S[i] = self.S[i].replace("(", "")            
            self.S[i] = self.S[i].replace(")", "")             
            self.S[i] = self.S[i].replace("?", "")    
            
    #--------------------------------------------------------------------------------------------------------#
    def _print_scentenses(self):
        for count,i in enumerate(self.S):
            print count,'-',self.S[i]
        print '--------------------------'
        
    #--------------------------------------------------------------------------------------------------------#
    def _fix_data(self):
        # correction to Data removing 20 and 40
        for i in self.Data:
            self.Data[i]['x'] = np.delete(self.Data[i]['x'],[self.step,2*self.step])
            self.Data[i]['y'] = np.delete(self.Data[i]['y'],[self.step,2*self.step])
            self.Data[i]['z'] = np.delete(self.Data[i]['z'],[self.step,2*self.step])
        
    #--------------------------------------------------------------------------------------------------------#
    def _find_unique_words(self):
        self.phrases = {}                       # hold the entire phrase up to a certain number of words
        self.words = {}                         # hold the list of independent words in a sentence
        # read the sentence
        for s in self.S:
            self.phrases[s] = []
            self.words[s] = []
            sentence = self.S[s]
            w = sentence.split(' ')
            for i in range(len(w)):
                if w[i]not in self.words[s]: self.words[s].append(w[i])
                for j in range(i+1,np.min([i+1+self.n_word,len(w)+1])):
                    self.phrases[s].append(' '.join(w[i:j]))
  
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
                A[A<=1] = 1
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
    def _grouping(self):
        self.G_motion = self._grouping_template(self.transition['motion'],self.motion_all)
        self.G_touch = self._grouping_template(self.transition['touch'],self.touch_all)
            
    #-------------------------------------------------#
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
    
    #--------------------------------------------------------------------------------------------------------#
    def _build_obj_hyp(self):
    
        for s in self.words:
            for word in self.words[s]:
                if word not in self.hyp_language:
                    self.hyp_language[word] = {}
                    self.hyp_language[word]['count'] = 0
                    self.hyp_language[word]['action'] = {}
                    self.hyp_language[word]['color'] = {}
                    self.hyp_language[word]['shape'] = {}
                    self.hyp_language[word]['direction'] = {}
                    self.hyp_language[word]['location'] = {}
                self.hyp_language[word]['count'] += 1
                
                """
                for action in self.unique_motions:
                    if action not in self.hyp_language[word]['action']:   self.hyp_language[word]['action'][action] = 1
                    else: self.hyp_language[word]['action'][action] += 1
                """    
                for color in self.unique_colors:
                    if color not in self.hyp_language[word]['color']:   self.hyp_language[word]['color'][color] = 1
                    else: self.hyp_language[word]['color'][color] += 1
                    
                for shape in self.unique_shapes:
                    if shape not in self.hyp_language[word]['shape']:   self.hyp_language[word]['shape'][shape] = 1
                    else: self.hyp_language[word]['shape'][shape] += 1
                    
                #for direction in self.directions:
                #    if direction not in self.hyp_language[word]['direction']:   self.hyp_language[word]['direction'][direction] = 1
                #    else: self.hyp_language[word]['direction'][direction] += 1
                '''
                for location in self.locations_m_i:
                    if location not in self.hyp_language[word]['location']:   self.hyp_language[word]['location'][location] = 1
                    else: self.hyp_language[word]['location'][location] += 1
                    
                for location in self.locations_m_f:
                    if location not in self.hyp_language[word]['location']:   self.hyp_language[word]['location'][location] = 1
                    else: self.hyp_language[word]['location'][location] += 1
                '''
    #--------------------------------------------------------------------------------------------------------#             
    def _test_language_hyp(self):
        self.hyp_language_pass = {}
        for word in self.hyp_language:
            count = float(self.hyp_language[word]['count'])
            for j in self.hyp_language[word]:
                if j != 'count': 
                    for k in self.hyp_language[word][j]:
                        prob = self._probability(count,self.hyp_language[word][j][k])
                        if prob>.98: 
                            if word not in self.hyp_language_pass: 
                                self.hyp_language_pass[word] = {}
                                self.hyp_language_pass[word]['possibilities'] = 0
                                self.hyp_language_pass[word]['all'] = []
                            if j not in self.hyp_language_pass[word]: 
                                self.hyp_language_pass[word][j] = []
                            self.hyp_language_pass[word]['possibilities'] += 1
                            self.hyp_language_pass[word][j].append((k,prob))
                            self.hyp_language_pass[word]['all'].append((j,k))
                            #print word,count
                            #print j+':',k,prob
                            #print '==---------------------=='
                        
    #--------------------------------------------------------------------------------------------------------#
    def _probability(self,count,value):
        P = (value/count)*(1.0/np.exp(1/count))
        return P
              
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
    def _test_sentence_hyp(self):
        # test the hypotheses sentence by sentence
        for scene in self.words:
            # get the words that have hypotheses and are in the sentence
            words_with_hyp =  list(set(self.hyp_language_pass.keys()).intersection(self.words[scene]))
            # generate all subsets (pick from 1 word to n words)
            for L in range(1, len(words_with_hyp)+1):
                for subset in itertools.combinations(words_with_hyp, L):
                    self._test(subset,scene)
                    #print '==------== subset'
                    #print
            #print '==-------------== sentence'
            #print

    #------------------------------------------------------------------#                     
    def _test(self,subset,scene):
        # this function tests one susbet of words at a time
        sentence = self.S[scene].split(' ')
        all_possibilities = []      # all the possibilities gathered in one list
        for word in subset:
            all_possibilities.append(self.hyp_language_pass[word]['all'])
        # find the actual possibilities for every word in the subset
        for element in itertools.product(*all_possibilities):
            indices = {}
            hyp_motion = {}
            motion_pass = 0
            # build the indices
            for k,word in enumerate(subset):
                indices[word] = [i for i, x in enumerate(sentence) if x == word]
            
            # no 2 words are allowed to mean the same thing
            the_same = 0
            for i in element:
                if element.count(i)>1:          
                    the_same = 1
            if the_same: 
                continue
               
            # 1) does actions match ?   it should match 100%
            for k,word in enumerate(subset):
                if element[k][0] == 'action':
                    a = element[k][1]
                    if a not in hyp_motion:     hyp_motion[a] = len(indices[word])
                    else:                       hyp_motion[a] += len(indices[word])
            for i in self.total_motion:
                if self.total_motion[i] == hyp_motion:
                    motion_pass = 1
                #else: print self.scene,'fail'
                    
            # 2) parse the sentence
            if motion_pass:
                parsed_sentence = []
                value_sentence = []
                for word in sentence:
                    if word in subset:
                        k = subset.index(word)
                        parsed_sentence.append(element[k][0])
                        value_sentence.append(element[k][1])
                    else:               
                        parsed_sentence.append('_')
                        value_sentence.append('_')
                print sentence
                print self.scene,parsed_sentence
                print self.scene,value_sentence
                    
                #print indices,word,element[k]
                #G_test = nx.Graph()
            #print '==--== hyp'
        
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
        
        
