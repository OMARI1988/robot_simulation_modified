import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
#from data_processing import *
import cv2
from os import listdir
import matplotlib as mat
import colorsys


class Learn():
    def __init__(self):
        self.dir1 = '/home/omari/Datasets/robot/motion/scene'
        self.dir2 = '/home/omari/Datasets/robot/scenes/'
        self.dir3 = '/home/omari/Datasets/robot/graphs/scene'
        #plt.ion()
        
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
            if data[s].split(':')[0] == 'GOOD': self.S[count] = data[s].split(':')[1]
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
    
    def _fix_data(self):
        # correction to Data removing 20 and 40
        for i in self.Data:
            self.Data[i]['x'] = np.delete(self.Data[i]['x'],[20,40])
            self.Data[i]['y'] = np.delete(self.Data[i]['y'],[20,40])
            self.Data[i]['z'] = np.delete(self.Data[i]['z'],[20,40])

            
    def _compute_features_for_all(self):             
        # initial parameter
        self.frames = len(self.Data['G']['x'])-1     # we remove the first one to compute speed
        self.keys = self.Data.keys()
        self.n = np.sum(range(len(self.keys)))  
        # computing distance and touch   (between all objects)
        self.dis_all = np.zeros((self.n,self.frames),dtype=np.float)           
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
                self.dis_all[counter,:] = A
                A[A<=1.0] = 1
                A[A>1.0] = 0
                self.touch_all[counter,:] = A
                counter += 1
                    
        # computing direction   (between all objects)
        self.dirx_all = np.zeros((self.n,self.frames),dtype=np.int8)
        self.diry_all = np.zeros((self.n,self.frames),dtype=np.int8)
        self.dirz_all = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)-1):
            for j in range(i+1,len(self.keys)):
                k1 = self.keys[i]
                k2 = self.keys[j]
                dx = self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:]
                dy = self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:]
                dz = self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:]
                self.dirx_all[counter,:] = np.sign(dx).astype(int)
                self.diry_all[counter,:] = np.sign(dy).astype(int)
                self.dirz_all[counter,:] = np.sign(dz).astype(int)
                counter += 1
                       
        # computing motion      (between all objects)
        for i in self.Data:
            dx = (self.Data[i]['x'][:-1]-self.Data[i]['x'][1:])**2
            dy = (self.Data[i]['y'][:-1]-self.Data[i]['y'][1:])**2
            dz = (self.Data[i]['z'][:-1]-self.Data[i]['z'][1:])**2
            self.Data[i]['motion'] = np.round(np.sqrt(dx+dy+dz)*100)
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
                A[A!=0] = 1
                self.motion_all[counter,:] = A
                counter += 1
                
    def _transition(self):
        # comput the transition intervals for motion
        col = self.motion_all[:,0]
        self.transition = {}
        self.transition['motion'] = [0]
        self.transition['all'] = [0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.motion_all[:,i]))!=0:
                col = self.motion_all[:,i]
                self.transition['motion'].append(i)
                self.transition['all'].append(i)
        # comput the transition intervals for touch
        self.transition['touch'] = [0]
        col = self.touch_all[:,0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.touch_all[:,i]))!=0:
                col = self.touch_all[:,i]
                self.transition['touch'].append(i)
                if i not in self.transition['all']: self.transition['all'].append(i)
        self.transition['all'] = sorted(self.transition['all'])
    
    def _grouping(self):
        self.G_motion = self._grouping_template(self.transition['motion'],self.motion_all)
        self.G_touch = self._grouping_template(self.transition['touch'],self.touch_all)
            
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
        
    def _compute_features_for_moving_object(self):
        # finding the moving object ! fix this
        self.m_obj = []
        for i in self.Data:
            if i != 'G':
                x = np.abs(self.Data[i]['x'][-1]-self.Data[i]['x'][0])
                y = np.abs(self.Data[i]['y'][-1]-self.Data[i]['y'][0])
                z = np.abs(self.Data[i]['z'][-1]-self.Data[i]['z'][0])
                if (x+y+z) > 0:
                    self.m_obj = i
                    
        # computing distance BINARY Distance (touch or no touch)
        self.dis_m = np.zeros((self.n,self.frames),dtype=np.float)
        self.touch_m = np.zeros((self.n,self.frames),dtype=np.uint8)
        counter = 0
        k1 = self.m_obj
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1:
                dx = np.abs(self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:])
                dy = np.abs(self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:])
                dz = np.abs(self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:])
                A = dx+dy+dz
                self.dis_m[counter,:] = A
                A[A<=1] = 1
                A[A>1] = 0
                self.touch_m[counter,:] = A
                counter += 1
                
        # computing direction
        self.dirx_m = np.zeros((self.n,self.frames),dtype=np.int8)
        self.diry_m = np.zeros((self.n,self.frames),dtype=np.int8)
        self.dirz_m = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                dx = self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:]
                dy = self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:]
                dz = self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:]
                self.dirx_m[counter,:] = np.sign(dx).astype(int)
                self.diry_m[counter,:] = np.sign(dy).astype(int)
                self.dirz_m[counter,:] = np.sign(dz).astype(int)
                counter += 1
                
        # finding locations
        self.loc_init = {}
        self.loc_final = {}
        for key in self.keys:
            self.loc_init[key] = [self.Data[key]['x'][0],self.Data[key]['y'][0]]
            self.loc_final[key] = [self.Data[key]['x'][-1],self.Data[key]['y'][-1]]

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

        
    def _plot_scene(self):
        import sys, select, os
        all_files = sorted(listdir(self.dir2+str(self.scene)+'/'))
        for i in all_files:
            img = cv2.imread(self.dir2+str(self.scene)+'/'+i)
            cv2.imshow('scene',img)
            cv2.waitKey(50) & 0xff
        plt.close(self.f)
        
    def _print_scentenses(self):
        for count,i in enumerate(self.S):
            print count,'-',self.S[i]
        #print self.words
        #print len(self.words)
        print '--------------------------'
        
    
def main():
    L = Learn()
    for scene in range(1,20):
        L._read(scene)
        L._print_scentenses()
        L._fix_data()
        L._compute_features_for_all()
        L._transition()                     
        L._grouping()
        L._compute_features_for_moving_object()
        L._plot_graphs()
        L._plot_final_graph()
        #L._plot_scene()
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
