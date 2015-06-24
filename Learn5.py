import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing import *
dir1 = '/home/omari/Datasets/robot/motion/scene'

hyp = {}
for scene in range(1,20):
    O,G,S = read(scene)     #Objects, Graph, Sentences
    
    # correction to Data removing 20 and 40
    #for i in O:
    #    O[i]['x'] = np.delete(O[i]['x'],[20,40])
    #    O[i]['y'] = np.delete(O[i]['y'],[20,40])
    #    O[i]['z'] = np.delete(O[i]['z'],[20,40])
    
    # initial parameter
    frames = len(O['G']['x'])     # we remove the first one to compute speed
    keys = O.keys()
    #n = np.sum(range(len(keys)))
    n = len(keys)-1
        
    # finding the moving object ! fix this
    m_obj = []
    for i in O:
        if i != 'G':
            x = np.abs(O[i]['x'][0]-O[i]['x'][-1])
            y = np.abs(O[i]['y'][0]-O[i]['y'][-1])
            z = np.abs(O[i]['z'][0]-O[i]['z'][-1])
            if (x+y+z) > 0:
                m_obj = i
                
    # computing distance BINARY Distance (touch or no touch)
    dis = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)):
        k2 = keys[i]
        if k2 != m_obj and k2 != 'G':
            dx = np.abs(O[m_obj]['x'][:]-O[k2]['x'][:])
            dy = np.abs(O[m_obj]['y'][:]-O[k2]['y'][:])
            dz = np.abs(O[m_obj]['z'][:]-O[k2]['z'][:])
            A = dx+dy+dz
            A[A<=1] = 1
            A[A>1] = 0
            dis[counter,:] = A
            counter += 1
            
    # computing direction
    dirx = np.zeros((n,frames),dtype=np.int8)
    diry = np.zeros((n,frames),dtype=np.int8)
    dirz = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)):
        k2 = keys[i]
        if k2 != m_obj and k2 != 'G':
            dx = O[m_obj]['x'][:]-O[k2]['x'][:]
            dy = O[m_obj]['y'][:]-O[k2]['y'][:]
            dz = O[m_obj]['z'][:]-O[k2]['z'][:]
            dirx[counter,:] = np.sign(dx).astype(int)
            diry[counter,:] = np.sign(dy).astype(int)
            dirz[counter,:] = np.sign(dz).astype(int)
            counter += 1
    
    # finding locations
    loci = {}
    locf = {}
    for key in keys:
        if key != 'G':
            loci[key] = [O[key]['x'][0],O[key]['y'][0]]
            locf[key] = [O[key]['x'][-1],O[key]['y'][-1]]
            
        
    # plotting the initial and final scene
    f, ax = plt.subplots(2) # first initial , second final
    for sub in range(2):
        plt.sca(ax[sub])
    
        if sub == 0:        #initial
            loc = loci
            distance = 0
        else:
            loc = locf
            distance = -1
    
        # Creating the graph structure
        G = nx.Graph()
        # creating the object layer
        r_count = 1.5                 # 1 is reserved for the moving object
        obj_count = 2.0                 # 1 is reserved for the moving object
        m_count = 1.0                     # moving object location
        for key in keys:
            if key != 'G':
                if key == m_obj:    
                    G.add_node(str(key),type1='mo',position=(m_count,3))
                    G.add_node(str(key)+'_c',value=O[key]['color'],type1='of',position=(m_count-.25,1));         #color
                    G.add_node(str(key)+'_s',value=O[key]['shape'],type1='of',position=(m_count,1));         #shape
                    G.add_node(str(key)+'_l',value=loc[key],type1='of',position=(m_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_c')
                    G.add_edge(str(key),str(key)+'_s')
                    G.add_edge(str(key),str(key)+'_l')
                else:               
                    G.add_node(str(key),type1='o',position=(obj_count,3))
                    G.add_node(str(key)+'_c',value=O[key]['color'],type1='of',position=(obj_count-.25,1));         #color
                    G.add_node(str(key)+'_s',value=O[key]['shape'],type1='of',position=(obj_count,1));         #shape
                    G.add_node(str(key)+'_l',value=loc[key],type1='of',position=(obj_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_c')
                    G.add_edge(str(key),str(key)+'_s')
                    G.add_edge(str(key),str(key)+'_l')
                    obj_count+=1
                    
        ######################################################################################################## for now start and end only ! this should change to as many motions
        # creating the relation layer
        k1 = m_obj
        counter = 0
        for k2 in keys:
            if k2 != k1 and k2 != 'G':
                G.add_node(str(k1)+'_'+str(k2),type1='r',position=(r_count,7.0))     # it's a directed node from k1 to k2
                G.add_edge(str(k1)+'_'+str(k2),str(k1))
                G.add_edge(str(k1)+'_'+str(k2),str(k2))
                G.add_node(str(k1)+'_'+str(k2)+'_dist',value=dis[counter,distance],type1='rf',position=(r_count,5));         #shape
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                counter += 1
                r_count += 1
                
        
        
        m_objects = list((n for n in G if G.node[n]['type1']=='mo'))
        objects = list((n for n in G if G.node[n]['type1']=='o'))
        objects_f = list((n for n in G if G.node[n]['type1']=='of'))
        relations = list((n for n in G if G.node[n]['type1']=='r'))
        relations_f = list((n for n in G if G.node[n]['type1']=='rf'))
        for node in relations_f:
            print G.node[node]
        
        #pos=nx.graphviz_layout(G,prog="neato")
    	pos = nx.get_node_attributes(G,'position')
        
        nx.draw_networkx_nodes(G,pos, nodelist=m_objects, node_color='r', node_size=100, alpha=1)
        nx.draw_networkx_nodes(G,pos, nodelist=objects, node_color='b', node_size=100, alpha=.8)
        nx.draw_networkx_nodes(G,pos, nodelist=objects_f, node_color='c', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G,pos, nodelist=relations, node_color='g', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G,pos, nodelist=relations_f, node_color='gray', node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G,pos, alpha=0.5)

                 
                 
                 
                 
                 
        ax[sub].axis('on')
        plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
        #plt.savefig("node_colormap.png") # save as png
    plt.show() # display

    
    
    
    
    
    
    
    
    
    
    
    
