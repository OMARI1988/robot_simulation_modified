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
    for i in O:
        O[i]['x'] = np.delete(O[i]['x'],[20,40])
        O[i]['y'] = np.delete(O[i]['y'],[20,40])
        O[i]['z'] = np.delete(O[i]['z'],[20,40])
    
    # initial parameter
    frames = len(O['G']['x'])-1     # we remove the first one to compute speed
    keys = O.keys()
    n = np.sum(range(len(keys)))
        
    # finding the moving object ! fix this
    m_obj = []
    for i in O:
        if i != 'G':
            x = np.abs(O[i]['x'][-1]-O[i]['x'][0])
            y = np.abs(O[i]['y'][-1]-O[i]['y'][0])
            z = np.abs(O[i]['z'][-1]-O[i]['z'][0])
            if (x+y+z) > 0:
                m_obj = i
           
    # computing distance    (between all objects)
    dis = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            dx = np.abs(O[k1]['x'][1:]-O[k2]['x'][1:])
            dy = np.abs(O[k1]['y'][1:]-O[k2]['y'][1:])
            dz = np.abs(O[k1]['z'][1:]-O[k2]['z'][1:])
            A = dx+dy+dz
            ind1 = np.where(A<=1.0)
            dis[counter,ind1[0]] = 1
            counter += 1
            
    # computing direction   (between all objects)
    dirx = np.zeros((n,frames),dtype=np.int8)
    diry = np.zeros((n,frames),dtype=np.int8)
    dirz = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            dx = O[k1]['x'][1:]-O[k2]['x'][1:]
            dy = O[k1]['y'][1:]-O[k2]['y'][1:]
            dz = O[k1]['z'][1:]-O[k2]['z'][1:]
            dirx[counter,:] = np.sign(dx).astype(int)
            diry[counter,:] = np.sign(dy).astype(int)
            dirz[counter,:] = np.sign(dz).astype(int)
            counter += 1
            
    # computing motion      (between all objects)
    for i in O:
        dx = (O[i]['x'][:-1]-O[i]['x'][1:])**2
        dy = (O[i]['y'][:-1]-O[i]['y'][1:])**2
        dz = (O[i]['z'][:-1]-O[i]['z'][1:])**2
        O[i]['speed'] = np.round(np.sqrt(dx+dy+dz)*100)
        O[i]['speed'][O[i]['speed']!=0] = 1
        
    # compute relative motion
    # similar speed == 1 different == 0
    r_speed = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            for count,s in enumerate(np.abs(O[k1]['speed'] - O[k2]['speed'])):
                if s != 0: a = 0
                elif s == 0: a = 1
                r_speed[counter,count] = a
            counter += 1
    
    # comput the transition intervals for motion
    col = r_speed[:,0]
    transition = [0]
    for i in range(1,frames):
        if np.sum(np.abs(col-r_speed[:,i]))!=0:
            col = r_speed[:,i]
            transition.append(i)
    # comput the transition intervals for distance
    col = dis[:,0]
    for i in range(1,frames):
        if np.sum(np.abs(col-dis[:,i]))!=0:
            col = dis[:,i]
            if i not in transition: transition.append(i)
    transition = sorted(transition)
    
    ####-------------------------------------------------------------------------####
    # computing distance BINARY Distance (touch or no touch)
    dis_m = np.zeros((n,frames),dtype=np.int8)
    n = len(keys)-1
    counter = 0
    for i in range(len(keys)):
        k2 = keys[i]
        if k2 != m_obj and k2 != 'G':
            dx = np.abs(O[k1]['x'][1:]-O[k2]['x'][1:])
            dy = np.abs(O[k1]['y'][1:]-O[k2]['y'][1:])
            dz = np.abs(O[k1]['z'][1:]-O[k2]['z'][1:])
            A = dx+dy+dz
            A[A<=1] = 1
            A[A>1] = 0
            dis_m[counter,:] = A
            counter += 1
            
    # computing direction
    dirx_m = np.zeros((n,frames),dtype=np.int8)
    diry_m = np.zeros((n,frames),dtype=np.int8)
    dirz_m = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)):
        k2 = keys[i]
        if k2 != m_obj and k2 != 'G':
            dx = O[k1]['x'][1:]-O[k2]['x'][1:]
            dy = O[k1]['y'][1:]-O[k2]['y'][1:]
            dz = O[k1]['z'][1:]-O[k2]['z'][1:]
            dirx_m[counter,:] = np.sign(dx).astype(int)
            diry_m[counter,:] = np.sign(dy).astype(int)
            dirz_m[counter,:] = np.sign(dz).astype(int)
            counter += 1
            
            
    # finding locations
    loci = {}
    locf = {}
    for key in keys:
        loci[key] = [O[key]['x'][0],O[key]['y'][0]]
        locf[key] = [O[key]['x'][-1],O[key]['y'][-1]]
            
        
    # plotting the all transitions
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
        m_count = 2.0                     # moving object location
        r_count = m_count+.5                 # 1 is reserved for the moving object
        obj_count = 3.0                 # 1 is reserved for the moving object
        G_count = 1.0
        for key in keys:
                if key == m_obj:    
                    G.add_node(str(key),type1='mo',position=(m_count,3))
                    G.add_node(str(key)+'_c',value=O[key]['color'],type1='of',position=(m_count-.25,1));         #color
                    G.add_node(str(key)+'_s',value=O[key]['shape'],type1='of',position=(m_count,1));         #shape
                    G.add_node(str(key)+'_l',value=loc[key],type1='of',position=(m_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_c')
                    G.add_edge(str(key),str(key)+'_s')
                    G.add_edge(str(key),str(key)+'_l')
                elif key == 'G':    
                    G.add_node(str(key),type1='G',position=(G_count,3))
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
                G.add_node(str(k1)+'_'+str(k2)+'_dist',value=dis_m[counter,distance],type1='rf',position=(r_count,5));         #distance
                direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                G.add_node(str(k1)+'_'+str(k2)+'_dir',value=direction,type1='rf',position=(r_count-.15,5));                     #direction
                G.add_node(str(k1)+'_'+str(k2)+'_mot',value=direction,type1='rf',position=(r_count+.15,5));                     #motion
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
                counter += 1
                r_count += 1
                
            if k2 != k1 and k2 == 'G':
                G.add_node(str(k1)+'_'+str(k2),type1='r',position=(G_count+.5,7.0))     # it's a directed node from k1 to k2
                G.add_edge(str(k1)+'_'+str(k2),str(k1))
                G.add_edge(str(k1)+'_'+str(k2),str(k2))
                G.add_node(str(k1)+'_'+str(k2)+'_dist',value=dis_m[counter,distance],type1='rf',position=(G_count+.5,5));         #distance
                direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                G.add_node(str(k1)+'_'+str(k2)+'_dir',value=direction,type1='rf',position=(G_count+.5-.15,5));                     #direction
                G.add_node(str(k1)+'_'+str(k2)+'_mot',value=direction,type1='rf',position=(G_count+.5+.15,5));                     #motion
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
                
        agents = list((n for n in G if G.node[n]['type1']=='G'))
        m_objects = list((n for n in G if G.node[n]['type1']=='mo'))
        objects = list((n for n in G if G.node[n]['type1']=='o'))
        objects_f = list((n for n in G if G.node[n]['type1']=='of'))
        relations = list((n for n in G if G.node[n]['type1']=='r'))
        relations_f = list((n for n in G if G.node[n]['type1']=='rf'))
        for node in relations_f:
            print G.node[node]
        
        #pos=nx.graphviz_layout(G,prog="neato")
    	pos = nx.get_node_attributes(G,'position')
        
        nx.draw_networkx_nodes(G,pos, nodelist=agents, node_color='c', node_size=100, alpha=1)
        nx.draw_networkx_nodes(G,pos, nodelist=m_objects, node_color='r', node_size=100, alpha=1)
        nx.draw_networkx_nodes(G,pos, nodelist=objects, node_color='b', node_size=100, alpha=.8)
        nx.draw_networkx_nodes(G,pos, nodelist=objects_f, node_color='y', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G,pos, nodelist=relations, node_color='g', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G,pos, nodelist=relations_f, node_color='y', node_size=100, alpha=0.8)
        #nx.draw_networkx_edges(G,pos, with_labels=False, edge_color='r', width=6.0, alpha=0.5)
        nx.draw_networkx_edges(G,pos, alpha=0.8)

                 
        ax[sub].axis('on')
        plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
        #plt.savefig("node_colormap.png") # save as png
    plt.show() # display

    
    
    
    
    
    
    
    
    
    
    
    
