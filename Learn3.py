import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing import *
dir1 = '/home/omari/Datasets/robot/motion/scene'

for scene in range(1,2):
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
    
    # computing distance
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
            
    # computing direction
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
            
    # computing motion
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
            

            
    f, ax = plt.subplots(len(transition),2) # first col motion , second distance
    
    for feature in range(2):
        # plot the different graphs of motion and distance
        for sub,T in enumerate(transition):
            print 'plotting graph : '+str(sub+1)+' from '+str(len(transition))
            for edge in G.edges(): G.remove_edge(edge[0],edge[1])
            counter = 0
            if feature == 0:
                for i in range(len(keys)-1):
                    for j in range(i+1,len(keys)):
                        k1 = keys[i]
                        k2 = keys[j]
                        speed = r_speed[counter,T]
                        counter += 1
                        if speed == 1:    G.add_edge(k1,k2,speed=1)
            else:
                for i in range(len(keys)-1):
                    for j in range(i+1,len(keys)):
                        k1 = keys[i]
                        k2 = keys[j]
                        d = dis[counter,T]
                        counter += 1
                        if d == 1:    G.add_edge(k1,k2,dis=1)
            plt.sca(ax[sub,feature])
            
            # layout graphs with positions using graphviz neato
            pos=nx.graphviz_layout(G,prog="neato")
            # color nodes the same in each connected subgraph
            C=nx.connected_component_subgraphs(G)
            cK = 0
            for i in C:
                cK += 1
            C=nx.connected_component_subgraphs(G)
            colors = np.linspace(.2,.6,cK)
            for count,g in enumerate(C):
                #c=[random.random()]*nx.number_of_nodes(g) # random color...
                c=[colors[count]]*nx.number_of_nodes(g) # same color...
                nx.draw(g,
                     pos,
                     node_size=80,
                     node_color=c,
                     vmin=0.0,
                     vmax=1.0,
                     with_labels=False
                     )
            #nx.draw(G)  # networkx draw()
            ax[sub,feature].axis('on')
            plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
            #plt.savefig("node_colormap.png") # save as png
    plt.show() # display
    

    
 
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
