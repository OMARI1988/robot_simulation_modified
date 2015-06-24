import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
dir1 = '/home/omari/Datasets/robot/motion/scene'

for scene in range(1,2):
    G=nx.Graph()
    print 'reading scene number:',scene
    f = open(dir1+str(scene)+'.txt', 'r')
    data = f.read().split('\n')
    sentences = []
    objects = []
    gripper = []
    for count,line in enumerate(data):
        if line.split(':')[0] == 'sentence':    sentences.append(count+1)
        if line.split(':')[0] == 'object' and line.split(':')[1]!='gripper':    objects.append(count+1)
        if line.split(':')[0] == 'object' and line.split(':')[1]=='gripper':    gripper.append(count+1)
            
    S = {}
    for count,s in enumerate(sentences):
        if data[s].split(':')[0] == 'GOOD': S[count] = data[s].split(':')[1]
        
    O = {}
    for count,o in enumerate(objects):
        G.add_node(count)
        O[count] = {}
        O[count][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))          #x
        O[count][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))      #y
        O[count][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))      #z
        O[count][data[o+3].split(':')[0]] = np.asarray(map(float,data[o+3].split(':')[1].split(',')))           #color
        O[count][data[o+4].split(':')[0]] = float(data[o+4].split(':')[1])                                      #shape
            
    for o in gripper:
        G.add_node('G')
        O['G'] = {}
        O['G'][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))      #x
        O['G'][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))  #y
        O['G'][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))  #z
        
    # comuting motion
    for i in O:
        dx = (O[i]['x'][:-1]-O[i]['x'][1:])**2
        dy = (O[i]['y'][:-1]-O[i]['y'][1:])**2
        dz = (O[i]['z'][:-1]-O[i]['z'][1:])**2
        O[i]['speed'] = np.round(np.sqrt(dx+dy+dz)*100)
        O[i]['speed'][O[i]['speed']!=0] = 1
        
    # removing frames were agent is static
    ind = np.where(O['G']['speed']==0)[0]
    for i in O:
        O[i]['speed'] = np.delete(O[i]['speed'],ind)
        
    # compute relative motion
    # similar speed == 1 different == 0
    frames = len(O['G']['speed'])
    keys = O.keys()
    n = np.sum(range(len(keys)))
    r = np.zeros((n,frames),dtype=np.int8)
    counter = 0
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            for count,s in enumerate(np.abs(O[k1]['speed'] - O[k2]['speed'])):
                if s != 0: a = 0
                elif s == 0: a = 1
                r[counter,count] = a
            counter += 1
                
    
    # comput the transition intervals
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            counter += 1
    col = r[:,0]
    transition = [0]
    for i in range(1,frames):
        if np.sum(np.abs(col-r[:,i]))!=0:
            col = r[:,i]
            transition.append(i)
    
          
    f, ax = plt.subplots(len(transition),2) # first col motion , second distance
    # plot the different graphs of motion
    for sub,T in enumerate(transition):
        print 'plotting graph : '+str(sub+1)+' from '+str(len(transition))
        for edge in G.edges(): G.remove_edge(edge[0],edge[1])
        counter = 0
        for i in range(len(keys)-1):
            for j in range(i+1,len(keys)):
                k1 = keys[i]
                k2 = keys[j]
                r_speed = r[counter,T]
                counter += 1
                if r_speed == 1:    G.add_edge(k1,k2,speed=1)
        plt.sca(ax[sub,0])
        
        # layout graphs with positions using graphviz neato
        pos=nx.graphviz_layout(G,prog="neato")
        # color nodes the same in each connected subgraph
        C=nx.connected_component_subgraphs(G)
        colors = np.linspace(.2,.8,2)
        for count,g in enumerate(C):
            #c=[random.random()]*nx.number_of_nodes(g) # random color...
            c=[colors[count]]*nx.number_of_nodes(g) # same color...
            nx.draw(g,
                 pos,
                 node_size=50,
                 node_color=c,
                 vmin=0.0,
                 vmax=1.0,
                 with_labels=False
                 )
        #nx.draw(G)  # networkx draw()
        ax[sub,0].axis('on')
        plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
        #plt.savefig("node_colormap.png") # save as png
    plt.show() # display
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
