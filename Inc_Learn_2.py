import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing import *
import operator

P = process_data()

for scene in range(1,2):
    P._read(scene)                                  #Objects, Graph, Sentences
    P._fix_sentences()                              #remove sapces and dots
    P._print_scentenses()
    P._fix_data()       # correction to Data removing 20 and 40
    P._compute_features_for_all()
    P._compute_features_for_moving_object()
    P._transition()         
    P._grouping()
    P._compute_unique_color_shape()
    P._compute_unique_dist_dir()
    P._compute_unique_motion()
    
    #print P.dirx_all_i
    #print P.dirx_all_f
    #print P.diry_all_i
    #print P.diry_all_f
    #print P.dirz_all_i
    #print P.dirz_all_f
    #print P.motion
    #print P.transition['motion']
    #print '*****-----*****'
    #print P.colors
    #print P.shapes
    #print P.directions_x
    #print P.directions_y
    #print P.directions_z
    #print P.motions
        
    #P._update_words_hyp()
    P._build_phrases()
    P._build_action_hyp()
    P._build_moving_obj_hyp()
    P._test_action_hyp()
    
    #if scene >= 110:
hyp = {}
valid_hyp = {}
for word in hyp:
    count = hyp[word]['counter']
    if count > 1:
        valid_hyp[word] = {}
        valid_hyp[word]['color'] = 0
        valid_hyp[word]['shape'] = 0
        print word,count
        A = hyp[word]['color']
        B = hyp[word]['shape']
        print 'color'
        for c in A:
            if float(A[c])/count > .9:  
                print c,float(A[c])/count
                valid_hyp[word]['color'] += 1
        print 'shape'
        for s in B:
            if float(B[s])/count > .9:  
                print s,float(B[s])/count
                valid_hyp[word]['shape'] += 1
        print '-----'    
print '-------------------------------------'
print len(valid_hyp)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
