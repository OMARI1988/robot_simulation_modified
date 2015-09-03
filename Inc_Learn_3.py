import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing import *

P = process_data()
for scan in range(1):
  print 'scan number :',scan
  for scene in range(1,100):
    if scene in [891,892]: continue
    P._read(scene)                                  # Objects, Graph, Sentences
    P._print_scentenses()
    P._fix_data()                                   # correction to Data removing 20 and 40
    P._find_unique_words()                          # find the unique words in every valid sentence = P.words
    #P._compute_features_for_all()                   # = self.touch_all, self.motion_all
    #P._compute_features_for_moving_object()         # = self.touch_m_i, self.touch_m_f, self.dir_touch_m_i, self.dir_touch_m_f, self.locations_m_i, self.locations_m_f
    # print P.all_words
    #########################################################################################################
    #   I will start with relations (only direction) between the moving object and other objects that are   #
    #   in contact with it.                                                                                 #
    #########################################################################################################

    #P._transition()                                 # P.transition['motion'] P.transition['touch'] P.transition['all']
    #P._grouping()                                   # generate the edges between the nodes that are the same in motion or touching
    #P._compute_unique_color_shape()                 # = P.unique_colors
    #P._compute_unique_direction()                   # = P.unique_direction
    #P._compute_unique_motion()                      # = P.total_motion = {1: {(0, 1): 1, (1, 0): 1}, 2: {(0, 1, 0): 1}} self.unique_motion
    #P._compute_unique_location()                    # = P.unique_locations

    #P._build_obj_hyp()                              # P.hyp_language
    #P._build_relation_hyp()                         # P.hyp_language

    #########################################################################################################
    #   I will pass hypotheses that have probabilities above 98% this needs a formal definition             #
    #########################################################################################################

    #P._test_language_hyp()                          # self.hyp_language_pass > .98
    #P._test_relation_hyp()                          # self.hyp_language_pass > .98
    #P._build_parser()                               #


    #P._test_sentence_hyp()                          # test if the whole sentence make sense
    # it should match 100% of the motion, which means the user should describe every single motion.
    ## so if someone says pick the blue object, and the blue object was trapped under another object, this
    ## won't work
    # no 2 words are allowed to mean the same thing
    # look for entities
    # how to idintify the moving object ?! if any ?
    # how to udintify it's target location?! if any ?
    # should I keep the assumption that verbs don't span in a sentence !?



    print '**================= end of scene ===================**'

file3 = open("/home/omari/Dropbox/robot_modified/EN/hypotheses/all_words.txt", "w")
file3.write((' ').join(P.all_words))
file3.close()


file3 = open("/home/omari/Dropbox/robot_modified/EN/hypotheses/all_commands.txt", "w")
for i in P.all_sentences:
    file3.write(i+'\n')
file3.close()

#print P.pcfg1
#for word in P.hyp_language_pass:
#    print word,P.hyp_language_pass[word]['all']

##############################################################################################################
#   code book                                                                                                #
#   self.S              = all the sentences for a given scene                                                #
#   self.words          = all uniqe words in each sentence                                                   #
#   self.touch_all      = a matrix that contains all the touch relations between every pair of objects       #
#   self.motion_all     = a matrix that contains all the relative motiona between every pair of objects      #
#   self.touch_m_i      = a list of objects in contact with the moving object at time = 0                    #
#   self.touch_m_f      = a list of objects in contact with the moving object at time = tf                   #
#   self.dir_touch_m_i  = a list of directions between the objects that were in contact with the moving      #
#                         object at t = 0
#   self.dir_touch_m_f  = a list of directions between the objects that were in contact with the moving
#                         object at t = tf
#   self.locations_m_i  = a list of the initial locations of the moving object >> so far it has only 1
#   self.locations_m_f  = a list of the final locations of the moving object >> sp far it has only 1
#   self.transition['motion']   = a list of the frame number at which a transition occured in the
#                                 relative motion for all objects
#   self.transition['touch']    = a list of the frame number at which a transition occured in the
#                                 relative touch for all objects
#   self.transition['all']      = a list that contains all the frame number at which any transition
#                                 has happened
#   self.G_motion       = a graph that has edges as relative motion connection
#   self.G_touch        = a graph that has edges as relative touch connections
#   self.unique_colors          = a list contain all the unique colors
#   self.unique_shapes          = a list contain all the unique shapes
#   self.total_motion           = a dictionary that contains the possible motions and the number of each
#                                 sub motion
#   self.unique_motions         = a list contains all the unique motions
#   self.hyp_language_pass      = a dictionery contains all the passed hypotheses from language
#   self.unique_direction       = a list of all unique directions between the moving obj and any other
#                                obj that is in touch with it
#   self.unique_locations       = a list of all the start and end locations of the moving obj
##############################################################################################################
