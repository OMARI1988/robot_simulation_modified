import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing_igmm import *
import time
import pickle

#########################################################################################################
#   Some good examples
#########################################################################################################
# scene 939 sentence 6

#########################################################################################################
#   inital processing                                                                                   #
#########################################################################################################
file1 = open('/home/omari/Dropbox/robot_modified/hypotheses/all_scenes.txt', 'r')
valid_scenes = [int(i.split('\n')[0]) for i in file1.readlines()]
plot = 0                                #
simple = 1                              # simple graph structure for multi processing
dropbox = 1                             # update dropbox

P = process_data(dropbox)
P.first_time = 1

print           'reading object hypotheses'
P.gmm_obj       = pickle.load( open( "/home/omari/Dropbox/robot_modified/pickle/gmm_obj_1000.p", "rb" ) )
print           'reading motion hypotheses'
P.hyp_motion    = pickle.load( open( "/home/omari/Dropbox/robot_modified/pickle/hyp_motion_1000.p", "rb" ) )
print           'reading relation hypotheses'
P.hyp_relation  = pickle.load( open( "/home/omari/Dropbox/robot_modified/pickle/hyp_relation_1000.p", "rb" ) )
print           'reading total motion'
P.all_total_motion  = pickle.load( open( "/home/omari/Dropbox/robot_modified/pickle/all_total_motion_1000.p", "rb" ) )

for scan in range(1):
  print 'scan number :',scan
  for scene in range(1,1001):
    if P.first_time:                P._read_grammar(scene,valid_scenes)
    #if scene not in valid_scenes:   continue
    #slightly hard [10,]
    if scene in [891,892]:          continue
    #ts = time.time()
    #########################################################################################################
    #   Read sentences and scenes                                                                           #
    #########################################################################################################
    P._read(scene)                                  # Objects, Graph, Sentences
    P._print_scentenses()

    #continue
    #########################################################################################################
    #  Process scenes                                                                                       #
    #########################################################################################################

    P._fix_data()                                   # correction to Data removing 20 and 40
    P._find_unique_words()                          # find the unique words in every valid sentence = P.words
    P._compute_features_for_all()                   # = self.touch_all, self.motion_all
    P._compute_features_for_moving_object()         # = self.touch_m_i, self.touch_m_f, self.dir_touch_m_i, self.dir_touch_m_f, self.locations_m_i, self.locations_m_f
    P._transition()                                 # P.transition['motion'] P.transition['touch'] P.transition['all']
    P._grouping()                                   # generate the edges between the nodes that are the same in motion or touching

    P._compute_unique_color_shape()                 # = P.unique_colors
    P._compute_unique_direction()                   # = P.unique_direction
    P._compute_unique_motion()                      # = P.total_motion = {1: {(0, 1): 1, (1, 0): 1}, 2: {(0, 1, 0): 1}} self.unique_motion
    P._compute_unique_location()                    # = P.unique_locations
    P._convert_color_shape_location_to_gmm(plot)    # = P.gmm_M, P.M
    #P._convert_direction_to_gmm(plot)              # to do
    #P._convert_motion_to_gmm(plot)                 # to do

    #########################################################################################################
    #   Learning starts here, first we update the histograms of objects, relations and motions              #
    #########################################################################################################
    P._build_obj_hyp_igmm()                         # build the object hypotheses with gmms
    P._build_relation_hyp()                         # keeps track of relations and words    P.hyp_relation
    P._build_motion_hyp()                           # keepps track of motion and words      P.hyp_motion
    P._save_all_features()                          # put all scene features in a single dict self.all_scene_features

    #########################################################################################################
    #   Testing hypotheses                                                                                  #
    #########################################################################################################
    P._test_relation_hyp()                          # self.hyp_relation_pass
    P._test_motion_hyp()                            # self.hyp_motion_pass
    P._test_obj_hyp()                               # self.hyp_language_pass > .7
    P._combine_language_hyp()                       # combine object, relations and motion hypotheses in one place.
    P._filter_hyp()                                 # filter hypotheses to include only .9 of the best hypothesis
    P._filter_phrases()                             # remove larger phrases (and small phrases commented) = self.hyp_language_pass

    #########################################################################################################
    #   Comparing language hypotheses to scene                                                              #
    #########################################################################################################
    P._create_scene_graph(simple)
    P._print_results()
    # if P.number_of_valid_hypotheses <= 50:
    P._get_all_valid_combinations()
    P._test_all_valid_combinations()
    #
    # #########################################################################################################
    # #   Build the grammar                                                                                   #
    # #########################################################################################################
    P._build_grammar()                               #
    P._analysis()
    print '**================= end of scene '+str(P.scene)+' ===================**'
    print '\n\n\n'

# P._save_all_sentences()
# pickle.dump( P.gmm_obj, open( "/home/omari/Datasets/robot_modified/pickle/gmm_obj_1000.p", "wb" ) )
# pickle.dump( P.hyp_motion, open( "/home/omari/Datasets/robot_modified/pickle/hyp_motion_1000.p", "wb" ) )
# pickle.dump( P.hyp_relation, open( "/home/omari/Datasets/robot_modified/pickle/hyp_relation_1000.p", "wb" ) )
# pickle.dump( P.all_total_motion, open( "/home/omari/Datasets/robot_modified/pickle/all_total_motion_1000.p", "wb" ) )
# print 'finished saving'


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
#   P.gmm_M                     = a dictionery of all bic gmms for all features
#   P.M                         = a dictionery for all M values of each gmm of each feature
#   self.hyp_language_pass      = a dictionery of all valid hypotheses in language
#   self.all_scene_features     = a dictionary that has all the scene features
#   self.G_i                    = a graph for the first frame in the scene
#   self.G_f                    = a graph for the final frame in the scene
#
##############################################################################################################

##############################################################################################################
# To Do
# 1- finish the matching
# 2- finish the grammar
# 3- translate into other languages and show it works
# 4-
##############################################################################################################
