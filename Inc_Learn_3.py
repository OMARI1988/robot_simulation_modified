import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from data_processing import *

P = process_data()
for scan in range(1):
  print 'scan number :',scan
  for scene in range(1,40):
    if scene in [891,892]: continue
    P._read(scene)                                  # Objects, Graph, Sentences
    P._print_scentenses()
    P._fix_data()                                   # correction to Data removing 20 and 40
    P._find_unique_words()                          # find the unique words in every valid sentence = P.words
    P._find_word_relations()                        # learns the word order pick is followed by up


    print '**================= end of scene ===================**'

# file3 = open("/home/omari/Dropbox/robot_modified/EN/hypotheses/all_words.txt", "w")
# file3.write((' ').join(P.all_words))
# file3.close()
#
#
# file3 = open("/home/omari/Dropbox/robot_modified/EN/hypotheses/all_commands.txt", "w")
# for i in P.all_sentences:
#     file3.write(i+'\n')
# file3.close()

print P.words_order

pickle.dump(P.words_order, open( "/home/omari/Dropbox/robot_modified/EN/pickle/words_order.p", "wb" ) )
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
