import numpy as np
# from nltk import PCFG
import pickle
from data_processing_igmm import *
# from xml_functions import *
from description_functions import *



file1 = open('/home/omari/Dropbox/robot_modified/EN/hypotheses/all_scenes.txt', 'r')
valid_scenes = [int(i.split('\n')[0]) for i in file1.readlines()]

# scenes
grammar_scene = 120
scene = 1

#change data
black = 0
sphere = 1
cylinder = 0

dropbox = 1                                     # update dropbox
P = process_data(dropbox)                       # to read grammar and data
R = Robot()
P._read_grammar(grammar_scene+1,valid_scenes)   # read grammar up to a certain scene
# P._read(scene)                                  # Objects, Graph, Sentences
# P._print_scentenses()
# print '--------------------------------- T'
# print P.T
R.T = P.T.copy()
# print '--------------------------------- N'
# print P.N
R.N = P.N.copy()
# print '--------------------------------- Valid'
# print P.all_valid_hypotheses
R.all_valid_hypotheses = P.all_valid_hypotheses.copy()
R._sum_of_all_hypotheses()
# print '----------------------------------------'
# print P.Data



R.scene = scene
R._initilize_values()
R._fix_sentences()
R._change_data(black,sphere,cylinder)
R._initialize_scene()                       # place the robot and objects in the initial scene position without saving or motion
R._draw_the_arrow()
R._move_robot(0)
# print R.positions
# print '----------------------------------------'
# print R.positions_f
# print '----------------------------------------'

R._get_unique_features()
R._generate_all_sentences()
