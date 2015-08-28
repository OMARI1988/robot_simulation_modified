import numpy as np
# from nltk import PCFG
import pickle

g = 'grammar_00092'
file1 = open('/home/omari/Dropbox/robot_modified/EN/grammar/'+g+'.txt', 'r')
grammar = ''
g1 = [i for i in file1.readlines()]
for i in g1:
    grammar += i
print grammar
#learned_pcfg = PCFG.fromstring(grammar)
#print learned_pcfg
grammar         = pickle.load( open( "/home/omari/Dropbox/robot_modified/EN/pickle/"+g+".p", "rb" ) )
T          = grammar[0]
N          = grammar[1]
no_match   = grammar[2]
print T['features']['F_SHAPE']
print T['sum']['F_SHAPE']
print T['features']['F_HSV']
print T['sum']['F_HSV']
print T['features']['F_POS']
print T['sum']['F_POS']
print N
