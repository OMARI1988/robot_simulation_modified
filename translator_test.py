import goslate
import numpy as np
import codecs

gs = goslate.Goslate()

# for scene in range(1,2):
#     print 'reading scene number:',scene
#     f = open(dir1+str(scene)+'.txt', 'r')
#     data = f.read().split('\n')
#     sentences = []
#     for count,line in enumerate(data):
#         if line.split(':')[0] == 'sentence':    sentences.append(count+1)
#     # reading sentences
#     S = {}
#     for count,s in enumerate(sentences):
#         if data[s].split(':')[0] == 'GOOD':
#             S[count] = (data[s].split(':')[1]).lower()
#
#     f = codecs.open('/home/omari/Datasets/robot_modified/arabic/'+str(scene)+'.txt','w','utf8')
#     sentence = ''
#     for i in S:
#         sentence += gs.translate(S[i], 'ar')+'\n'
#     f.write(sentence)  # Stored on disk as UTF-8


f = codecs.open('/home/omari/Dropbox/robot_modified/languages/test.txt','w','utf8')
sentence = ''
test = ['move the red block and place it on top of the blue block that is on top of a green block','move the blue ball so that it is on top of the black cube in the back left corner']
test = ['pick','place','put','pick up','put down','move','shift','drop','take','remove']
for test1 in test:
    sentence += test1+'\n'
    sentence += gs.translate(test1, 'ar')+'\n'
    # sentence += gs.translate(test1, 'it')+'\n'
    # sentence += gs.translate(test1, 'es')+'\n'
    # sentence += gs.translate(test1, 'ch')+'\n'
    # sentence += gs.translate(test1, 'fr')+'\n'+'\n'
f.write(sentence)  # Stored on disk as UTF-8
f.close()
