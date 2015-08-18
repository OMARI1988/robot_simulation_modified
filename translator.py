import goslate
import numpy as np
import codecs
import binascii

dir1 = '/home/omari/Dropbox/robot_modified/motion/scene'
gs = goslate.Goslate()

for scene in range(1,2):
    f_new = codecs.open('/home/omari/Dropbox/robot_modified/motion_AR/scene'+str(scene)+'.txt','w','utf8')
    print 'reading scene number:',scene
    f_english = open(dir1+str(scene)+'.txt', 'r')
    data = f_english.read().split('\n')
    sentence = ''

    next_line = 0
    for count,line in enumerate(data):

        if next_line:
            sentence += line.split(':')[0]+':'
            s = gs.translate(line.split(':')[1], 'ar').split(' ')
            print s
            print gs.translate(line.split(':')[1], 'ar')
            print '------------'
            s = s
            # corrections
            sentence += ' '.join(s)+'\n'
            next_line = 0
        else:
            sentence += line+'\n'

        if line.split(':')[0] == 'sentence':    #sentences.append(count+1)
            next_line = 1

    # reading sentences
    # S = {}
    # for count,s in enumerate(sentences):
    #     if data[s].split(':')[0] == 'GOOD':
    #         S[count] = (data[s].split(':')[1]).lower()
    #
    # for i in S:
    #     sentence += gs.translate(S[i], 'ar')+'\n'
    f_new.write(sentence)  # Stored on disk as UTF-8
