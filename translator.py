import goslate
import numpy as np
import codecs

dir1 = '/home/omari/Datasets/robot_modified/motion/scene'
gs = goslate.Goslate()

for scene in range(1,2):
    print 'reading scene number:',scene
    f = open(dir1+str(scene)+'.txt', 'r')
    data = f.read().split('\n')
    sentences = []
    for count,line in enumerate(data):
        if line.split(':')[0] == 'sentence':    sentences.append(count+1)
    # reading sentences
    S = {}
    for count,s in enumerate(sentences):
        if data[s].split(':')[0] == 'GOOD':
            S[count] = (data[s].split(':')[1]).lower()

    f = codecs.open('/home/omari/Datasets/robot_modified/arabic/'+str(scene)+'.txt','w','utf8')
    sentence = ''
    for i in S:
        sentence += gs.translate(S[i], 'ar')+'\n'
    f.write(sentence)  # Stored on disk as UTF-8
