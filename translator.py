#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys

import goslate
import numpy as np
import codecs
import binascii

dir1 = '/home/omari/Dropbox/robot_modified/motion/scene'
gs = goslate.Goslate()
bad_AR_words = []
# bad_AR_words.append('بيضاء')
# bad_AR_words.append('يمكن')
# bad_AR_words.append('نقل')
# bad_AR_words.append('الأخضر')
# bad_AR_words.append('وتحريكه')
# bad_AR_words.append('تحريك')
# bad_AR_words.append('وضع')
# bad_AR_words.append('مر')
# bad_AR_words.append('اسطوانة')
# bad_AR_words.append('الهرم الخضراء')
# bad_AR_words.append('الهرم الحمراء')
# bad_AR_words.append('الهرم الصفراء')
# bad_AR_words.append('الهرم الزرقاء')

good_AR_words = []
good_AR_words.append('البيضاء')
good_AR_words.append('العلبة')
good_AR_words.append('أنقل')
good_AR_words.append('الخضراء')
good_AR_words.append('و حركها')
good_AR_words.append('حرك')
good_AR_words.append('ضع')
good_AR_words.append('سطح')
good_AR_words.append('الاسطوانة')
# good_AR_words.append('الهرم الأخضر')
# good_AR_words.append('الهرم الأحمر')
# good_AR_words.append('الهرم الأصفر')
# good_AR_words.append('الهرم الأزرق')

for scene in range(1,1001):
    f_new = codecs.open('/home/omari/Dropbox/robot_modified/motion_IT/scene'+str(scene)+'.txt','w','utf8')
    print 'reading scene number:',scene
    f_english = open(dir1+str(scene)+'.txt', 'r')
    data = f_english.read().split('\n')
    sentence = ''

    next_line = 0
    for count,line in enumerate(data):

        if next_line:
            sentence += line.split(':')[0]+':'
            s = gs.translate(line.split(':')[1], 'it').split(' ')
            # corrections
            # print s
            for count,word in enumerate(s):
                for g_word,b_word in zip(good_AR_words,bad_AR_words):
                    if unicode(b_word,encoding='UTF-8') == word:
                        s[count]=unicode(g_word,encoding='UTF-8')
            # print s
            # print '------------'
            sentence += ' '.join(s)+'\n'
            next_line = 0
        else:
            sentence += line+'\n'

        if line.split(':')[0] == 'sentence':    #sentences.append(count+1)
            next_line = 1

    f_new.write(sentence)  # Stored on disk as UTF-8
