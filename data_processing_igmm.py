import numpy as np
import networkx as nx
import itertools
from nltk import PCFG
import operator
#from sklearn import mixture
from GMM_functions import *
from GMM_BIC import *
from IGMM import *
import copy
import colorsys

import multiprocessing

"""
Focus 7bibi
el pcfg, the words that you don't know yet, keep them as a terminal. and build relations between all the word in the sentence. Once you figure out a word that belong to a feature.
Convert all the probabilities of that terminal the feature terminal .. ! this should work !
"""


#--------------------------------------------------------------------------------------------------------#
def _distance_test(m1,m2):
    return np.sqrt(np.sum((m1-m2)**2))/np.sqrt(len(m1))

#--------------------------------------------------------------------------------------------------------#
# no 2 phrases are allowed to intersect in the same sentence
def _intersection(subset, indices):
    no_intersection = 1
    all_indeces = []
    for w in subset:
        for w1 in indices[w]:
            for w2 in w1:
                if w2 not in all_indeces:
                    all_indeces.append(w2)
                else:
                    no_intersection = 0
    return no_intersection

#--------------------------------------------------------------------------------------------------------#
# this function tests one susbet of words at a time
def _all_possibilities_func(subset, hyp_language_pass):
    all_possibilities = []      # all the possibilities gathered in one list
    for word in subset:
        all_possibilities.append(hyp_language_pass[word]['all'])
    return all_possibilities

#--------------------------------------------------------------------------------------------------------#
# no 2 words are allowed to mean the same thing
def _not_same_func(element):
    not_same = 1
    features = {}
    for i in element:
        if i[0] not in features: features[i[0]] = []
        features[i[0]].append(i[1])
    for f in features:
        if len(features[f])>1:
            for f1 in range(len(features[f])-1):
                for f2 in range(f1+1,len(features[f])):
                    m1 = np.asarray(list(features[f][f1]))
                    m2 = np.asarray(list(features[f][f2]))
                    if len(m1) != len(m2):          continue        # motions !
                    if _distance_test(m1,m2)<.25:
                        not_same = 0
                        continue
    return not_same, features

#--------------------------------------------------------------------------------------------------------#
# does actions match ?   it should match 100%
def _motion_match(subset,element,indices,total_motion):
    hyp_motion = {}
    motion_pass = 0
    for k,word in enumerate(subset):
        if element[k][0] == 'motion':
            a = element[k][1]
            if a not in hyp_motion:     hyp_motion[a] = len(indices[word])
            else:                       hyp_motion[a] += len(indices[word])
    for i in total_motion:
        if total_motion[i] == hyp_motion:
            motion_pass = 1
    return motion_pass

#--------------------------------------------------------------------------------------------------------#
# all features should be in the scene
# NOTE: the location feature needs to be multiplied by 7
def _all_features_match(features,all_scene_features):
    # all features should be in the scene
    feature_match = 1
    matched_features = {}
    for f1 in features:
        if f1 == 'motion':
            for k1 in range(len(features[f1])):
                matched_features[features[f1][k1]] = features[f1][k1]
        for f2 in all_scene_features:
            if f1==f2:
                for k1 in range(len(features[f1])):
                    passed = 0
                    for k2 in range(len(all_scene_features[f2])):
                        m1 = np.asarray(features[f1][k1])
                        m2 = np.asarray(all_scene_features[f2][k2])
                        if f2=='location':  m2 /= 7.0
                        if _distance_test(m1,m2)<.25:
                            passed = 1
                            matched_features[features[f1][k1]] = all_scene_features[f2][k2]
                    if not passed:                                  feature_match = 0
    return feature_match, matched_features

#--------------------------------------------------------------------------------------------------------#
# parse the sentence according to hypotheses
def _parse_sentence(S,indices,subset,element,matched_features):
    parsed_sentence = []
    value_sentence = []
    for i in S.split(' '):
        #parsed_sentence.append('_')
        #value_sentence.append('_')
        parsed_sentence.append(i)
        value_sentence.append(i)
    for word1 in subset:
        for i1 in indices[word1]:
            for k1,j1 in enumerate(i1):
                if k1 == 0:
                    k = subset.index(word1)
                    parsed_sentence[j1] =   element[k][0]
                    value_sentence[j1]  =   matched_features[element[k][1]]
                else:
                    parsed_sentence[j1] =   'delete'
                    value_sentence[j1]  =   'delete'
    # remove the multipple word phrase and combine it into one
    while 1:
        if 'delete' in parsed_sentence:
            parsed_sentence.remove('delete')
            value_sentence.remove('delete')
        else:       break
    return parsed_sentence, value_sentence

#--------------------------------------------------------------------------------------------------------#
# divide sentence with activities
def _activity_domain(parsed_sentence, value_sentence, subset, element):
    motion_ind = [i for i, x in enumerate(parsed_sentence) if x == 'motion']
    motion_ind2 = [0]
    for i in motion_ind:
        motion_ind2.append(i)
        motion_ind2.append(i+1)
    motion_ind2 = motion_ind2+[len(parsed_sentence)]

    sentence = []
    values = []
    for i in range(len(motion_ind2)/2):
        sentence.append(parsed_sentence[motion_ind2[2*i]:motion_ind2[2*i+1]])
        values.append(value_sentence[motion_ind2[2*i]:motion_ind2[2*i+1]])

    verb_sentence = {}
    # verb_names = {}
    for k,ind in enumerate(motion_ind):
        verb_sentence[value_sentence[ind]] = {}

        # verb_name = 'NONE'
        # for i1,i2 in zip(subset,element):
        #     if i2[0]=='motion' and i2[1]==value_sentence[ind]:
        #         verb_name = i1
        #
        # verb_sentence[value_sentence[ind]] = verb_name


        # Before the verb
        verb_sentence[value_sentence[ind]]['before'] = {}
        verb_sentence[value_sentence[ind]]['before']['value'] = []
        verb_sentence[value_sentence[ind]]['before']['type'] = []
        c = sentence[k]
        d = values[k]
        if c != '':
            verb_sentence[value_sentence[ind]]['before']['type'] = c
            verb_sentence[value_sentence[ind]]['before']['value'] = d

        # After the verb
        verb_sentence[value_sentence[ind]]['after'] = {}
        verb_sentence[value_sentence[ind]]['after']['value'] = []
        verb_sentence[value_sentence[ind]]['after']['type'] = []
        c = sentence[k+1]
        d = values[k+1]
        if c != '':
            verb_sentence[value_sentence[ind]]['after']['type'] = c
            verb_sentence[value_sentence[ind]]['after']['value'] = d
    return verb_sentence

#--------------------------------------------------------------------------------------------------------#
# divide activity domain into TE and TV according to the activity
def _divide_into_TE_and_TV(activity_sentence):
    domains = ['before','after']
    for v in activity_sentence:
        for d in domains:
            activity_sentence[v][d]['valid_configurations'] = []
            if activity_sentence[v][d]['type'] != []:
                s1 = activity_sentence[v][d]['type']
                s2 = activity_sentence[v][d]['value']
                if v == (0,1,0,):
                    # we need both TE and TV
                    for L in range(1,len(s1)):
                        x = _check_TE_TV(s1[0:L],s1[L:len(s1)],s2[0:L],s2[L:len(s2)])
                        if x != []:
                            activity_sentence[v][d]['valid_configurations'].append(x)

                if v == (0,1,):
                    # we need TE
                    pass

                if v == (1,0,):
                    # we need TV
                    pass

            # for k in activity_sentence[v][d]['valid_configurations']:
            #     for i in k:
            #         print 'order',i[0]
            #         print 'sentences_a',i[1][0]
            #         print 'sentences_b',i[1][1]
            #         print 'TE results',i[2][0]
            #         print 'TE values',i[3][0]
            #         print 'TV results',i[4][0]
            #         print 'TV values',i[5][0]
            # print '----'

    return activity_sentence

#--------------------------------------------------------------------------------------------------------#
# check domain for TE and TV
def _check_TE_TV(a,b,a_val,b_val):
    Valid = []
    # [[the order TV,TE or TE,TV].[the sentences a,b],[TE results],[TV results]]
    TE_result,[s1,e1,r1],[s1_v,e1_v,r1_v] = _check_TE(a[:],a_val[:])
    TV_result,[s2,e2,r2],[s2_v,e2_v,r2_v] = _check_TV(b[:],b_val[:])
    if TE_result and TV_result:
        Valid.append([['TE','TV'],[a,b],[s1,e1,r1],[s1_v,e1_v,r1_v],[s2,e2,r2],[s2_v,e2_v,r2_v]])
    # TE_result,[s1,e1,r1] = _check_TE(b)
    # TV_result,[s2,e2,r2] = _check_TV(a)
    # if TE_result and TV_result:
    #     Valid.append([['TV','TE'],[a,b],[s1,e1,r1],[s2,e2,r2]])
    return Valid

#--------------------------------------------------------------------------------------------------------#
# check domain for TE e,ere,erere,...
def _check_TE(sentence,value):
    [s,e,r],[s_val,e_val,r_val] = _check_R_E(sentence[:],value[:])
    a = len(e)
    b = len(r)
    test_result = 0
    if a!=0:
        # MORE ENTITIES than RELATIONs
        if a-b == 1:                    # perfect case were entities are 1 more than relation
            test_result = 1
        ############# IF YOU WANT TO LEARN BETWEEN CHANGET THIS ! THIS DONT ALLOW 3 E and 1 R
        # elif a-b > 1:                   # Wrong with no special cases ex 3 e and 1 r
        #     test_result = 0
        # # MORE or EQUAL RELATIONS THAN ENTITIES
        ############# IF YOU WANT TO LEARN ((PICK UP the [[green object, which the pyrimd]] is near it )) uncomment
        # elif a-b<1:                     # there are more or equal relations, check if number of indivual entities can be more
        #     count = 0
        #     for i in e:
        #         count += len(i)
        #     if count > b:
        #         test_result = 1
        #     else:
        #         test_result = 0
    return test_result,[s,e,r],[s_val,e_val,r_val]

#--------------------------------------------------------------------------------------------------------#
# check domain for TV location,re,rere,...
def _check_TV(sentence,value):
    [s,e,r],[s_val,e_val,r_val] = _check_R_E(sentence[:],value[:])
    a = len(e)
    b = len(r)
    test_result = 0
    if a!=0:
        # MORE ENTITIES than RELATIONs
        if a-b == 0:                    # perfect case were entities are equal to relations
            test_result = 1
        elif b == 0 and a == 1:                  # there might be a special case if one of the entities is a location
            if e[0][0]=='location':
                test_result = 1
    return test_result,[s,e,r],[s_val,e_val,r_val]

#--------------------------------------------------------------------------------------------------------#
# check domain for relations and entities
def _check_R_E(a,b):
    e = []
    r = []
    s = [[]]
    e_val = []
    r_val = []
    s_val = [[]]
    entity = ['color','shape','location']
    relation = ['direction']
    to_remove = []
    for k,i in enumerate(a):
        if i not in entity and i not in relation:
            to_remove.append(k)
    for j in reversed(to_remove):
        a.pop(j)
        b.pop(j)
    if a != []:
        for word,value in zip(a,b):
            if s[-1] == []:
                s[-1] = [word]
                s_val[-1] = [value]
            else:
                if s[-1][-1] in entity:         #previous is entity
                    if word in entity       :   # new is entity
                        if word not in s[-1]:
                            s[-1].append(word)
                            s_val[-1].append(value)
                        else:
                            s.append([word])
                            s_val.append([value])
                    if word in relation     :   #new is relation
                        s.append([word])
                        s_val.append([value])

                elif s[-1][-1] in relation:       #previous is relation
                    if word in entity       :   #new is entity
                        s.append([word])
                        s_val.append([value])
                    if word in relation     :   #new is relation
                        if word not in s[-1]:
                            s[-1].append(word)
                            s_val[-1].append(value)
                        else:
                            s.append([word])
                            s_val.append([value])
        for sub,val in zip(s,s_val):
        # for sub in s:
            if sub[0] in entity:
                e.append(sub)
                e_val.append(val)
            if sub[0] in relation:
                r.append(sub)
                r_val.append(val)
    return [s,e,r],[s_val,e_val,r_val]

#--------------------------------------------------------------------------------------------------------#
# check verb sentences with for relations and entities
def _match_scene_to_hypotheses(activity_sentence,scene_i,scene_f):
    for v in activity_sentence:
        for d in activity_sentence[v]:
            activity_sentence[v][d]['valid_hypotheses'] = []
            for k in activity_sentence[v][d]['valid_configurations']:
                for i in k:
                    match = 0
                    order = i[0]
                    sentences_a = i[1][0]
                    sentences_b = i[1][1]
                    TE_results = i[2]
                    TE_values = i[3]
                    TV_results = i[4]
                    TV_values = i[5]
                    if v == (0,1,0,):
                        objects = _get_the_TE_from_scene(TE_results[:],TE_values[:],scene_i)
                        if len(objects)==1:
                            #print 'the object is:',objects
                            location = _get_the_TV_from_scene(TV_results[:],TV_values[:],scene_i)
                            if len(location)==1:
                                #print 'the target location is:',location
                                match = _match_final_scene(objects,'location',location,scene_f)
                                #print match
                    if v == (0,1,):
                        pass

                    if v == (1,0,):
                        pass
                    activity_sentence[v][d]['valid_hypotheses'].append(match)
    return activity_sentence

#--------------------------------------------------------------------------------------------------------#
# This function gets all the objects in the scene with certain features NO RELATIONS (get target entitity)
def _get_the_TE_from_scene(TE_results,TE_values,scene):
    m_objects = list((n for n in scene if scene.node[n]['type1']=='mo'))
    objects = list((n for n in scene if scene.node[n]['type1']=='o'))#
    all_objects = m_objects+objects
    features = TE_results[1]
    relations = TE_results[2]
    features_v = TE_values[1]
    relations_v = TE_values[2]
    obj_pass = []
    if relations == []:
        for obj in all_objects:
            ok = 1
            for feature,value in zip(features[0],features_v[0]):
                if np.sum(np.abs(np.asarray(scene.node[obj+'_'+feature]['value'])-np.asarray(value))) != 0:
                    ok = 0
            if ok:  obj_pass.append(obj)
    return obj_pass

#--------------------------------------------------------------------------------------------------------#
# This function gets all the objects in the scene with certain features NO RELATIONS (get target entitity)
def _get_the_TV_from_scene(TV_results,TV_values,scene):
    m_objects = list((n for n in scene if scene.node[n]['type1']=='mo'))
    objects = list((n for n in scene if scene.node[n]['type1']=='o'))#
    all_objects = m_objects+objects
    features = TV_results[1]
    relations = TV_results[2]
    features_v = TV_values[1]
    relations_v = TV_values[2]
    # print '---',features
    # print '---',features_v
    # print '---',relations
    # print '---',relations_v
    location_pass = []
    if relations == []:
        if len(features[0]) == 1:
            if features[0][0] == 'location':
                loc = np.asarray(features_v[0][0])/7.0
                location_pass.append(loc)
    elif len(relations)==1 and len(features)==1:
        obj_pass = []
        for obj in all_objects:
            ok = 1
            for feature,value in zip(features[0],features_v[0]):
                if np.sum(np.abs(np.asarray(scene.node[obj+'_'+feature]['value'])-np.asarray(value))) != 0:
                    ok = 0
            if ok:  obj_pass.append(obj)
        for obj in obj_pass:
            #NOTE: if you want to learn right and left this is where you have to look
            #NOTE: this will only work for over !
            #NOTE: if you want to learn nearset its also here
            loc = scene.node[obj+'_'+'location']['value']
            location_pass.append(loc)
    return location_pass

#--------------------------------------------------------------------------------------------------------#
# match the final scene location
def _match_final_scene(objects,target_feature,target_value,scene):
    match = 0
    loc = scene.node[objects[0]+'_'+target_feature]['value']
    if np.sum(np.abs(np.asarray(scene.node[objects[0]+'_'+target_feature]['value'])-np.asarray(target_value)))==0:
        match = 1
    return match

#--------------------------------------------------------------------------------------------------------#
def _print_results(activity_sentence,scene_description,parsed_sentence,subset,element,matched_features,scene,L):
    results = []
    for v in activity_sentence:
        for d in ['before','after']:
            for k2,k in enumerate(activity_sentence[v][d]['valid_hypotheses']):
                if k:
                    print scene_description
                    print 'scene number:',scene
                    print 'L:',L
                    for a,b in zip(subset,element):
                        print a,b
                    print '---------------'
                    for k3,k1 in enumerate(activity_sentence[v][d]['valid_configurations']):
                        if k2==k3:
                            for i in k1:
                                print 'order        :',i[0]
                                print 'target entity:',i[1][0]
                                print 'target value :',i[1][1]
                                print 'TE results   :',i[2][0]
                                print 'TE values    :',i[3][0]
                                print 'TV results   :',i[4][0]
                                print 'TV values    :',i[5][0]
                                results.append([i[0],i[1][0],i[1][1],subset,element,parsed_sentence,d,v,scene_description])
                    print '*****************'
                    print
                    print
    return results








#--------------------------------------------------------------------------------------------------------#
# check verb sentences with for relations and entities
def _check_relation_entity_numbers(verb_sentence):

    possibilities = ['before','after']
    entity = ['color','shape','location']
    relation = ['direction']
    structure = {}
    for p in possibilities:
        for v in verb_sentence:
            if v not in structure:  structure[v] = {}
            structure[v][p] = {}
            structure[v][p]['s'] = []
            structure[v][p]['e'] = []
            structure[v][p]['r'] = []
            structure[v][p]['s_val'] = []
            structure[v][p]['e_val'] = []
            structure[v][p]['r_val'] = []
            structure[v][p]['RE_count_result'] = 0
            #print 'the verb is',v,'the sentence is',p
            sentence    = verb_sentence[v][p]['type']
            values      = verb_sentence[v][p]['value']
            while 1:
                if '_' in sentence:     sentence.remove('_')
                else:                   break
            while 1:
                if '_' in values:       values.remove('_')
                else:                   break
            if sentence != []:
                e = []
                r = []
                s = [[]]
                e_val = []
                r_val = []
                s_val = [[]]
                # keeping in mind that relations dont span, and objects dont span
                # finding the minimum entities in a sentence
                for word,value in zip(sentence,values):
                    if s[-1] == []:
                        s[-1] = [word]
                        s_val[-1] = [value]
                    else:
                        if s[-1][-1] in entity:         #previous is entity
                            if word in entity       :   # new is entity
                                if word not in s[-1]:
                                    s[-1].append(word)
                                    s_val[-1].append(value)
                                else:
                                    s.append([word])
                                    s_val.append([value])
                            if word in relation     :   #new is relation
                                s.append([word])
                                s_val.append([value])

                        elif s[-1][-1] in relation:       #previous is relation
                            if word in entity       :   #new is entity
                                s.append([word])
                                s_val.append([value])
                            if word in relation     :   #new is relation
                                if word not in s[-1]:
                                    s[-1].append(word)
                                    s_val[-1].append(value)
                                else:
                                    s.append([word])
                                    s_val.append([value])
                for sub,val in zip(s,s_val):
                    if sub[0] in entity:
                        e.append(sub)
                        e_val.append(val)
                    if sub[0] in relation:
                        r.append(sub)
                        r_val.append(val)
                # based on the number of allowed features to be in each entity or relation divide the sentence to get all posiible options :)
                # then check the 2n(O) and 1n(R)
                a = len(e)
                b = len(r)
                test_result = 'VERY BAD'
                # MORE ENTITIES than RELATIONs
                if a-b == 1:                    # perfect case were entities are 1 more than relation
                    test_result = 1
                ############# IF YOU WANT TO LEARN BETWEEN CHANGET THIS ! THIS DONT ALLOW 3 E and 1 R
                elif a-b > 2:                   # Wrong with no special cases ex 3 e and 1 r
                    test_result = 0
                elif a-b == 2:                  # there might be a special case if one of the entities is a location
                    for i in e:
                        if len(i) == 1 and i[0]=='location':
                            test_result = 1
                        else:
                            test_result = 0
                # MORE or EQUAL RELATIONS THAN ENTITIES
                elif a-b<1:                     # there are more or equal relations, check if number of indivual entities can be more
                    count = 0
                    for i in e:
                        count += len(i)
                    if count > b:
                        test_result = 1
                    else:
                        test_result = 0

                structure[v][p]['s'] = s
                structure[v][p]['e'] = e
                structure[v][p]['r'] = r
                structure[v][p]['s_val'] = s_val
                structure[v][p]['e_val'] = e_val
                structure[v][p]['r_val'] = r_val
                structure[v][p]['RE_count_result'] = test_result
    return structure

#--------------------------------------------------------------------------------------------------------#
# This needs to check target entity and target location for every operation in every sentence
def _check_graph_structure(structure):
    # structure is a dictionary that has the verbs, then before,after the verb, (sentence,entity,relations,and their values stored in s,e,r,s_val,e_val,r_val)
    for v in structure:
        for p in structure[v]:
            structure[v][p]['graph_result'] = 0
            if structure[v][p]['RE_count_result'] == 1:
                s = structure[v][p]['s']
                e = structure[v][p]['e']
                r = structure[v][p]['r']
                s_val = structure[v][p]['s_val']
                e_val = structure[v][p]['e_val']
                r_val = structure[v][p]['r_val']
                structure[v][p]['T_entity'] = []
                structure[v][p]['T_value']  = []
                a = len(e)
                b = len(r)
                if v == (0,1,0,):
                    # we need a target entity and a target value
                    if a-b == 1:                    # perfect case were entities are 1 more than relation
                        if b == 0:                  # there is only a single entity in the description
                            if 'location' in e[0] and len(e[0])>1:
                                if e[0][0] == 'location':
                                    T_entity        = e[0][1:-1]
                                    T_entity_val    = e_val[0][1:-1]
                                    T_value         = e[0][0]
                                    T_value_val     = e_val[0][0]
                                    structure[v][p]['T_entity'] = [[T_entity,T_entity_val]]
                                    structure[v][p]['T_value']  = [[T_value ,T_value_val ]]
                                    structure[v][p]['graph_result'] = 1
                                elif e[0][-1] == 'location':
                                    T_entity        = e[0][0:len(e[0])-1]
                                    T_entity_val    = e_val[0][0:len(e[0])-1]
                                    T_value         = e[0][-1]
                                    T_value_val     = e_val[0][-1]
                                    structure[v][p]['T_entity'] = [[T_entity,T_entity_val]]
                                    structure[v][p]['T_value']  = [[T_value ,T_value_val ]]
                                    structure[v][p]['graph_result'] = 1
                        if b > 0:                  # there are relations also
                            pass
                            # structure[v][p]['graph_result'] = 1

                if v == (0,1,):
                    # we need a target entitiy
                    pass


                if v == (1,0,):
                    # we need a target value
                    pass
    return structure

#--------------------------------------------------------------------------------------------------------#
# check verb sentences with for relations and entities
def _match_scene_to_hypotheses2(structure,scene_i,scene_f):
    for v in structure:
        for p in structure[v]:
            structure[v][p]['match_result'] = 0
            structure[v][p]['valid_hypotheses'] = []
            if structure[v][p]['graph_result'] == 1:
                if v == (0,1,0,):
                    for E,V in zip(structure[v][p]['T_entity'],structure[v][p]['T_value']):
                        match = 0
                        T_entity        = E[0]
                        T_entity_val    = E[1]
                        objects = _get_the_objects_from_scene2(T_entity,T_entity_val,scene_i)
                        if len(objects)==1:
                            T_value     =   V[0]
                            T_value_val =   V[1]
                            match = _match_final_scene(objects,T_value,T_value_val,scene_f)
                            if match:
                                structure[v][p]['match_result'] = 1
                        structure[v][p]['valid_hypotheses'].append(match)
                if v == (0,1,):
                    pass

                if v == (1,0,):
                    pass
    return structure

#--------------------------------------------------------------------------------------------------------#
# This function gets all the objects in the scene with certain features NO RELATIONS
def _get_the_objects_from_scene2(obj_features,obj_value,scene):

    m_objects = list((n for n in scene if scene.node[n]['type1']=='mo'))
    objects = list((n for n in scene if scene.node[n]['type1']=='o'))#
    all_objects = m_objects+objects

    obj_pass = []
    for obj in all_objects:
        ok = 1
        for feature,value in zip(obj_features,obj_value):
            if np.sum(np.abs(np.asarray(scene.node[obj+'_'+feature]['value'])-np.asarray(value))) != 0:
                ok = 0
        if ok:  obj_pass.append(obj)
    return obj_pass

#--------------------------------------------------------------------------------------------------------#
def _get_results(structure,scene_description,parsed_sentence,subset,element,matched_features):
    for v in structure:
        for p in structure[v]:
            A = structure[v][p]
            if A['match_result'] == 1:
                for result,T_e,T_v in zip(A['valid_hypotheses'],A['T_entity'],A['T_value']):
                    if result:
                        print scene_description
                        print parsed_sentence
                        print T_e
                        print T_v
                        for k,word in enumerate(subset):
                            print word,'-',element[k][0],matched_features[element[k][1]]
                        print '-----------------------------------'




#--------------------------------------------------------------------------------------------------------#
def calc(data):
    # initial
    subset = data[0]
    indices = data[1]
    hyp_language_pass = data[2]
    total_motion = data[3]
    scene_description = data[4]
    all_scene_features = data[5]
    graph_i = data[6][0]
    graph_f = data[6][1]
    scene = data[7]
    L = data[8]
    results = [[]]
    #---------------------------------------------------------------#
    # no 2 phrases are allowed to intersect in the same sentence
    no_intersection = _intersection(subset,indices)
    if no_intersection:
        all_possibilities = _all_possibilities_func(subset, hyp_language_pass)
        #---------------------------------------------------------------#
        # find the actual possibilities for every word in the subset
        for element in itertools.product(*all_possibilities):
            #---------------------------------------------------------------#
            # no 2 words are allowed to mean the same thing
            not_same, features = _not_same_func(element)
            if not_same:
                #---------------------------------------------------------------#
                # all features should be in the scene
                feature_match, matched_features = _all_features_match(features,all_scene_features)
                if feature_match:
                    #---------------------------------------------------------------#
                    # does actions match ?   it should match 100%
                    motion_pass = _motion_match(subset,element,indices,total_motion)
                    if motion_pass:
                        #---------------------------------------------------------------#
                        # parse the sentence
                        parsed_sentence, value_sentence = _parse_sentence(scene_description,indices,subset,element,matched_features)
                        #---------------------------------------------------------------#
                        # divide sentence with verbs
                        activity_sentence = _activity_domain(parsed_sentence, value_sentence, subset, element)
                        activity_sentence = _divide_into_TE_and_TV(activity_sentence)
                        activity_sentence = _match_scene_to_hypotheses(activity_sentence,graph_i,graph_f)
                        results.append(_print_results(activity_sentence,scene_description,parsed_sentence,subset,element,matched_features,scene,L))
                        #print scene,L,'---',results
                        # old method ! not clear
                        # structure = _check_relation_entity_numbers(activity_sentence)
                        # structure = _check_graph_structure(structure)
                        # structure = _match_scene_to_hypotheses2(structure,graph_i,graph_f)
                        # results   = _get_results(structure,scene_description,parsed_sentence,subset,element,matched_features)
    return (results,)


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

class process_data():
    def __init__(self):
        self.dir1 = '/home/omari/Datasets/robot_modified/motion/scene'
        self.dir2 = '/home/omari/Datasets/robot_modified/scenes/'
        self.dir3 = '/home/omari/Datasets/robot_modified/graphs/scene'

        # initial language grammar
        self.N                      = {}                    # non-terminals
        self.N['feature']           = {}                    # non-terminals
        self.N['sum']               = {}                    # non-terminals
        #self.N['e']                 = {}                    # non-terminals entity
        #self.N['e']['sum']          = 0.0                   # non-terminals entity counter
        self.T = {}                                         # terminals
        self.T['features'] = {}
        self.T['sum'] = {}

        self.no_match = {}
        self.no_match['features'] = {}
        self.no_match['sum'] = {}

        # connecting language to vision hypotheses
        self.hyp_language           = {}
        self.hyp_relation           = {}
        self.hyp_motion             = {}
        self.hyp_language_pass      = {}
        self.hyp_obj_pass           = {}
        self.hyp_relation_pass      = {}
        self.hyp_motion_pass        = {}
        self.feature_pass           = {}
        self.all_total_motion       = {}
        self.all_scene_features     = {}


        #phrases
        self.n_word                 = 3
        self.step                   = 3
        self.all_words              = []

        #gmm
        self.gmm_obj                = {}
        self.gmm_M                  = {}
        #self.cv_types = ['spherical', 'tied', 'diag', 'full']
        self.cv_types = ['full']

        # habituation parameter
        self.p_parameter            = 3.0                       # probability parameter for exp function in habituation
        self.pass_distance          = .25                       # distance test for how much igmms match
        self.pass_distance_phrases  = .25                       # distance test for how much phrases match
        self.p_obj_pass             = .7                        # for object
        self.p_relation_pass        = .8                        # for both relation and motion
        self.pool = multiprocessing.Pool(8)

    #--------------------------------------------------------------------------------------------------------#
    # read the sentences and data from file
    def _read(self,scene):
        self.scene = scene
        print 'reading scene number:',self.scene
        f = open(self.dir1+str(scene)+'.txt', 'r')
        data = f.read().split('\n')
        self.G = nx.Graph()
        sentences = []
        objects = []
        gripper = []
        for count,line in enumerate(data):
            if line.split(':')[0] == 'sentence':    sentences.append(count+1)
            if line.split(':')[0] == 'object' and line.split(':')[1]!='gripper':    objects.append(count+1)
            if line.split(':')[0] == 'object' and line.split(':')[1]=='gripper':    gripper.append(count+1)
        # reading sentences
        self.S = {}
        for count,s in enumerate(sentences):
            if data[s].split(':')[0] == 'GOOD': self.S[count] = (data[s].split(':')[1]).lower()
        # reading Data of objects
        self.Data = {}
        for count,o in enumerate(objects):
            self.G.add_node(str(count),type1='obj')
            self.Data[count] = {}
            self.Data[count][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))          #x
            self.Data[count][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))      #y
            self.Data[count][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))      #z
            self.Data[count][data[o+3].split(':')[0]] = np.asarray(map(float,data[o+3].split(':')[1].split(',')))           #color
            self.Data[count][data[o+4].split(':')[0]] = float(data[o+4].split(':')[1])                                      #shape
        # reading Data of robot
        for o in gripper:
            self.G.add_node('G',type1='G')
            self.Data['G'] = {}
            self.Data['G'][data[o].split(':')[0]] = np.asarray(map(float,data[o].split(':')[1].split(',')[:-1]))      #x
            self.Data['G'][data[o+1].split(':')[0]] = np.asarray(map(float,data[o+1].split(':')[1].split(',')[:-1]))  #y
            self.Data['G'][data[o+2].split(':')[0]] = np.asarray(map(float,data[o+2].split(':')[1].split(',')[:-1]))  #z

    #--------------------------------------------------------------------------------------------------------#
    # print sentences on terminal
    def _print_scentenses(self):
        for count,i in enumerate(self.S):
            print count,'-',self.S[i]
        print '====-----------------------------------------------------------------===='

    #--------------------------------------------------------------------------------------------------------#
    # correcting motion planning simulation
    def _fix_data(self):
        # correction to Data removing the step
        for i in self.Data:
            self.Data[i]['x'] = np.delete(self.Data[i]['x'],[self.step,2*self.step])
            self.Data[i]['y'] = np.delete(self.Data[i]['y'],[self.step,2*self.step])
            self.Data[i]['z'] = np.delete(self.Data[i]['z'],[self.step,2*self.step])

    #--------------------------------------------------------------------------------------------------------#
    # find phrases and words for each sentence
    def _find_unique_words(self):
        self.phrases = {}                       # hold the entire phrase up to a certain number of words
        self.words = {}                         # hold the list of independent words in a sentence
        # read the sentence
        for s in self.S:
            self.phrases[s] = []
            self.words[s] = []
            sentence = self.S[s]
            w = sentence.split(' ')
            phrases = []
            for i in range(len(w)):
                if w[i]not in self.words[s]: self.words[s].append(w[i])
                for j in range(i+1,np.min([i+1+self.n_word,len(w)+1])):
                    phrases.append(' '.join(w[i:j]))
            for i in phrases:
                if i not in self.phrases[s]:    self.phrases[s].append(i)

    #--------------------------------------------------------------------------------------------------------#
    def _compute_features_for_all(self):
        # this function comutes the following
        # self.touch_all
        # self.motion_all

        # initial parameter
        self.frames = len(self.Data['G']['x'])-1     # we remove the first one to compute speed
        self.keys = self.Data.keys()
        self.n = np.sum(range(len(self.keys)))
        # computing distance and touch   (between all objects including robot)
        #self.dis_all = np.zeros((self.n,self.frames),dtype=np.float)
        self.touch_all = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)-1):
            for j in range(i+1,len(self.keys)):
                k1 = self.keys[i]
                k2 = self.keys[j]
                dx = np.abs(self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:])
                dy = np.abs(self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:])
                dz = np.abs(self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:])
                A = dx+dy+dz
                #self.dis_all[counter,:] = A
                A[A<=1.0] = 1
                A[A>1.0] = 0
                self.touch_all[counter,:] = A
                counter += 1

        """
        # computing direction   (between all objects not robot)
        self.dirx_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirx_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.diry_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.diry_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirz_all_i = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        self.dirz_all_f = np.zeros((len(self.keys)-1,len(self.keys)-1),dtype=np.int8)
        for i in self.keys:
            for j in self.keys:
                if i!=j and i!='G' and j!='G':
                    k1 = self.keys[i]
                    k2 = self.keys[j]
                    dxi = self.Data[k1]['x'][0]-self.Data[k2]['x'][0]
                    dxf = self.Data[k1]['x'][-1]-self.Data[k2]['x'][-1]
                    dyi = self.Data[k1]['y'][0]-self.Data[k2]['y'][0]
                    dyf = self.Data[k1]['y'][-1]-self.Data[k2]['y'][-1]
                    dzi = self.Data[k1]['z'][0]-self.Data[k2]['z'][0]
                    dzf = self.Data[k1]['z'][-1]-self.Data[k2]['z'][-1]
                    self.dirx_all_i[i,j] = np.sign(dxi)
                    self.dirx_all_f[i,j] = np.sign(dxf)
                    self.diry_all_i[i,j] = np.sign(dyi)
                    self.diry_all_f[i,j] = np.sign(dyf)
                    self.dirz_all_i[i,j] = np.sign(dzi)
                    self.dirz_all_f[i,j] = np.sign(dzf)
        """

        # computing motion      (for all objects)
        for i in self.Data:
            dx = (self.Data[i]['x'][:-1]-self.Data[i]['x'][1:])**2
            dy = (self.Data[i]['y'][:-1]-self.Data[i]['y'][1:])**2
            dz = (self.Data[i]['z'][:-1]-self.Data[i]['z'][1:])**2
            self.Data[i]['motion'] = (np.round(np.sqrt(dx+dy+dz)*10000)).astype(int)
            self.Data[i]['motion'][self.Data[i]['motion']!=0] = 1

        # compute relative motion
        # similar motion == 1 different == 0
        self.motion_all = np.zeros((self.n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)-1):
            for j in range(i+1,len(self.keys)):
                k1 = self.keys[i]
                k2 = self.keys[j]
                A = np.abs(self.Data[k1]['motion'] - self.Data[k2]['motion'])
                A[A==0] = 2
                A[A!=2] = 0
                A[A!=0] = 1     # a little trick to make 0=1 and 1=0
                self.motion_all[counter,:] = A
                counter += 1

    #--------------------------------------------------------------------------------------------------------#
    def _compute_features_for_moving_object(self):
        # finding the moving object ! fix this
        # self.m_obj
        # self.touch_m
        # self.touch_m_i        # initially touching the moving obj
        # self.touch_m_f        # finally touching the moving obj

        self.m_obj = []
        for i in self.Data:
            if i != 'G':
                x = np.abs(self.Data[i]['x'][-1]-self.Data[i]['x'][0])
                y = np.abs(self.Data[i]['y'][-1]-self.Data[i]['y'][0])
                z = np.abs(self.Data[i]['z'][-1]-self.Data[i]['z'][0])
                if (x+y+z) > 0:
                    self.m_obj = i

        # computing distance BINARY Distance (touch or no touch all obj no robot)
        n = len(self.Data)-2
        #self.dis_m = np.zeros((n,self.frames),dtype=np.float)
        self.touch_m = np.zeros((n,self.frames),dtype=np.uint8)
        counter = 0
        k1 = self.m_obj

        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                dx = np.abs(self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:])
                dy = np.abs(self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:])
                dz = np.abs(self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:])
                A = dx+dy+dz
                #self.dis_m[counter,:] = A
                A[A<=1.5] = 1
                A[A>1] = 0
                self.touch_m[counter,:] = A
                counter += 1

        counter = 0
        self.touch_m_i = []                 #which objects were in touch with the moving objects initially
        self.touch_m_f = []                 #which objects were in touch with the moving objects finally
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                if self.touch_m[counter,0]: self.touch_m_i.append(i)
                if self.touch_m[counter,-1]: self.touch_m_f.append(i)
                counter += 1

        # computing direction
        self.dirx_m = np.zeros((n,self.frames),dtype=np.int8)
        self.diry_m = np.zeros((n,self.frames),dtype=np.int8)
        self.dirz_m = np.zeros((n,self.frames),dtype=np.int8)
        counter = 0
        for i in range(len(self.keys)):
            k2 = self.keys[i]
            if k2 != k1 and k2 != 'G':
                dx = self.Data[k1]['x'][1:]-self.Data[k2]['x'][1:]
                dy = self.Data[k1]['y'][1:]-self.Data[k2]['y'][1:]
                dz = self.Data[k1]['z'][1:]-self.Data[k2]['z'][1:]
                self.dirx_m[counter,:] = np.sign((dx).astype(int))
                self.diry_m[counter,:] = np.sign((dy).astype(int))
                self.dirz_m[counter,:] = np.sign(np.round(dz))
                counter += 1

        self.dir_touch_m_i = []
        self.dir_touch_m_f = []
        for i in self.touch_m_i:
            if i > int(k1): val = i-1
            else:           val = i
            a = self.dirx_m[val,0]
            b = self.diry_m[val,0]
            c = self.dirz_m[val,0]
            d = (a,b,c)
            if d not in self.dir_touch_m_i:    self.dir_touch_m_i.append(d)
        for i in self.touch_m_f:
            if i > int(k1): val = i-1
            else:           val = i
            a = self.dirx_m[val,-1]
            b = self.diry_m[val,-1]
            c = self.dirz_m[val,-1]
            d = (a,b,c)
            if d not in self.dir_touch_m_f:    self.dir_touch_m_f.append(d)

        # finding locations of moving object
        self.locations_m_i = []
        self.locations_m_f = []
        self.locations_m_i.append((self.Data[self.m_obj]['x'][0],self.Data[self.m_obj]['y'][0]))
        self.locations_m_f.append((self.Data[self.m_obj]['x'][-1],self.Data[self.m_obj]['y'][-1]))
        #self.locations.append(di)
        #if df not in self.locations: self.locations.append(df)

    #--------------------------------------------------------------------------------------------------------#
    # find the transitions for touch and motion
    def _transition(self):
        # comput the transition intervals for motion
        self.motion = [self.Data[self.m_obj]['motion'][0]]
        col = self.motion_all[:,0]
        self.transition = {}
        self.transition['motion'] = [0]
        self.transition['all'] = [0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.motion_all[:,i]))!=0:
                col = self.motion_all[:,i]
                self.transition['motion'].append(i)
                self.transition['all'].append(i)
                self.motion.append(self.Data[self.m_obj]['motion'][i])
        # comput the transition intervals for touch
        self.transition['touch'] = [0]
        col = self.touch_all[:,0]
        for i in range(1,self.frames):
            if np.sum(np.abs(col-self.touch_all[:,i]))!=0:
                col = self.touch_all[:,i]
                self.transition['touch'].append(i)
                if i not in self.transition['all']: self.transition['all'].append(i)
        self.transition['all'] = sorted(self.transition['all'])

    #--------------------------------------------------------------------------------------------------------#
    # group touch and motion objects
    def _grouping(self):
        self.G_motion = self._grouping_template(self.transition['motion'],self.motion_all)
        self.G_touch = self._grouping_template(self.transition['touch'],self.touch_all)

    #-------------------------------------------------#
    # sub funtion
    def _grouping_template(self,transition,feature):
        G_all = {}
        for T in transition:
            G = self.G.copy()
            counter = 0
            for i in range(len(self.keys)-1):
                for j in range(i+1,len(self.keys)):
                    k1 = self.keys[i]
                    k2 = self.keys[j]
                    a = feature[counter,T]
                    counter += 1
                    if a == 1:    G.add_edge(str(k1),str(k2),value=1)
            G_all[T] = {}
            G_all[T]['graph'] = G
            G_all[T]['groups'] = []
            C=nx.connected_component_subgraphs(G)
            for g in C:
                G_all[T]['groups'].append(g.nodes())
        return G_all

    #--------------------------------------------------------------------------------------------------------#
    # compute unique color and shape for every scene
    def _compute_unique_color_shape(self):
        self.unique_colors = []
        self.unique_shapes = []
        for i in self.Data:
            if i != 'G':
                c = self.Data[i]['color']
                C = (c[0],c[1],c[2])
                s = self.Data[i]['shape']
                if C not in self.unique_colors: self.unique_colors.append(C)
                if s not in self.unique_shapes: self.unique_shapes.append(s)

    #--------------------------------------------------------------------------------------------------------#
    # compute unique motion for every scene
    def _compute_unique_motion(self):
        self.unique_motions = []
        self.total_motion = {}
        for i in range(2, len(self.motion)+1):  #possible windows
            self.total_motion[i-1] = {}
            for j in range(len(self.motion)+1-i):
                c = self.motion[j:j+i]
                if i == 2:  C = (c[0],c[1])
                if i == 3:  C = (c[0],c[1],c[2])
                if i == 4:  C = (c[0],c[1],c[2],c[3])
                if i == 5:  C = (c[0],c[1],c[2],c[3],c[4])
                if i == 6:  C = (c[0],c[1],c[2],c[3],c[4],c[5])

                if C not in self.unique_motions:    self.unique_motions.append(C)
                if C not in self.total_motion[i-1]:   self.total_motion[i-1][C] = 1
                else:                               self.total_motion[i-1][C] += 1
        self.all_total_motion[self.scene] = self.total_motion

    #--------------------------------------------------------------------------------------------------------#
    # compute uniqe direction for every scene
    def _compute_unique_direction(self):
        self.unique_direction = []
        for i in self.dir_touch_m_i:
            if i not in self.unique_direction:  self.unique_direction.append(i)
        for i in self.dir_touch_m_f:
            if i not in self.unique_direction:  self.unique_direction.append(i)

    #--------------------------------------------------------------------------------------------------------#
    # compute unique location for the moving object
    def _compute_unique_location(self):
        self.unique_locations = []
        for i in self.locations_m_i:
            if i not in self.unique_locations:  self.unique_locations.append(i)
        for i in self.locations_m_f:
            if i not in self.unique_locations:  self.unique_locations.append(i)

    #--------------------------------------------------------------------------------------------------------#
    # making simulation as real world
    def _convert_color_shape_location_to_gmm(self,plot):
        self.real = 0
        unique_colors = []
        unique_shapes = []
        unique_locations = []
        self.gmm_M = {}

        # more real simulation but more time !
        if self.real:
            for i in self.Data:
                if i != 'G':
                    for k in range(50):
                        c = self.Data[i]['color']
                        r = np.random.normal(0, .005, 3)
                        #c += r
                        hsv = colorsys.rgb_to_hsv(c[0], c[1], c[2])
                        r1 = np.random.normal(0, .06, 1)
                        r2 = np.random.normal(0, .05, 1)
                        r3 = np.random.normal(0, .04, 1)
                        h = hsv[0]*2*np.pi  + r1[0]
                        s = hsv[1]          + r2[0]
                        v = hsv[2]          + r3[0]
                        if s<0:      s+=2*np.abs(r2[0])
                        if s>1:      s-=2*np.abs(r2[0])
                        if v<0:      v+=2*np.abs(r3[0])
                        if v>1:      v-=2*np.abs(r3[0])
                        x,y,z = self.hsv2xyz(h,s,v)
                        C = (x[0],y[0],z[0])
                        #print c,[h,s,v],C
                        s = self.Data[i]['shape']
                        r = np.random.normal(0, .03, 1)
                        s += r[0]
                        if s<0:      s+=2*np.abs(r[0])
                        if s>1:      s-=2*np.abs(r[0])
                        unique_colors.append(C)
                        unique_shapes.append(s)
        else:
            # less real simulation but more quick
            for color in self.unique_colors:
                for k in range(10):
                    r = np.random.normal(0, .005, 3)
                    c = color+r
                    if c[0]<0:      c[0]=0
                    if c[1]<0:      c[1]=0
                    if c[2]<0:      c[2]=0
                    if c[0]>1:      c[0]=1
                    if c[1]>1:      c[1]=1
                    if c[2]>1:      c[2]=1
                    C = (c[0],c[1],c[2])
                    """
                    hsv = colorsys.rgb_to_hsv(c[0], c[1], c[2])
                    r1 = np.random.normal(0, .06, 1)
                    r2 = np.random.normal(0, .05, 1)
                    r3 = np.random.normal(0, .04, 1)
                    h = hsv[0]*2*np.pi  + r1[0]
                    s = hsv[1]          + r2[0]
                    v = hsv[2]          + r3[0]
                    if s<0:      s+=2*np.abs(r2[0])
                    if s>1:      s-=2*np.abs(r2[0])
                    if v<0:      v+=2*np.abs(r3[0])
                    if v>1:      v-=2*np.abs(r3[0])
                    x,y,z = self.hsv2xyz(h,s,v)
                    C = (x[0],y[0],z[0])
                    """
                    unique_colors.append(C)
            for shape in self.unique_shapes:
                for k in range(10):
                    r = np.random.normal(0, .005, 1)
                    s = r[0]+shape
                    if s<0:      s+=2*np.abs(r[0])
                    if s>1:      s-=2*np.abs(r[0])
                    unique_shapes.append(s)
            # just a note to should reverce x and y in testing
            for i,l in enumerate(self.unique_locations):
                    l = [l[0],l[1]]
                    for k in range(10):
                        r = np.random.normal(0, .005, 2)
                        l += r
                        for j,l1 in enumerate(l):
                            if l1<0:      l[j]+=2*np.abs(r[j])
                            if l1>7:      l[j]-=2*np.abs(r[j])
                        L = (l[0]/7.0,l[1]/7.0)
                        unique_locations.append(L)

        gmm_c = self._bic_gmm(unique_colors,plot)
        gmm_s = self._bic_gmm(unique_shapes,plot)
        gmm_l = self._bic_gmm(unique_locations,plot)
        self.gmm_M['color'] = copy.deepcopy(gmm_c)
        self.gmm_M['shape'] = copy.deepcopy(gmm_s)
        self.gmm_M['location'] = copy.deepcopy(gmm_l)

    #-------------------------------------------------------------------------------------#
    def hsv2xyz(self,H, S, V):
    	x = [S*np.cos(H)]
    	y = [S*np.sin(H)]
    	z = [V]
    	return x,y,z

    #--------------------------------------------------------------------------------------------------------#
    # To Do !
    def _convert_direction_to_gmm(self,plot):
        print self.unique_direction

    #--------------------------------------------------------------------------------------------------------#
    # sub function for _convert_color_shape_location_to_gmm
    def _bic_gmm(self,points,plot):
        gmm1 = {}
        lowest_bic = np.infty
        bic = []
        X = []
        for point in points:
            if X == []:         X = [point]
            else:               X = np.vstack([X,point])
        k = np.minimum(len(X),7)
        best_gmm, bic = gmm_bic(X, k, self.cv_types)
        if plot:
            plot_data(X, best_gmm, bic, k, self.cv_types,0, 1)
            plt.show()
            plt.savefig("/home/omari/Datasets/robot_modified/clusters/" + str(len(X[0])) + "-" + str(self.scene) +".png")

        for i, (mean, covar) in enumerate(zip(best_gmm.means_, best_gmm._get_covars())):
            gmm1[i] = {}
            gmm1[i]['mean'] = mean
            gmm1[i]['covar'] = covar
            gmm1[i]['N'] = 1.0
        return gmm1

    #--------------------------------------------------------------------------------------------------------#
    def _build_obj_hyp_igmm(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                #if word == 'green':
                    if word not in self.gmm_obj:
                        self.gmm_obj[word] = {}
                        for f in self.gmm_M:
                            self.gmm_obj[word][f] = {}
                            self.gmm_obj[word][f]['N']    = 1.0
                            self.gmm_obj[word][f]['gmm']  = copy.deepcopy(self.gmm_M[f])
                    else:
                        for f in self.gmm_M:
                            gmm_N   = dict(self.gmm_obj[word][f]['gmm'])
                            gmm_M   = dict(self.gmm_M[f])
                            N       = self.gmm_obj[word][f]['N']
                            M       = 1.0
                            #============================#
                            # with compition between gmms
                            accepted_ind = []
                            smallest_matrix = np.zeros((len(gmm_N),len(gmm_M)),dtype=np.float)
                            for i in gmm_N:
                                mean_i = gmm_N[i]['mean']
                                covar_i = gmm_N[i]['covar']
                                m_min = np.inf
                                m_ind = []
                                for j in gmm_M:
                                    mean_j = gmm_M[j]['mean']
                                    dis = self._distance_test(mean_i, mean_j)
                                    smallest_matrix[i,j] = dis
                                    #dis1 = self.Mean_Test(mean_j,mean_i,covar_i)
                                    #smallest_matrix[i,j] = dis1
                            for k in range(np.min((len(gmm_N),len(gmm_M)))):
                                #print smallest_matrix
                                i,j = np.unravel_index(smallest_matrix.argmin(), smallest_matrix.shape)
                                m_max = smallest_matrix[i,j]
                                #print m_max
                                if m_max < self.pass_distance:               # == 75% matching in the t-test
                                    # - .04*(1-(1.0/np.exp(15.0/N)))
                                    gmm_N = self._update_gmm(gmm_N, gmm_M, i, j, N, M, 1)
                                    if j not in accepted_ind:   accepted_ind.append(j)
                                    gmm_N[i]['N'] += 1.0
                                    smallest_matrix[i,:] = +np.inf  # row
                                    smallest_matrix[:,j] = +np.inf  # col
                            #print accepted_ind
                            """
                            #============================#
                            # no competition between the different gmms
                            accepted_ind = []
                            for i in gmm_N:
                                mean_i = gmm_N[i]['mean']
                                m_min = np.inf
                                m_ind = []
                                for j in gmm_M:
                                    mean_j = gmm_M[j]['mean']
                                    dis = np.sqrt(np.sum((mean_i-mean_j)**2))
                                    #print f,i,j,dis
                                    if dis < m_min:
                                        m_min = dis
                                        m_ind = j
                                if m_min < .12:               # == 88% matching in the t-test
                                    gmm_N = self._update_gmm(gmm_N, gmm_M, i, m_ind, N, M, 1)
                                    if m_ind not in accepted_ind:   accepted_ind.append(m_ind)
                                    gmm_N[i]['N'] += 1.0
                            """
                            for key in gmm_M.keys():
                                if key not in accepted_ind:
                                    # we have to add this gmm to the hypotheses list with N = 1
                                    gmm_N = self._add_gmm(gmm_N, gmm_M, key)
                            #print 'new clusters',len(accepted_ind)-len(gmm_M.means_)
                            self.gmm_obj[word][f]['gmm'] = copy.deepcopy(gmm_N)
                            self.gmm_obj[word][f]['N'] += M

    #--------------------------------------------------------------------------------------------------------#
    # 3.2 Testing for equality to a mean vector
    # To Do sub function for _build_obj_hyp_igmm
    def Mean_Test(self, x_mean, mean, S):
        # compute sample mean
        d = len(x_mean)
        n = 10
        # compute the sample covariance
        S_inv = np.linalg.inv(S)
        # computing the T squared test
        c1 = np.transpose([mean - x_mean])
        T = n*np.dot(np.dot(np.transpose(c1),S_inv),c1)
        F = T[0][0]*float(n-d)/float(d*(n-1))
        #alpha = .005 #Or whatever you want your alpha to be.
        p_value = scipy.stats.f.pdf(F, d, n-d)
        #result = 0
        #if p_value>alpha:
            #result = 1.0
    	return p_value

    #--------------------------------------------------------------------------------------------------------#
    # sub funtion to compute counting probability
    def _probability(self,count,value):
        P = (value/count)*(1.0/np.exp(self.p_parameter/count))
        return P

    #--------------------------------------------------------------------------------------------------------#
    # 3.3.1 Merging Components
    def _update_gmm(self,gmm1, gmm2, j, k, N, M, Mk):
        mu_j = gmm1[j]['mean']
        S_j = gmm1[j]['covar']
        pi_j = 1
        mu_k = gmm2[k]['mean']
        S_k = gmm2[k]['covar']
        pi_k = 1
        # update the mean
        mu = ( N*pi_j*mu_j + Mk*mu_k )/( N*pi_j + Mk )
        # update the covariance matrix
        mu = np.array(mu)[np.newaxis]
        mu_j = np.array(mu_j)[np.newaxis]
        mu_k = np.array(mu_k)[np.newaxis]
        S = ((N*pi_j*S_j + Mk*S_k)/(N*pi_j+Mk)) + ((N*pi_j*np.dot(mu_j.T,mu_j)+Mk*np.dot(mu_k.T,mu_k))/(N*pi_j+Mk)) - np.dot(mu.T,mu)
        # update the weight
        pi = 1
        # update gmm1
        gmm1[j]['mean'] = mu[0]
        gmm1[j]['covar'] = S
        return gmm1

    #--------------------------------------------------------------------------------------------------------#
    # 3.3.1 Adding Components
    def _add_gmm(self, gmm1, gmm2, i):
            new_key = np.max(gmm1.keys())+1
            gmm1[new_key] = gmm2[i]
            return gmm1

    #--------------------------------------------------------------------------------------------------------#
    def _build_relation_hyp(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                # not_ok = 1
                # for w in word.split(' '):
                #     if w in ['top','over','on','above','placed','front','sitting','onto','underneath','infront','besides','resting','sit','sits','topmost','ontop','sat','stood','higher','downwords']:
                #         not_ok = 0
                if word not in self.hyp_relation:
                    self.hyp_relation[word] = {}
                    self.hyp_relation[word]['count'] = 0
                    self.hyp_relation[word]['direction'] = {}
                # if self.unique_direction != []:
                self.hyp_relation[word]['count'] += 1
                for direction in self.unique_direction:
                    if direction not in self.hyp_relation[word]['direction']:   self.hyp_relation[word]['direction'][direction] = 1
                    else: self.hyp_relation[word]['direction'][direction] += 1

    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this one has a hack, please make sure to clear it.
    def _build_motion_hyp(self):
        for s in self.phrases:
            for word in self.phrases[s]:
                not_ok = 1
                for w in word.split(' '):
                    if w in ['pick','place','put','up','down','move','put','shift','drop','take','remove']:
                        not_ok = 0
                if word not in self.hyp_motion:
                    self.hyp_motion[word] = {}
                    self.hyp_motion[word]['count'] = 0
                    self.hyp_motion[word]['motion'] = {}
                if not_ok:              continue
                if self.unique_motions != []:         self.hyp_motion[word]['count'] += 1
                for motion in self.unique_motions:
                    if motion not in self.hyp_motion[word]['motion']:
                        self.hyp_motion[word]['motion'][motion] = 1
                    else: self.hyp_motion[word]['motion'][motion] += 1

    #--------------------------------------------------------------------------------------------------------#
    def _save_all_features(self):
        self.all_scene_features[self.scene] = {}
        self.all_scene_features[self.scene]['color'] = self.unique_colors
        self.all_scene_features[self.scene]['shape'] = self.unique_shapes
        self.all_scene_features[self.scene]['direction'] = self.unique_direction
        self.all_scene_features[self.scene]['location'] = self.unique_locations

    #--------------------------------------------------------------------------------------------------------#
    def _test_relation_hyp(self):
        self.hyp_relation_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    count = float(self.hyp_relation[word]['count'])
                    for j in self.hyp_relation[word]:
                        if j != 'count':
                            for k in self.hyp_relation[word][j]:
                                prob = self._probability(count,self.hyp_relation[word][j][k])
                                if prob>self.p_relation_pass:
                                    if word not in self.hyp_relation_pass:
                                        self.hyp_relation_pass[word] = {}
                                        self.hyp_relation_pass[word]['possibilities'] = 0
                                    if j not in self.hyp_relation_pass[word]:
                                        self.hyp_relation_pass[word][j] = {}
                                    self.hyp_relation_pass[word]['possibilities'] += 1
                                    self.hyp_relation_pass[word][j][k] = prob

    #--------------------------------------------------------------------------------------------------------#
    def _test_motion_hyp(self):
        self.hyp_motion_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    count = float(self.hyp_motion[word]['count'])
                    for j in self.hyp_motion[word]:
                        if j != 'count':
                            for k in self.hyp_motion[word][j]:
                                prob = self._probability(count,self.hyp_motion[word][j][k])
                                if prob>self.p_relation_pass:
                                    if word not in self.hyp_motion_pass:
                                        self.hyp_motion_pass[word] = {}
                                        self.hyp_motion_pass[word]['possibilities'] = 0
                                    if j not in self.hyp_motion_pass[word]:
                                        self.hyp_motion_pass[word][j] = {}
                                    self.hyp_motion_pass[word]['possibilities'] += 1
                                    self.hyp_motion_pass[word][j][k] = prob

    #--------------------------------------------------------------------------------------------------------#
    def _test_obj_hyp(self):
        hyp_language_pass = {}
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    for f in self.gmm_obj[word]:
                        N = self.gmm_obj[word][f]['N']
                        for i in self.gmm_obj[word][f]['gmm']:
                            value = self.gmm_obj[word][f]['gmm'][i]['mean']
                            count = self.gmm_obj[word][f]['gmm'][i]['N']
                            p = self._probability(N,count)
                            if p >self.p_obj_pass:
                                #print f,'>>>',word,value,p
                                if word not in hyp_language_pass:
                                    hyp_language_pass[word] = {}
                                    hyp_language_pass[word]['possibilities'] = 1
                                else:
                                    hyp_language_pass[word]['possibilities'] += 1
                                if f not in hyp_language_pass[word]:
                                    hyp_language_pass[word][f] = {}
                                hyp_language_pass[word][f][tuple(value)] = p
        #for word in hyp_language_pass:
        self.hyp_obj_pass = copy.deepcopy(hyp_language_pass)

    #--------------------------------------------------------------------------------------------------------#
    def _combine_language_hyp(self):
        ### this is just a test
        self.hyp_language_pass = {}
        self.hyp_all_features = []
        self._combine_hyp(self.hyp_language_pass,self.hyp_obj_pass)
        self._combine_hyp(self.hyp_language_pass,self.hyp_relation_pass)
        self._combine_hyp(self.hyp_language_pass,self.hyp_motion_pass)


    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this will pass all hypotheses that are .9 of maximum value
    def _combine_hyp(self,A,B):
        ### adding two hypotheses A = A+B

        for word in B:
            if word not in A:
                A[word] = {}
                A[word]['possibilities'] = 0
            if word in A:
                for f in B[word]:
                    if f != 'possibilities':
                        if f not in self.hyp_all_features:      self.hyp_all_features.append(f)
                        if len(B[word][f]) > 1:
                            C = copy.deepcopy(B[word][f])
                            maxval = max(C.iteritems(), key=operator.itemgetter(1))[1]
                            keys = [k for k,v in C.items() if v>.9*maxval]
                            A[word][f] = {}
                            for key in keys:
                                A[word]['possibilities'] += 1
                                A[word][f][key] = C[key]
                        else:
                            A[word][f] = copy.deepcopy(B[word][f])
                            A[word]['possibilities'] += 1

    #--------------------------------------------------------------------------------------------------------#
    # NOTE: this will pass all hypotheses that are .9 of maximum value expect location
    def _filter_hyp(self):
        for word in self.hyp_language_pass:
            max_hyp = 0
            keys_remove = []
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities':
                    # find the max hypotheses
                    for key in self.hyp_language_pass[word][f].keys():
                        if self.hyp_language_pass[word][f][key]>max_hyp:            max_hyp = self.hyp_language_pass[word][f][key]

            # find all hypotheses that are within .9 of maximum hyp
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities' and f!= 'location':
                    for key in self.hyp_language_pass[word][f].keys():
                        if self.hyp_language_pass[word][f][key]<.9*max_hyp:         keys_remove.append([f,key])
            for A in keys_remove:
                f=A[0]
                key=A[1]
                self._remove_phrase(word, f, key)


    #--------------------------------------------------------------------------------------------------------#
    # remove sub phrases and bigger phrases based on their meanings
    # NOTE: this will remove sub phrases
    def _filter_phrases(self):
        phrases_to_remove = {}                                                     # this contains the list of phrases that are already described in smaller phrases
        hyp = self.hyp_language_pass
        checked = []
        for s in self.phrases:
            for word in self.phrases[s]:
                if word not in checked:
                    checked.append(word)
                    if word in hyp:
                        words = self._get_phrases(word)
                        for i in words:
                            for feature in hyp[word]:
                                if feature == 'possibilities': continue
                                for p1 in hyp[word][feature]:
                                    score = []
                                    matching = {}
                                    for sub_phrase in words[i]:
                                        score.append(0)
                                        if sub_phrase in hyp:
                                            if feature in hyp[sub_phrase]:
                                                for p2 in hyp[sub_phrase][feature]:
                                                    m1 = np.asarray(list(p1))
                                                    m2 = np.asarray(list(p2))
                                                    if len(m1) != len(m2):          continue        # motions !
                                                    if self._distance_test(m1,m2)<self.pass_distance_phrases:
                                                        score[-1] = 1
                                                        matching[sub_phrase] = p2
                                    N = float(np.sum(score))/float(len(words[i]))
                                    #print N
                                    # case 1 if N == 1 it means I should remove sub phrases
                                    # case 2 if N == 0 I should keep everything
                                    # case 3 if N < 1  I should remove phrase
                                    if N == 1:
                                        for key in matching:
                                            pass
                                             #if key not in phrases_to_remove:            phrases_to_remove[key] = {}
                                             #if feature not in phrases_to_remove[key]:   phrases_to_remove[key][feature] = []
                                             #if matching[key] not in phrases_to_remove[key][feature]:
                                            #    phrases_to_remove[key][feature].append(matching[key])
                                    elif N > 0:
                                        #print '##########################################################################',word,p1
                                        if word not in phrases_to_remove:               phrases_to_remove[word] = {}
                                        if feature not in phrases_to_remove[word]:      phrases_to_remove[word][feature] = []
                                        if p1 not in phrases_to_remove[word][feature]:
                                            phrases_to_remove[word][feature].append(p1)
        #print phrases_to_remove.keys()
        for word in phrases_to_remove:
            for feature in phrases_to_remove[word]:
                for key in phrases_to_remove[word][feature]:
                    self._remove_phrase(word, feature, key)

    #--------------------------------------------------------------------------------------------------------#
    # sub function
    def _remove_phrase(self, p, f, p_remove):
        # p is the word
        # f is the feature
        # p_remove is the key in the feature in the word
        # remove a phrase from self.hyp_language_pass
        self.hyp_language_pass[p][f].pop(p_remove, None)
        if len(self.hyp_language_pass[p][f].keys()) == 0:
            self.hyp_language_pass[p].pop(f,None)
        self.hyp_language_pass[p]['possibilities'] -= 1
        if self.hyp_language_pass[p]['possibilities'] == 0:
            self.hyp_language_pass.pop(p,None)

    #--------------------------------------------------------------------------------------------------------#
    def _distance_test(self,m1,m2):
        return np.sqrt(np.sum((m1-m2)**2))/np.sqrt(len(m1))

    #--------------------------------------------------------------------------------------------------------#
    # sub function to get all possible phrases that can make a certain phrase
    def _get_phrases(self,word):
        # get all possible combination of sub phrases for a phrase
        words = {}
        w = word.split(' ')
        n = len(w)
        if n == 2:
            words[0] = [w[0],w[1]]
        elif n == 3:
            words[0] = [w[0],w[1],w[2]]
            words[1] = [' '.join(w[0:2]),w[2]]
            words[2] = [w[0],' '.join(w[1:3])]
        return words

    #--------------------------------------------------------------------------------------------------------#
    # generate all possible combinitaiton for each phrase in a single list
    def _get_all_valid_combinations(self):
        for word in self.hyp_language_pass:
            self.hyp_language_pass[word]['all'] = []
            for f in self.hyp_language_pass[word]:
                if f != 'all' and f != 'possibilities':
                    for k in self.hyp_language_pass[word][f]:
                        self.hyp_language_pass[word]['all'].append((f,k))

    #--------------------------------------------------------------------------------------------------------#
    # generate all possible sentences that have hypotheses in language hypotheses pass
    def _test_all_valid_combinations(self):
        # test the hypotheses sentence by sentence
        self.valid_combination = {}
        if 'motion' in self.hyp_all_features:
            self._get_indices()
            for scene in self.phrases:
                self.valid_combination[scene] = {}
                #print scene
                # get the words that have hypotheses and are in the sentence
                phrases_with_hyp = list(set(self.hyp_language_pass.keys()).intersection(self.phrases[scene]))

                # generate all subsets (pick from 1 word to n words) with no repatetion in phrases
                for L in range(2, len(phrases_with_hyp)+1):
                    self.valid_combination[scene][L] = zip(*self.pool.map(calc, [[subset,self.indices[scene],self.hyp_language_pass,self.all_total_motion[self.scene],self.S[scene],self.all_scene_features[self.scene],[self.G_i,self.G_f],scene,L] for subset in itertools.combinations(phrases_with_hyp, L)]))
                    # for subset in itertools.combinations(phrases_with_hyp, L):
                    #     out1 = calc([subset,self.indices[scene],self.hyp_language_pass,self.all_total_motion[self.scene],self.S[scene],self.all_scene_features[self.scene],[self.G_i,self.G_f],scene,L])
                        #
                        # for o1 in out1[0]:
                        #     if o1 != []:
                        #         print '-------',o1

    #------------------------------------------------------------------#
    # get the index of every phrase in every sentence
    def _get_indices(self):
        self.indices = {}
        for scene in self.phrases:
            self.indices[scene] = {}
            sentence = self.S[scene].split(' ')
            # build the indices
            phrases_with_hyp = list(set(self.hyp_language_pass.keys()).intersection(self.phrases[scene]))
            for word in phrases_with_hyp:
                w = word.split(' ')
                if len(w) == 1:
                    A = [[i] for i, x in enumerate(sentence) if x == word]
                    self.indices[scene][word] = A
                if len(w) == 2:
                    ind = {}
                    for wc,w1 in enumerate(w):
                        ind[wc] = []
                        [ind[wc].append(i) for i, x in enumerate(sentence) if x == w1]
                    for i1 in ind[0]:
                        for i2 in ind[1]:
                            if i2-i1 == 1:
                                if word not in self.indices[scene]:
                                    self.indices[scene][word] = []
                                self.indices[scene][word].append([i1,i2])
                if len(w) == 3:
                    ind = {}
                    for wc,w1 in enumerate(w):
                        ind[wc] = []
                        [ind[wc].append(i) for i, x in enumerate(sentence) if x == w1]
                    for i1 in ind[0]:
                        for i2 in ind[1]:
                            for i3 in ind[2]:
                                if i2-i1 == 1 and i3-i2 == 1:
                                    if word not in self.indices[scene]:
                                        self.indices[scene][word] = []
                                    self.indices[scene][word].append([i1,i2,i3])
        #for s in self.indices:
        #    print s,self.indices[s]

    #------------------------------------------------------------------#
    def _test(self,subset,scene):

        # no 2 phrases are allowed to intersect in the same sentence
        no_intersection = 1
        all_indeces = []
        for w in subset:
            #print w
            for w1 in self.indices[scene][w]:
                for w2 in w1:
                    if w2 not in all_indeces:
                        all_indeces.append(w2)
                    else:
                        no_intersection = 0
        #print no_intersection
        #print '------'

        if no_intersection:
            # this function tests one susbet of words at a time
            all_possibilities = []      # all the possibilities gathered in one list
            for word in subset:
                all_possibilities.append(self.hyp_language_pass[word]['all'])

            #print all_possibilities
            #print '---------------',len(all_possibilities)
            # find the actual possibilities for every word in the subset
            #c = 1
            for element in itertools.product(*all_possibilities):
                #print c
                #c+=1
                hyp_motion = {}
                motion_pass = 0

                # no 2 words are allowed to mean the same thing
                not_same = 1
                features = {}
                for i in element:
                    if i[0] not in features: features[i[0]] = []
                    features[i[0]].append(i[1])

                for f in features:
                    if len(features[f])>1:
                        for f1 in range(len(features[f])-1):
                            for f2 in range(f1+1,len(features[f])):
                                m1 = np.asarray(list(features[f][f1]))
                                m2 = np.asarray(list(features[f][f2]))
                                if len(m1) != len(m2):          continue        # motions !
                                if self._distance_test(m1,m2)<self.pass_distance:
                                    not_same = 0
                                    continue
                if not_same:
                    # 1) does actions match ?   it should match 100%
                    for k,word in enumerate(subset):
                        if element[k][0] == 'motion':
                            a = element[k][1]
                            if a not in hyp_motion:     hyp_motion[a] = len(self.indices[scene][word])
                            else:                       hyp_motion[a] += len(self.indices[scene][word])
                    for i in self.total_motion:
                        if self.total_motion[i] == hyp_motion:
                            motion_pass = 1
                        #else: print self.scene,'fail'

                    # 2) parse the sentence
                    if motion_pass:
                        parsed_sentence = []
                        value_sentence = []
                        for i in self.S[scene].split(' '):
                            parsed_sentence.append('_')
                            value_sentence.append('_')
                        for word1 in subset:
                            for i1 in self.indices[scene][word1]:
                                for j1 in i1:
                                    #print word1,j1
                                    k = subset.index(word1)
                                    parsed_sentence[j1] = element[k][0]
                                    value_sentence[j1] = map(prettyfloat, element[k][1])
                        print subset
                        print self.S[scene].split(' ')
                        print parsed_sentence
                        print value_sentence
                        print '----'

    #--------------------------------------------------------------------------------------------------------#
    # takes the valid hypotheses and build the grammar
    def _build_grammar(self):
        self.max_L = {}
        for scene in self.valid_combination:
            self.max_L[scene] = -1
            for L in self.valid_combination[scene]:
                # print scene,L
                for i in self.valid_combination[scene][L]:
                    for j in i:
                        for k in j:
                            if k != []:
                                if L>self.max_L[scene]:
                                    self.max_L[scene] = L
                                # for k1 in k:
                                #     print self.S[scene]
                                #     print 'Scene:',scene,'L:',L,k1
                                #     print '******'
        self._update_terminals()
        self._update_nonterminals()
        self._build_PCFG()

    #--------------------------------------------------------------------------------------------------------#
    def _update_terminals(self):
        hypotheses = self.hyp_language_pass
        for scene in self.valid_combination:
            #for L in self.valid_combination[scene]:
            L = self.max_L[scene]
            if L>0:
                for i in self.valid_combination[scene][L]:
                    for j in i:
                        for k in j:
                            if k != []:
                                for k1 in k:
                                    order   = k1[0]
                                    words   = k1[3]
                                    values  = k1[4]
                                    for word,value in zip(words,values):
                                        feature = value[0]
                                        if value[0] == 'motion':
                                            if len(order)==2:
                                                feature = value[0]+'_'+order[0]+order[1]
                                            if len(order)==1:
                                                feature = value[0]+'_'+order[0]
                                        val = hypotheses[word][value[0]][value[1]]
                                        if feature not in self.T['features']:
                                            self.T['features'][feature] = {}
                                            self.T['sum'][feature]      = 0.0
                                        if len(word.split(' '))>1:
                                            words = word.split(' ')
                                            word = '_'+'_'.join(words)
                                            if word not in self.T['features'][feature]:
                                                self.T['features'][feature][word] = 0.0
                                            self.T['features'][feature][word] += val
                                            self.T['sum'][feature] += val

                                            for w in words:
                                                if w not in self.no_match['features']:
                                                    self.no_match['features'][w] = {}
                                                    self.no_match['sum'][w]      = 1.0
                                                    self.no_match['features'][w][w] = 1.0

                                            if word not in self.N:
                                                self.N[word] = {}
                                                self.N[word][' '.join(words)] = 1.0
                                                self.N[word]['sum'] = 1.0

                                            # if feature not in self.N:
                                            #     self.N[feature] = {}
                                            #     self.N[feature]['sum'] = 0.0
                                            # if word not in self.N[feature]:
                                            #     self.N[feature][word] = 0.0
                                            # self.N[feature][word] += 1.0
                                            # self.N[feature]['sum'] += 1.0
                                        else:
                                            if word not in self.T['features'][feature]:
                                                self.T['features'][feature][word] = 0.0
                                            self.T['features'][feature][word] += val
                                            self.T['sum'][feature] += val

    #--------------------------------------------------------------------------------------------------------#
    def _update_nonterminals(self,):
        # there is a threshold in NLTK to drop a hypotheses 9.99500249875e-05 I think it;s 1e-4
        entity = ['shape','color','location']
        relation = ['direction']
        for scene in self.valid_combination:
            #for L in self.valid_combination[scene]:
            L = self.max_L[scene]
            if L>0:
                for i in self.valid_combination[scene][L]:
                    for j in i:
                        for k in j:
                            if k != []:
                                for k1 in k:
                                    order   = k1[0]
                                    TE      = k1[1]
                                    TV      = k1[2]
                                    subset   = k1[3]
                                    element  = k1[4]
                                    PS      = k1[5]
                                    ba      = k1[6]
                                    verb                    = k1[7]
                                    scene_description       = k1[8]
                                    for i1,i2 in zip(subset,element):
                                        if i2[0]=='motion' and i2[1]==verb:
                                            verb_name = i1
                                            if len(order)==2:
                                                TETV_grammar = order[0]+'_'+order[1]
                                                verb_grammar = i2[0]+'_'+order[0]+order[1]
                                            if len(order)==1:
                                                TETV_grammar = order[0]
                                                verb_grammar = i2[0]+'_'+order[0]

                                    # print 'target E          :',TE
                                    # print 'target V          :',TV
                                    # print 'Parsed S          :',PS
                                    # print 'Scene description :',scene_description
                                    # print 'before/after      :',ba
                                    # print 'motion            :',verb
                                    # print 'motion name       :',verb_name
                                    # print 'features          :',verb_grammar
                                    # print '*****'

                                    #building the sentence level
                                    if 'S' not in self.N:
                                        self.N['S'] = {}
                                        self.N['S']['sum'] = 0.0

                                    if ba == 'after':
                                        S1 = verb_grammar+' '+TETV_grammar
                                    if ba == 'before':
                                        S1 = TETV_grammar+' '+verb_grammar
                                    if S1 not in self.N['S']:
                                        self.N['S'][S1] = 1.0
                                    else:
                                        self.N['S'][S1] += 1.0
                                    self.N['S']['sum'] += 1.0

                                    #building the target entity and target value
                                    if TETV_grammar not in self.N:
                                        self.N[TETV_grammar] = {}
                                        self.N[TETV_grammar]['sum'] = 0.0

                                    if len(order)==2:
                                        connecter = []
                                        if order[0] == 'TE':
                                            for l in reversed(range(len(TE))):
                                                if TE[l] in entity or TE[l] in relation:
                                                    break
                                            TE_f = TE[0:l+1]
                                            for l1 in TE[l+1:len(TE)]:
                                                connecter.append(l1)
                                            for l in range(len(TV)):
                                                if TV[l] in entity or TV[l] in relation:
                                                    break
                                            TV_f = TV[l:len(TV)]
                                            for l1 in TV[0:l]:
                                                connecter.append(l1)
                                        if order[0] == 'TV':
                                            for l in reversed(range(len(TV))):
                                                if TV[l] in entity or TV[l] in relation:
                                                    break
                                            TV_f = TV[0:l+1]
                                            for l1 in TV[l+1:len(TV)]:
                                                connecter.append(l1)
                                            for l in range(len(TE)):
                                                if TE[l] in entity or TE[l] in relation:
                                                    break
                                            TE_f = TE[l:len(TE)]
                                            for l1 in TE[0:l]:
                                                connecter.append(l1)

                                    # build TE and TV them self
                                    if len(order)==2:
                                        #print '>>>>',TE_f
                                        #print '>>>>',TV_f
                                        TE_converted,e, e_bar = self._TE_TV_conversion(TE_f)
                                        TV_converted,v, v_bar = self._TE_TV_conversion(TV_f)

                                        #---------------------------------#
                                        # TE = the _entity
                                        T = 'TE'
                                        if T not in self.N:
                                            self.N[T] = {}
                                            self.N[T]['sum'] = 0.0
                                        if TE_converted not in self.N[T]:
                                            self.N[T][TE_converted] = 1.0
                                        else:
                                            self.N[T][TE_converted] += 1.0
                                        self.N[T]['sum'] += 1.0

                                        T = 'TV'
                                        if T not in self.N:
                                            self.N[T] = {}
                                            self.N[T]['sum'] = 0.0
                                        if TV_converted not in self.N[T]:
                                            self.N[T][TV_converted] = 1.0
                                        else:
                                            self.N[T][TV_converted] += 1.0
                                        self.N[T]['sum'] += 1.0
                                        #---------------------------------#
                                        # _entity and _shape and _location
                                        for T,val in zip(e_bar,e):
                                            if T not in self.N:
                                                self.N[T] = {}
                                                self.N[T]['sum'] = 0.0
                                            value = ' '.join(val)

                                            if value not in self.N[T]:
                                                self.N[T][value] = 1.0
                                            else:
                                                self.N[T][value] += 1.0
                                            self.N[T]['sum'] += 1.0

                                        for T,val in zip(v_bar,v):
                                            if T not in self.N:
                                                self.N[T] = {}
                                                self.N[T]['sum'] = 0.0
                                            value = ' '.join(val)

                                            if value not in self.N[T]:
                                                self.N[T][value] = 1.0
                                            else:
                                                self.N[T][value] += 1.0
                                            self.N[T]['sum'] += 1.0

                                        #---------------------------------#
                                        # no meaning words in TE and TV
                                        for word in TE_converted.split(' '):
                                            if word[0] != '_':
                                                if word not in self.no_match['features']:
                                                    self.no_match['features'][word] = {}
                                                    self.no_match['sum'][word]      = 1.0
                                                    self.no_match['features'][word][word] = 1.0

                                        for word in TV_converted.split(' '):
                                            if word[0] != '_':
                                                if word not in self.no_match['features']:
                                                    self.no_match['features'][word] = {}
                                                    self.no_match['sum'][word]      = 1.0
                                                    self.no_match['features'][word][word] = 1.0

                                    if len(order)==2:
                                        if connecter != []:
                                            S1 = order[0]+' '+order[0]+order[1]+'_connect'+' '+order[1]
                                        else:
                                            S1 = order[0]+' '+order[1]
                                        if S1 not in self.N[TETV_grammar]:
                                            self.N[TETV_grammar][S1] = 1.0
                                        else:
                                            self.N[TETV_grammar][S1] += 1.0
                                    self.N[TETV_grammar]['sum'] += 1.0

                                    #connecter
                                    if connecter != []:
                                        C = order[0]+order[1]+'_connect'
                                        if C not in self.N:
                                            self.N[C] = {}
                                            self.N[C]['sum'] = 0.0
                                        con = ' '.join(connecter)
                                        if con not in self.N[C]:
                                            self.N[C][con]      = 0.0
                                        self.N[C][con]    += 1.0
                                        self.N[C]['sum']  += 1.0

                                        for i in connecter:
                                            if i not in self.no_match['features']:
                                                self.no_match['features'][i] = {}
                                                self.no_match['sum'][i]      = 1.0
                                                self.no_match['features'][i][i] = 1.0

    #--------------------------------------------------------------------------------------------------------#
    def _TE_TV_conversion(self,T):
        entity = ['shape','color','location']
        relation = ['direction']
        s = [[]]
        s_bar = []
        final_T = []
        for word in T:
            if word in entity or word in relation:
                if s[-1] == []:
                    s[-1] = [word]
                    if word in entity:
                        final_T.append('_'+word)
                        s_bar.append('_'+word)
                    if word in relation:
                        final_T.append('_'+word)
                        s_bar.append('_'+word)
                else:
                    if s[-1][-1] in entity:         #previous is entity
                        if word in entity       :   # new is entity
                            if word not in s[-1]:
                                s[-1].append(word)
                                final_T[-1] = '_entity'
                                s_bar[-1] = '_entity'
                            else:
                                s.append([word])
                                final_T.append('_'+word)
                                s_bar.append('_'+word)
                        if word in relation     :   #new is relation
                            s.append([word])
                            final_T.append('_'+word)
                            s_bar.append('_'+word)

                    elif s[-1][-1] in relation:       #previous is relation
                        if word in entity       :   #new is entity
                            s.append([word])
                            final_T.append('_'+word)
                            s_bar.append('_'+word)
                        if word in relation     :   #new is relation
                            if word not in s[-1]:
                                s[-1].append(word)
                                final_T[-1] = ('_relation')
                                s_bar[-1] = ('_relation')
                            else:
                                s.append([word])
                                final_T.append('_'+word)
                                s_bar.append('_'+word)
            else:
                final_T.append(word)
        return ' '.join(final_T), s, s_bar


    #--------------------------------------------------------------------------------------------------------#
    def _build_PCFG(self):
        self.grammar = ''
        # Non terminals
        for feature in self.N:
            if feature == 'S':
                sorted_f = sorted(self.N[feature].items(), key=operator.itemgetter(1))
                for l in range(len(sorted_f),0,-1):
                    hyp = sorted_f[l-1]
                    if hyp[0] != 'sum':
                        val = hyp[1]/self.N[feature]['sum']
                        if val > 1e-4:
                            self.grammar += feature+" -> "+hyp[0]+" ["+str(val)+"]"+'\n'

        # Non terminals
        for feature in self.N:
            if feature != 'S':
                sorted_f = sorted(self.N[feature].items(), key=operator.itemgetter(1))
                for l in range(len(sorted_f),0,-1):
                    hyp = sorted_f[l-1]
                    if hyp[0] != 'sum':
                        val = hyp[1]/self.N[feature]['sum']
                        if val > 1e-4:
                            self.grammar += feature+" -> "+hyp[0]+" ["+str(val)+"]"+'\n'

        # No match
        for feature in self.no_match['features']:
            for hyp in self.no_match['features'][feature]:
                val = self.no_match['features'][feature][hyp]
                self.grammar += feature+" -> '"+hyp+"' ["+str(val/self.no_match['sum'][feature])+"]"+'\n'

        # Terminals
        for feature in self.T['features']:
            for hyp in self.T['features'][feature]:
                val = self.T['features'][feature][hyp]
                if hyp[0] == '_':
                    self.grammar += feature+" -> "+hyp+" ["+str(val/self.T['sum'][feature])+"]"+'\n'
                else:
                    self.grammar += feature+" -> '"+hyp+"' ["+str(val/self.T['sum'][feature])+"]"+'\n'

        # PCFG
        print self.grammar
        if self.grammar != '':
            self.pcfg1 = PCFG.fromstring(self.grammar)
            print self.pcfg1

    #--------------------------------------------------------------------------------------------------------#
    def _print_results(self):
        print '====-----------------------------------------------------------------===='
        for word in self.hyp_language_pass:
            for f in self.hyp_language_pass[word]:
                if f != 'possibilities' and f != 'all':
                    for value in self.hyp_language_pass[word][f]:
                        if len(f) < 7:
                            print f,'\t\t>>>\t',
                        elif len(f) < 15:
                            print f,'\t>>>\t',
                        if len(word) < 7:
                            print word,'\t\t\t>>>\t',
                        elif len(word) < 15:
                            print word,'\t\t>>>\t',
                        elif len(word) < 23:
                            print word,'\t>>>\t',
                        print map(prettyfloat, value),
                        print("{0:.3f}".format( self.hyp_language_pass[word][f][value]))

    #--------------------------------------------------------------------------------------------------------#
    def _create_scene_graph(self):
        # Creating the graph structure
        G = nx.Graph()
        # creating the object layer
        m_count = 2.0                     # moving object location
        r_count = m_count+.5                 # 1 is reserved for the moving object
        obj_count = 3.0                 # 1 is reserved for the moving object
        G_count = 1.0
        for I in [0,-1]:

            for key in self.keys:
                #print key,self.Data[key]['color']
                if key == 'G':
                    G.add_node(str(key),type1='G',position=(G_count,3))
                else:
                    if key == self.m_obj:
                        G.add_node(str(key),type1='mo',position=(m_count,3))
                    else:
                        G.add_node(str(key),type1='o',position=(obj_count,3))
                        obj_count+=1
                    G.add_node(str(key)+'_color',type1='of',type2='color', value=self.Data[key]['color'],position=(m_count-.25,1));         #color
                    G.add_node(str(key)+'_shape',type1='of',type2='shape', value=self.Data[key]['shape'],position=(m_count,1));         #shape
                    x = self.Data[key]['x'][I]/7.0
                    y = self.Data[key]['y'][I]/7.0
                    G.add_node(str(key)+'_location',type1='of',type2='location', value=[x,y],position=(m_count+.25,1));         #location
                    G.add_edge(str(key),str(key)+'_color')
                    G.add_edge(str(key),str(key)+'_shape')
                    G.add_edge(str(key),str(key)+'_location')

            # creating the relation layer
            counter = 0
            for k1 in self.keys:
                for k2 in self.keys:
                    if k2 != k1 and k2 != 'G' and k1 != 'G':
                        #if k1 == self.m_obj:
                            G.add_node(str(k1)+'_'+str(k2),type1='r',position=(r_count,7.0))     # it's a directed node from k1 to k2
                            G.add_edge(str(k1)+'_'+str(k2),str(k1))
                            G.add_edge(str(k1)+'_'+str(k2),str(k2))
                            #G.add_node(str(k1)+'_'+str(k2)+'_dist',type1='rf',position=(r_count-.15,5));         #distance
                            #direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                            x1 = self.Data[k1]['x'][I]
                            y1 = self.Data[k1]['y'][I]
                            z1 = self.Data[k1]['z'][I]
                            x2 = self.Data[k2]['x'][I]
                            y2 = self.Data[k2]['y'][I]
                            z2 = self.Data[k2]['z'][I]
                            d = [x1-x2,y1-y2,z1-z2]
                            if np.abs(sum(np.abs(d)))<=1.1:
                                if d[0] != 0:   d[0] /= np.abs(d[0])
                                if d[1] != 0:   d[1] /= np.abs(d[1])
                                if d[2] != 0:   d[2] /= np.abs(d[2])
                            else:
                                d = [0,0,0]

                            G.add_node(str(k1)+'_'+str(k2)+'_dir',type1='rf',type2='direction',value=d,position=(r_count,5));                     #direction
                            #G.add_node(str(k1)+'_'+str(k2)+'_mot',type1='rf',position=(r_count+.15,5));                     #motion
                            #G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                            G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                            #G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
                            counter += 1
                            r_count += 1

                """
                    if k2 != k1 and k2 == 'G':
                        G.add_node(str(k1)+'_'+str(k2),type1='r',position=(G_count+.5,7.0))     # it's a directed node from k1 to k2
                        G.add_edge(str(k1)+'_'+str(k2),str(k1))
                        G.add_edge(str(k1)+'_'+str(k2),str(k2))
                        G.add_node(str(k1)+'_'+str(k2)+'_dist',type1='rf',position=(G_count+.5,5));         #distance
                        #direction = [dirx_m[counter,distance],diry_m[counter,distance],dirz_m[counter,distance]]
                        G.add_node(str(k1)+'_'+str(k2)+'_dir',type1='rf',position=(G_count+.5-.15,5));                     #direction
                        G.add_node(str(k1)+'_'+str(k2)+'_mot',type1='rf',position=(G_count+.5+.15,5));                     #motion
                        G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dist')
                        G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_dir')
                        G.add_edge(str(k1)+'_'+str(k2),str(k1)+'_'+str(k2)+'_mot')
                """
            if I == 0:          self.G_i = G.copy()
            if I == -1:         self.G_f = G.copy()






    #--------------------------------------------------------------------------------------------------------#
    def _plot_graphs(self):
        self.f,self.ax = plt.subplots(len(self.transition['all']),4,figsize=(14,10)) # first col motion , second distance
        self.f.suptitle('Scene : '+str(self.scene), fontsize=20)
        for feature in [0,2]:
            # plot the different graphs of motion and distance
            for sub,T in enumerate(self.transition['all']):
                plt.sca(self.ax[sub,feature])
                print 'plotting graph : '+str(sub+1)+' from '+str(len(self.transition['all']))
                if feature == 0:
                    if T not in self.transition['motion']:
                        for i in self.transition['motion']:
                            if i<T: t=i
                    else: t=T
                    G=self.G_motion[t]['graph']
                elif feature == 2:
                    if T not in self.transition['touch']:
                        for i in self.transition['touch']:
                            if i<T: t=i
                    else: t=T
                    G=self.G_touch[t]['graph']
                # layout graphs with positions using graphviz neato
                pos=nx.graphviz_layout(G,prog="neato")
                # color nodes the same in each connected subgraph
                C=nx.connected_component_subgraphs(G)
                cK = 0
                for i in C:  cK += 1
                C=nx.connected_component_subgraphs(G)
                colors = np.linspace(.2,.6,cK)
                for count,g in enumerate(C):
                    c=[colors[count]]*nx.number_of_nodes(g) # same color...
                    nx.draw(g,pos,node_size=80,node_color=c,vmin=0.0,vmax=1.0,with_labels=False)
                    #nx.draw_networkx_edges(g,pos, with_labels=False, edge_color=c[0], width=6.0, alpha=0.5)
                nx.draw_networkx_nodes(self.G,pos, node_color='b', node_size=100, alpha=1)
                nx.draw_networkx_nodes(self.G,pos, nodelist=['G'], node_color='r', node_size=100, alpha=1)
                nx.draw_networkx_nodes(self.G,pos, nodelist=[str(self.m_obj)], node_color='c', node_size=100, alpha=1)
                nx.draw_networkx_edges(G,pos, alpha=0.8)
                #nx.draw(G)  # networkx draw()
                self.ax[sub,feature].axis('on')
                self.ax[sub,feature].axis('equal')
                plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')
                if feature == 0:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('motion')
                if feature == 2:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('connectivity')

        #plt.draw() # display
        #plt.pause(.00001)
        #plt.show()

    #--------------------------------------------------------------------------------------------------------#
    def _plot_final_graph(self):
        for feature in [1,3]:
            for sub,T in enumerate(self.transition['all']):
                plt.sca(self.ax[sub,feature])

                G = self._create_moving_obj_graph()
                # Creating the group effect
                if feature == 1:
                    if T not in self.transition['motion']:
                        for i in self.transition['motion']:
                            if i<T: t=i
                    else: t=T
                    G_group = self.G_motion[t]['groups']
                if feature == 3:
                    if T not in self.transition['touch']:
                        for i in self.transition['touch']:
                            if i<T: t=i
                    else: t=T
                    G_group = self.G_touch[t]['groups']

                N = len(G_group)
                HSV = [(x*.9/N, 0.9, 0.9) for x in range(N)]
                RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV)
                for c,group in enumerate(G_group):
                    for node in group:
                        p = G.node[node]['position']
                        rect1 = mat.patches.Rectangle((p[0]-.4,p[1]-3), .8, 4, color=RGB[c],alpha=.5)
                        self.ax[sub,feature].add_patch(rect1)

                agents = list((n for n in G if G.node[n]['type1']=='G'))
                m_objects = list((n for n in G if G.node[n]['type1']=='mo'))
                objects = list((n for n in G if G.node[n]['type1']=='o'))
                objects_f = list((n for n in G if G.node[n]['type1']=='of'))
                relations = list((n for n in G if G.node[n]['type1']=='r'))
                relations_f = list((n for n in G if G.node[n]['type1']=='rf'))

                #pos=nx.graphviz_layout(G,prog="neato")
                pos = nx.get_node_attributes(G,'position')

                nx.draw_networkx_nodes(G,pos, nodelist=agents, node_color='r', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=m_objects, node_color='c', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=objects, node_color='b', node_size=100, alpha=1)
                nx.draw_networkx_nodes(G,pos, nodelist=objects_f, node_color='y', node_size=100, alpha=0.8)
                nx.draw_networkx_nodes(G,pos, nodelist=relations, node_color='g', node_size=100, alpha=0.8)
                nx.draw_networkx_nodes(G,pos, nodelist=relations_f, node_color='y', node_size=100, alpha=0.8)
                #nx.draw_networkx_edges(G,pos, with_labels=False, edge_color='r', width=6.0, alpha=0.5)
                nx.draw_networkx_edges(G,pos, alpha=0.8)

                if feature == 1:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('motion')

                if feature == 3:
                    self.ax[sub,feature].set_ylabel('frame : '+str(T))
                    if sub == 0:
                        self.ax[sub,feature].set_title('connectivity')

                self.ax[sub,2].axis('on')
                plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                plt.tick_params(axis='y',which='both',right='off',left='off',labelleft='off')



        plt.savefig(self.dir3+'_'+str(self.scene)+'.png',dpi=200) # save as png
        plt.show() # display

    #--------------------------------------------------------------------------------------------------------#
    def _plot_scene(self):
        import sys, select, os
        all_files = sorted(listdir(self.dir2+str(self.scene)+'/'))
        for i in all_files:
            img = cv2.imread(self.dir2+str(self.scene)+'/'+i)
            cv2.imshow('scene',img)
            cv2.waitKey(50) & 0xff
        plt.close(self.f)
