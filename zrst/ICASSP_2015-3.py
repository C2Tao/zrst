import sys
import shelve
import pyASR
import pySFX
import pyEXT
#from zrst import pyASR
#from zrst import pySFX
#from zrst import pyEXT
import os
import std
import util
import numpy as np
import cPickle as pickle
#######################phase1####################
# generate patterns
'''
### set paths ###
drpbox_path = r'/home/c2tao/Dropbox/'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'
target_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/addmix5034/'
#init generated by matlab/clusterDetection.m
intial_path = drpbox_path + r'ULT_v0.2/matlab/IDump_timit_test_50_flatten.txt'
history_path = drpbox_path+ r'Semester 9.5/'

S_list = [3,7,13]
M_list = [50,100,300]
for S in S_list:
    for M in M_list: 
        config_name = '5034_'+str(M)+'_'+str(S)
        ### continue interrupted work ###
        # declare object
        A = pyASR.ASR(corpus = corpus_path, target = target_path, nState = S)
        # select interrupted folder
        A.offset = 30
        A.readASR(history_path+str(A.offset)+'_'+config_name)
        # continue for some iterations
        A.iteration('a_mix',config_name)
        A.iteration('a_keep',config_name)
        A.iteration('a_mix',config_name)
        A.iteration('a_keep',config_name)
        A.iteration('a_mix',config_name)
        A.iteration('a_keep',config_name)
'''


#######################phase 2 ##################################
# fix_messy syntax issues for queries
'''

drpbox_path = r'/home/c2tao/Dropbox/'
querie_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/'
messie_path = drpbox_path + r'Semester 8.5/_readonly/'
pa = messie_path +  r'homophone_character.mlf'
pb = messie_path + r'homophone_time_missing.mlf'
pc = messie_path + r'querycontent_homo.txt'

fc = messie_path + r'fucked_up.mlf'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'


M = util.MLF(pa)
N = util.MLF(pb)

for i in range(len(M.wav_list)):
    w = M.wav_list[i]
    if w not in N.wav_list:
        print w
        print M.wav_list[i+1]

'''
"*/N200108091200-23-02.rec homophone_time_missing"
"""
*/N200108091200-23-02.rec
0 2200000 [AE61] 0
2200000 4400000 [B8CC] 0
4400000 6600000 [B351] 0
6600000 8800000 [B573] 0
8800000 11000000 [B8E9] 0
11000000 13200000 [AB49] 0
13200000 13500000 [A44A] 0
.
"""
#A = util.read_feature(querie_path + '5034_query_active/feature_file/N200108091200-23-02.mfc')
#print np.shape(A)

#with open(pc,'r') as pcf:
#    homo = pcf.readlines()
#query = map(lambda x: x.strip().replace(')(',' ')[1:-1].split(),homo)
#print query

#with open(querie_path+'q_list.txt','w') as f:
#    for q in query:
#        f.write(' '.join(q)+'\n')
#print N.tag_list[0]

#query = map(lambda x:tuple(x.strip().split()),open(querie_path+'q_list.txt','r').readlines())


#with open(messie_path+r'fucked_up.mlf','w') as ofile:
#    ofile.write('#!MLF!#\n')
#    for l in open(messie_path+r'corpuscontent_asc.mlf','r').readlines():
#        l = l.strip()
#        if l=='.':
#            ofile.write(l+'\n')
#        elif 'chn' in l:
#            ofile.write('"*/'+l.split('/')[-1]+'\n')
#        else:
#            l=l.replace(' ','').replace('][',']\n[')
#            ofile.write(l+'\n')

#W = util.MLF(fc)
#W.write(fc,dot='.mlf')

#import pyULT
#pyULT.HVite(messie_path+r'fucked_up.mlf',querie_path + r'5034_query_active/feature_list.scp',querie_path+'fa.mlf')




#######################phase3####################

# constuct std object
drpbox_path = r'/home/c2tao/Dropbox/'
querie_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/5034_query_active/'
labels_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/'+r'fa.mlf'
answer_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/'+r'querycontent.txt'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'

Q = std.STD(root = querie_path, label = labels_path, qlist = answer_path, corpus = corpus_path)
S_list = [3,7,13]
M_list = [50,100,300]
K_list = [36,40]
for S in S_list:
    for M in M_list: 
        for K in K_list: 
            p_name = str(K)+'_5034_'+str(M)+'_'+str(S)
            A = pyASR.ASR(target = drpbox_path+r'Semester 9.5/'+p_name+'/')
            Q.add_pattern(A,p_name)

Q.query_init()
#Q.query_copy()
Q.query_build()
Q.parse_query()






'''
Q.pattern_similarity("40_5034_300_3") 

batches = range(0,len(Q.pattern_list),len(Q.pattern_list)/6)+ [len(Q.pattern_list)]
print batches
i=4
print Q.pattern_list[batches[i]:batches[i+1]]
for p in Q.pattern_list[batches[i]:batches[i+1]]:
    Q.pattern_similarity(p) 
#similarity = {}
#for i in range(len(Q.query)):
#    similarity[i] = Q.feature_dtw(i,0)
#pickle.dump(similarity,open(querie_path+'similarity','wb'))
#similarity = pickle.load(open(querie_path+'similarity','rb'))

'''
'''
mean_average_precision = []
#q_answer = np.zeros((len(Q.query),len(Q.query_mlf.wav_list)))
q_answer = [0.0 for i in range(len(Q.query_mlf.wav_list))]
print len(Q.query_mlf.wav_list)
for i in Q.query_answer:
    for j in range(len(Q.query_answer[i])):
        #q_answer[i,j] = 1.0
        q_answer[i] = 1.0
    #mean_average_precision += util.average_precision(q_answer[i,:],np.array(similarity[i])),
    mean_average_precision += util.average_precision(q_answer,np.array(similarity[i])),
print np.mean(mean_average_precision)
print np.mean(sorted(mean_average_precision,reverse= True)[:16])
#similarity = pickle.load(open(querie_path+'similarity','rb'))
'''

sims = {}
#Q.feature_similarity()
for p in Q.pattern_list:
    sims[p] = Q.pattern_similarity(p)

simb = Q.feature_similarity()

import math
def to_relative(myList):
    adjusted = map(lambda x: math.erf(x),np.linspace(0, 1, num=5034))[::-1]
    scoindex = [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]
    return [adjusted[scoindex[i]] for i in range(len(myList))]

similarity0 = {}
similarity1 = {}
similarity2 = {}
for q in range(len(Q.query_answer)):
    similarity0[q] = -np.array(simb[q])
    similarity1[q] = np.zeros(5034)
    similarity2[q] = np.zeros(5034)
    for p in Q.pattern_list:
        if "36" in p:
            similarity1[q] += np.exp(-np.array(sims[p][q])/100.0)
            #similarity1[q] += to_relative(np.array(sims[p][q]))
        elif "40" in p:
            similarity2[q] += np.exp(-np.array(sims[p][q])/100.0)
            #similarity2[q] -= np.array(sims[p][q])
            #similarity2[q] += to_relative(np.array(sims[p][q]))

L0 = Q.mean_average_precision(similarity0)
L1 = Q.mean_average_precision(similarity1)
L2 = Q.mean_average_precision(similarity2)

Li = sorted(range(52),key = lambda x:L1[x]-L2[x])


for kkk in range(52):
    print np.mean([L2[i] for i in Li[:kkk]]) - np.mean([L1[i] for i in Li[:kkk]]),np.mean([L1[i] for i in Li[:kkk]]), np.mean([L2[i] for i in Li[:kkk]])




print np.mean([L0[i] for i in Li[:10]])
print np.mean([L1[i] for i in Li[:10]])
print np.mean([L2[i] for i in Li[:10]])




S_list = [3,7,13]
M_list = [50,100,300]
K_list = [36,40]
for K in K_list:     
    for M in M_list: 
        for S in S_list:    
            similarity ={}
            p = str(K)+'_5034_'+str(M)+'_'+str(S)
            for q in range(len(Q.query_answer)):
                similarity[q] = -1.0*np.array(sims[p][q])
            print np.mean([(Q.mean_average_precision(similarity))[i] for i in Li[:10]]) ,' ...'
            #print p, np.mean([(Q.mean_average_precision(similarity))[i] for i in Li[:10]])





'''

0.221916496907
0.233847939669
0.245052861082
36_5034_50_3 0.204076894809
36_5034_50_7 0.191556832007
36_5034_50_13 0.187760290978
36_5034_100_3 0.209363039572
36_5034_100_7 0.188080745216
36_5034_100_13 0.189663281696
36_5034_300_3 0.227309109344
36_5034_300_7 0.196890322538
36_5034_300_13 0.18908033873
40_5034_50_3 0.211931618365
40_5034_50_7 0.188168565132
40_5034_50_13 0.199919218462
40_5034_100_3 0.212888971024
40_5034_100_7 0.19836951053
40_5034_100_13 0.203318299388
40_5034_300_3 0.236256390727
40_5034_300_7 0.194514315332
40_5034_300_13 0.198749974955


'''
'''
[0.204076894809  ...
0.191556832007  ...
0.187760290978  ...
0.209363039572  ...
0.188080745216  ...
0.189663281696  ...
0.227309109344  ...
0.196890322538  ...
0.18908033873]
[0.211931618365  ...
0.188168565132  ...
0.199919218462  ...
0.212888971024  ...
0.19836951053  ...
0.203318299388  ...
0.236256390727  ...
0.194514315332  ...
0.198749974955]
'''
