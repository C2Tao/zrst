import sys
import shelve
import pyASR
import pySFX
import pyEXT

#from zrst import pyASR
#from zrst import pySFX
#from zrst import pyEXT
import os

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
        '''
        ### generate pattern ###
        # declare object
        A = pyASR.ASR(corpus = corpus_path, target = target_path, nState = S, dump = intial_path)
        # initialize
        A.initialization(comment = config_name)
        # run for several iterations
        A.iteration('a', config_name)
        # increase gaussian count by 1
        # always use 'a_keep' instead of 'a' when having more than 1 gaussian mixture
        A.iteration('a_mix', config_name)
        A.iteration('a_keep', config_name)
        '''
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