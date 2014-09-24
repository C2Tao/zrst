import sys
import shelve
sys.path.append('/home/c2tao/Dropbox/ULT_v0.3/')
import pyASR
import pySFX
import pyEXT
import os

### set paths ###
drpbox_path = r'/home/c2tao/Dropbox/'
corpus_path = drpbox_path + r'Semester 9.5/Corpus_TIMIT_test/'
target_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/testrun/'
intial_path = drpbox_path + r'ULT_v0.3/matlab/IDump_timit_test_50_flatten.txt'
config_name = 'test_experiment'

### generate pattern ###
# declare object
A = pyASR.ASR(corpus = corpus_path, target = target_path, nState = 3, dump = intial_path)
# initialize
A.initialization(comment = config_name)
# run for several iterations
A.iteration('a', config_name)
# increase gaussian count
A.iteration('a_mix', config_name)
A.iteration('a_keep', config_name)

### continue interrupted work ###
# declare object
A = pyASR.ASR(corpus = corpus_path, target = target_path, nState = 3, dump = intial_path)
# select interrupted folder
A.offset = 3
A.readASR(str(A.offset)+'_'+config_name)
# continue for some iterations
A.iteration('a',config_name)




'''
#B = pyASR.ASR(corpus = '../Corpus_TIMIT_dev/',   target = './timit_dev/',dump = 'IDump_timit_dev_50.txt')
#B.initialization('dev_dummy')

S_list = [3,5,7,9,11]
M_list = [50,100,200,300]
#M = 50
#S = 3
for M in M_list:
    for S in S_list:
        A = pyASR.ASR(corpus = '../Corpus_TIMIT_train/',   target = './timit_train_'+str(M)+'_'+str(S)+'/',      nState = S,dump = 'IDump_timit_train_'+str(M)+'_flatten.txt')
        A.readASR('../40_timit_train_'+str(M)+'_'+str(S)+'_relabel/')
        A.offset = 40
        A.external_testing('../dev_dummy/library/list.scp','query_dev/40_timit_train_'+str(M)+'_'+str(S)+'_relabel.mlf')
        A.exp_testing('../dev_dummy/library/list.scp','query_dev_nbest/40_timit_train_'+str(M)+'_'+str(S)+'_relabel.mlf',5)

        #A.lat_testing()

#MS_list = [(300,7),(100,3),(100,7),(100,13)]
#for MS in MS_list:
#    M = MS[0]
#    S = MS[1]
#    A = pyASR.ASR(corpus = '../../Semester 8.5/Corpus_5034wav/', target = './5034_'+str(M)+'_'+str(S)+'/',       nState = S,dump = 'IDump_5034wav_'+str(M)+'_flatten.txt')
#    A.initialization(comment ='0_'+ A.target.split('/')[1])
#    for i in range(30):
#        A.iteration('a',A.target.split('/')[1],do_slice = False)







#S_list = [3,7,13]
#M_list = [50,100,300]        
#for M in M_list:
#    for S in S_list:
#        A = pyASR.ASR(corpus = '../../Semester 8.5/Corpus_5034wav/', target = './5034_'+str(M)+'_'+str(S)+'/',       nState = S,dump = 'IDump_5034wav_'+str(M)+'_flatten.txt')
#        A.readASR('../30_5034_'+str(M)+'_'+str(S)+'/')
#        A.offset = 30
#        A.iteration('a_mix',A.target.split('/')[1],do_slice = False)
#        A.iteration('a_keep',A.target.split('/')[1],do_slice = False)
#        A.iteration('a_mix',A.target.split('/')[1],do_slice = False)
#        A.iteration('a_keep',A.target.split('/')[1],do_slice = False)
#        A.iteration('a_mix',A.target.split('/')[1],do_slice = False)
#        A.iteration('a_keep',A.target.split('/')[1],do_slice = False)


#for S in S_list:
#    A = pyASR.ASR(corpus = '../Corpus_TIMIT_test/',             target = './timit_test_50_'+str(S)+'/',      nState = S,dump = 'IDump_timit_test_50_flatten.txt')
#    A.initialization(comment ='0_'+ A.target.split('/')[1])
#    for i in range(30):
#        A.iteration('a',A.target.split('/')[1],do_slice = False)

#for S in S_list:
#    A = pyASR.ASR(corpus = '../Corpus_TIMIT_test/',             target = './timit_test_300_'+str(S)+'/',      nState = S,dump = 'IDump_timit_test_300_flatten.txt')
#    A.initialization(comment ='0_'+ A.target.split('/')[1])
#    for i in range(30):
#        A.iteration('a',A.target.split('/')[1],do_slice = False)

#for S in S_list:
#    A = pyASR.ASR(corpus = '../Corpus_TIMIT_train/',             target = './timit_train_50_'+str(S)+'/',      nState = S,dump = 'IDump_timit_train_50_flatten.txt')
#    A.initialization(comment ='0_'+ A.target.split('/')[1])
#    for i in range(30):
#        A.iteration('a',A.target.split('/')[1],do_slice = False)

#for S in S_list:
#    A = pyASR.ASR(corpus = '../Corpus_TIMIT_train/',             target = './timit_train_300_'+str(S)+'/',      nState = S,dump = 'IDump_timit_train_300_flatten.txt')
#    A.initialization(comment ='0_'+ A.target.split('/')[1])
#    for i in range(30):
#        A.iteration('a',A.target.split('/')[1],do_slice = False)

'''