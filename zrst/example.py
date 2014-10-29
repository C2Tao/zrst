from zrst import asr

# ## set paths ###
dropbox_path = r'/home/c2tao/Dropbox/'
corpus_path = dropbox_path + r'Semester 9.5/Corpus_TIMIT_test/'
target_path = dropbox_path + r'Semester 12.5/ICASSP 2015 Data/testrun/'
# init generated by matlab/clusterDetection.m
initial_path = dropbox_path + r'ULT_v0.2/matlab/IDump_timit_test_50_flatten.txt'
config_name = 'test_experiment'

# ## generate pattern ###
# declare object
A = asr.ASR(corpus=corpus_path, target=target_path, dump=initial_path)
# initialize
A.initialization(comment=config_name)
# run for several iterations
A.iteration('a', config_name)
# increase gaussian count by 1
# always use 'a_keep' instead of 'a' when having more than 1 gaussian mixture
A.iteration('a_mix', config_name)
A.iteration('a_keep', config_name)

# ## continue interrupted work ###
# declare object
# A = asr.ASR(corpus=corpus_path, target=target_path, dump=initial_path)
# select interrupted folder
# A.offset = 3
# A.readASR(str(A.offset) + '_' + config_name)
# continue for some iterations
# A.iteration('a', config_name)
