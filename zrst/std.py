from pyASR import ASR
from pyHTK import HTK
from pyASR import SYS
import util
import pyHMM
import os
import shutil

class STD(object):
    def __init__(self, root, label, corpus = ''):
        self.root = root
        self.label = label
        self.corpus = corpus
        
        self.pattern_list = []
        self.pattern_dict = {}

        self.feature_fold = root + r'/feature_file/'
        self.feature_file = root + r'/feature_list.scp'
        
        self.distanc_fold = root + r'/pattern_distance/'
        self.distanc_file = {}
        self.kl = []

        self.decoded_fold = root + r'/pattern_decoded/'
        self.decoded_file = {}


    def query_init(self):
        try: os.mkdir(self.feature_fold)
        except: True
        try: os.mkdir(self.distanc_fold)
        except: True
        try: os.mkdir(self.decoded_fold)
        except: True

        if os.path.exists(self.feature_file): return  
        files = util.make_feature(self.corpus,self.feature_fold)
        with open(self.feature_file,'w') as myfile:
            for f in files:
                myfile.write( f +'\n')
            myfile.close()

    def query_build(self):
        for p in self.pattern_list:
            self.pattern_distanc(p)
            self.pattern_decoded(p)

    def add_pattern(self, pattern, pattern_name):
        self.pattern_list += pattern_name,
        self.pattern_dict[pattern_name] = pattern
        self.decoded_file[pattern_name] = self.decoded_fold + '/' + pattern_name +'.mlf'
        self.distanc_file[pattern_name] = self.distanc_fold + '/' + pattern_name +'.kl'

    def pattern_decoded(self, pattern_name):
        if os.path.exists(self.decoded_file[pattern_name]): return  
        self.pattern_dict[pattern_name].external_testing(Q.feature_file,Q.decoded_file[pattern_name])

    def pattern_distanc(self, pattern_name):
        if os.path.exists(self.distanc_file[pattern_name]): return  
        dm = {}
        H = pyHMM.parse_hmm(self.pattern_dict[pattern_name].X['models_hmm'])
        p_list = []
        for p in H.keys():
            if 's' not in p:
                p_list.append(p)
        
        for i in p_list:
            for j in p_list:
                if (j,i) not in dm:
                    dm[(i,j)] = pyHMM.kld_hmm(H[i],H[j])
        pickle.dump(dm, open(self.distanc_file[pattern_name],'wb'))

    def read_distance(self, pattern_name):
        return pickle.load(open(self.decoded_file[pattern_name] ,'rb'))
    
    def query_copy(self):
        for p in pattern_list:
            shutil.copyfile(pattern_dict[p].X['result_mlf'],self.decoded_file[p])

'''
drpbox_path = r'/home/c2tao/Dropbox/'
querie_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/5034_query_active/'
labels_path = drpbox_path + r'Semester 8.5/_readonly/homophone_time_missing.mlf'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'
target_path1 = drpbox_path + r'Semester 9.5/40_5034_50_3/'
target_path2 = drpbox_path + r'Semester 9.5/36_5034_50_3/'

A = ASR(target = target_path1)
B = ASR(target = target_path2)
Q = STD(root = querie_path, label = labels_path, corpus = corpus_path)
Q.add_pattern(A,'40_5034_50_3')
Q.add_pattern(B,'36_5034_50_3')
Q.query_init()
Q.query_build()
'''
