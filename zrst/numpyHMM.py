import pyHMM
import pyASR
import pySFX
import pyEXT
import pprint
import numpy as np
### set paths ###
drpbox_path = r'/home/c2tao/Dropbox/'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'
target_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/5034_relabeled_active/'
contue_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/5034_relabeled/'

H = pyHMM.parse_hmm(target_path + r'/hmm/models')
p_list = []
for p in H.keys():
    if 's' not in p:
        p_list.append(p)
for i in p_list:
    for j in p_list:
        print i,j, pyHMM.kld_hmm(H[i],H[j])

