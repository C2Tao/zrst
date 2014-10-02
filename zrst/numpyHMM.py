import pyHMM
import pyASR
import pySFX
import pyEXT
import pprint
import numpy as np
import cPickle as pickle
from pprint import pprint

### set paths ###
drpbox_path = r'/home/c2tao/Dropbox/'
corpus_path = drpbox_path + r'Semester 8.5/Corpus_5034wav/'
target_path = drpbox_path + r'Semester 9.5/'
contue_path = drpbox_path + r'Semester 12.5/ICASSP 2015 Data/kl_distance/'

def get_dm(I,J,K):
    dm = {}
    H = pyHMM.parse_hmm(target_path+str(K)+'_5034_'+str(J)+'_'+str(I)+ r'/hmm/models')
    p_list = []
    for p in H.keys():
        if 's' not in p:
            p_list.append(p)
    
    for i in p_list:
        for j in p_list:
            if (j,i) not in dm:
                dm[(i,j)] = pyHMM.kld_hmm(H[i],H[j])
    pickle.dump(dm,open(contue_path +str(K)+'_5034_'+str(J)+'_'+str(I)+'.kl' ,'wb'))
    return dm

for I in [3,7,13]:
    for J in [50,100,300]:
        for K in [36, 40]:
            print I,J,K
            get_dm(I,J,K)
            


'''
dm = get_dm(3,50,40)
pprint(dm)
pickle.dump(dm,open(contue_path + 'distance_matrix','wb'))

dm = pickle.load(open(contue_path + 'distance_matrix','rb'))
pprint(dm)
'''