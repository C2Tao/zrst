import numpy as np
import struct
class DTW(object):
    def __init__(self, seq1, seq2, distance_func=None):
        '''
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        '''
        self._seq1 = seq1
        self._seq2 = seq2
        self._distance_func = distance_func if distance_func else lambda: 0
        self._map = {(-1, -1): 0.0}
        self._distance_matrix = {}
        self._path = []
 
    def get_distance(self, i1, i2):
        ret = self._distance_matrix.get((i1, i2))
        if not ret:
            ret = self._distance_func(self._seq1[i1], self._seq2[i2])
            self._distance_matrix[(i1, i2)] = ret
        return ret
 
    def calculate_backward(self, i1, i2):
        '''
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        '''
        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]
 
        if i1 == -1 or i2 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')
 
        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))
 
        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
            self.calculate_backward(min_i1, min_i2)
 
        return self._map[(i1, i2)]
 
    def get_path(self):
        '''
        Calculate the path mapping.
        Must be called after calculate()
        '''
        i1, i2 = (len(self._seq1) - 1, len(self._seq2) - 1)
        while (i1, i2) != (-1, -1):
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        return self._path
 
    def calculate(self): 
        return self.calculate_backward(len(self._seq1) - 1,
                                       len(self._seq2) - 1)

class SubDTW(DTW):
    def __init__(self, seq1, seq2, distance_func=None):
        '''
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        '''
        DTW.__init__(self, seq1, seq2, distance_func)
        
        self._map = {}
        for ___ in range(-1,len(seq1)):
            self._map[(___, -1)] = 0.0
        self.end_pos = -1
        self.beg_pos = -1
    def calculate_backward(self, i1, i2):
        '''
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        '''
        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]
 
        if i1 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')
 
        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))
 
        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
            self.calculate_backward(min_i1, min_i2)
 
        return self._map[(i1, i2)]
 
    def get_path(self):
        '''
        Calculate the path mapping.
        Must be called after calculate()
        '''
        i1, i2 = (self.end_pos, len(self._seq2) - 1)
        while i2 != -1:
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        self.beg_pos = self._path[-1][0]
        return self._path

    def calculate(self): 
        self.calculate_backward(len(self._seq1) - 1,
                                       len(self._seq2) - 1)
        self.end_pos = min(range(-1,len(self._seq1)), key=lambda x: self._map[(x,len(self._seq2)-1)]) 
        return self._map[self.end_pos,len(self._seq2) - 1]


'''debug DTW
a = [0,1,2,3,4,5,6,7]
b = [2,3,4,5]
A = DTW(a,b,lambda x,y:abs(x-y))
print A.calculate()
print A.get_path()

B = SubDTW(a,b,lambda x,y:abs(x-y))
print B.calculate()
print B.get_path()
'''


def read_feature(file):
    fin = open(file,'rb')
    nN =struct.unpack('<i', fin.read(4))[0]
    period = struct.unpack('<i', fin.read(4))[0]
    nF = struct.unpack('<h', fin.read(2))[0]/4
    ftype = struct.unpack('<h', fin.read(2))[0]
    fmat = np.zeros([nN,nF])
    for i in range(nN):
        for j in range(nF):
            fmat[i,j] = struct.unpack('<f', fin.read(4))[0]
    return fmat
'''
from pylab import *
file = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/N200108011200-01-01.mfc'
fmat = read_feature(file)
matshow(fmat)
show()
'''

class MLF(object):
    def __init__(self,path,mlf_name = ''):
        self.path     = path
        self.mlf_name = mlf_name
        
        lines = open(self.path).readlines()
        wav_list = []
        int_list,tag_list,med_list = [],[],[]
        tok_list = []
        log_list = []
        mlf_type = True
        for line in lines[1:]:
            line = line.strip()
            if '"' in line:
                mode = 'wav'
                wav_list += line[3:-5],
                int_temp = []
                tag_temp = []
                med_temp = []
                log_temp = []
                #print wav_list[-1]
            elif line =='.':
                int_list += int_temp,
                tag_list += tag_temp,
                med_list += med_temp,
                log_list += log_temp,
                tok_list = list(set(tok_list + tag_temp))
                continue
            else:
                try:
                    ibeg = int(line.split()[0])/100000
                    iend = int(line.split()[1])/100000
                    if int_temp and iend == int_temp[-1]: continue
                    int_temp += iend,
                    tag_temp += line.split()[2],
                    med_temp += (ibeg+iend)/2,
                    log_temp += float(line.split()[3]),
                except:
                    ibeg = 0
                    iend = 0
                    int_temp += iend,
                    tag_temp += line.split()[0],
                    med_temp += (ibeg+iend)/2,
                    log_temp += 0,
                    mlf_type = False
        self.wav_list = wav_list
        self.int_list = int_list
        self.tag_list = tag_list
        self.med_list = med_list
        self.log_list = log_list
        self.tok_list = sorted(tok_list)
        self.mlf_type = mlf_type
    def fetch(self,med_list):
        return_list = []
        for I,T,Q in zip(self.int_list,self.tag_list,med_list):
            R = []
            pos = 0
            pi = 0         
            for i,t,j in zip(I,T,range(len(T))):
                if pos >= len(Q): break
                match = (pi <= Q[pos] and i > Q[pos])
                while match:
                    pos += 1
                    R += j,
                    #R += t,
                    if pos >= len(Q): break
                    match = (pi <= Q[pos] and i > Q[pos])
                pi=i
            return_list += R,
            assert len(R) == len(Q)
        assert len(return_list) == len(med_list)
        return return_list
    def accuracy(self,answer,acc ='acc.txt'):
        A = open('phone.txt','w')
        for p in answer.tok_list:
            A.write(p+'\n')
        os.system('HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}" "{}" "{}" >> "{}"'.format(\
            answer.path, 'phone.txt', self.path, acc))
        lines = open(acc).readlines()
        return  float(lines[-2].split()[1][6:-1])


    def expand(self,phone_mlf, dictionary, edit_file = 'w2p.led', command = 'EX\n' ):
        file = open(edit_file,'w')
        file.write(command)
        file.close()
        os.system("""HLEd -l '*' -d "{}" -i "{}" "{}" "{}" """.format(\
            dictionary, phone_mlf, edit_file, self.path))
            
    def write(self,path ='temp.mlf',dot='.rec',selection = []):
        if not selection:
            index_list = range(len(self.tag_list))
            wav_list   = self.wav_list
        else:
            index_list,wav_list = zip(*selection)
            #tuples of the form (original_wav_index,new_wav_name)
        if dot == '.lab': self.mlf_type = False
        M = open(path,'w')
        M.write('#!MLF!#\n')
        for i,w in zip(index_list,wav_list):
            M.write('"*/'+w+dot+'"\n')
            pj = 0
            for j in range(len(self.tag_list[i])):
                if self.mlf_type:
                    w1 = str(pj*100000)
                    w2 = str(self.int_list[i][j]*100000)
                    #w1 = str(pj)
                    #w2 = str(self.int_list[i][j])
                    w3 = self.tag_list[i][j]
                    w4 = str(self.log_list[i][j])
                    M.write('{} {} {} {}\n'.format(w1,w2,w3,w4))
                    pj = self.int_list[i][j]
                else:
                    w3 = self.tag_list[i][j]
                    M.write('{}\n'.format(w3))
            M.write('.\n')
        M.close()
        #self.path = path
        return MLF(path)
    def merge(self,ext):
        self.wav_list += ext.wav_list
        self.int_list += ext.int_list
        self.tag_list += ext.tag_list
        self.med_list += ext.med_list
        self.log_list += ext.log_list
        
        self.tok_list += ext.tok_list
        self.tok_list.sort()
'''
A = MLF(r'/home/c2tao/Dropbox/Semester 9.5/40_timit_train_300_11_relabel/result/result.mlf')
print A.wav_list[:10]
print A.int_list[:10]
'''
class Purity(object):
    def __init__(self,pat_MLF,ref_MLF):
        assert(len(pat_MLF.wav_list)==len(ref_MLF.wav_list))
        self.pat_MLF = pat_MLF
        self.ref_MLF = ref_MLF
        #nW number of wav files
        self.nW = len(self.pat_MLF.int_list)
        #feature length of per wav file
        self.nF = [self.pat_MLF.int_list[i][-1] for i in range(self.nW) ]
        self.purities = np.zeros(self.nW)
        self.purities_non = np.zeros(self.nW)
        self.purity = 0
        self.purity_non = 0
    def compute(self):
        for j in range(self.nW):
            str1 = []
            ilist = [0]+self.pat_MLF.int_list[j]
            for i in range(len(self.pat_MLF.int_list[j])):
                str1 += [self.pat_MLF.tag_list[j][i]]*(ilist[i+1]-ilist[i])
            str1 = np.array(str1)
            
            str2 = []
            ilist = [0]+self.ref_MLF.int_list[j]
            for i in range(len(self.ref_MLF.int_list[j])):
                str2 += [self.ref_MLF.tag_list[j][i]]*(ilist[i+1]-ilist[i])
            str2 = np.array(str2)

            mat1 = (str1[None,:] == str1[:,None])
            mat2 = (str2[None,:] == str2[:,None])
            mat3 = (mat1 == mat2)
            mat4 = (~mat1 * ~mat2)
            self.purities_non[j] = float(np.sum(mat3)-np.sum(mat4))/(self.nF[j]*self.nF[j]-np.sum(mat4))
            self.purities[j] = float(np.sum(mat3))/(self.nF[j]*self.nF[j])
        self.purity = np.mean(self.purities)
        self.purity_non = np.mean(self.purities_non)
        
        return self.purity
'''
A = MLF(r'/home/c2tao/Dropbox/Semester 9.5/40_timit_train_50_3_relabel/result/result.mlf')
B = MLF(r'/home/c2tao/Dropbox/Semester 9.5/1_timit_train_50_3/result/result.mlf')
C = MLF(r'/home/c2tao/Dropbox/Semester 9.5/timit_baseline/training_transcription7.mlf')

P = Purity(A,C)
P.compute()
print P.purity

Q = Purity(B,C)
Q.compute()
print Q.purity
'''

class Query(MLF):
    def __init__(self,path,query_name,query_list = {}):
        __init__(path,query_name)
        self.MLF()

        
 
