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

class subDTW(DTW):
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

B = subDTW(a,b,lambda x,y:abs(x-y))
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
