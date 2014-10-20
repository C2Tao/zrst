#!/usr/bin/env python
import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
#import matplotlib.axes as ax
from pprint import pprint
#from pylab import *
import os


class HTK:
    def __init__(self):
        #MLF
        self.wav2word = dict({})
        self.word2wav = dict({})
        self.wavList = []
        #DCT
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.wordList = []
        #SCP
        self.directory=[]
        
    def readSCP(self, path):
        self.directory = path
        return self
        
    def writeSCP(self, path, optList=[], mfcpath = ''):
        scp = open(path,'w')
        contents = os.listdir(self.directory)
        for file in contents:
            if 'hcopy' in optList:
                scp.write('"'+self.directory + file +'" ')
            scp.write('"'+mfcpath +file[:-3]+'mfc'+'"'+'\n')
        return self

    def convertMLF2DCT(self, threshold = 0):
        "word is deleted if count in histogram is smaller than threshold, threshold [0, +inf]"
        #"word is deleted if percentage in histogram is smaller than threshold, threshold (0,1)"
        #note: should run "testing" right after this if threshold is not 0
        
        word2pone = self.word2pone
        pone2word = self.pone2word
        wordList  = self.wordList
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.wordList  = []
        
        count = dict({})
        total = 0
        for word in self.word2wav:
            count[word] = len(self.word2wav[word])
            total += count[word]
        
        for word in self.word2wav:
            #if float(count[word])/total >= threshold:
            if count[word] >= threshold:
                self.wordList.append(word)
                self.word2pone[word] = word2pone[word]
                pone = word2pone[word]
                self.pone2word[pone] = word
        return self
    
    def writeDCT(self, path, optList=[]):
        dct = open(path,'w')
        
        if 'model' in optList:
            phones = set()
            for word in self.wordList:
                for phone in self.word2pone[word]:
                    phones.add(phone)
            phones = sorted(list(phones))
            for phone in phones:
                dct.write(phone + '\n')
            for extra in optList:
                if extra != 'model':
                    dct.write(extra + '\n')
            return self

        if 'grammar' in optList:
            dct.write('$word=')
            for word in self.wordList:
                dct.write(word)
                if self.wordList.index(word) != len(self.wordList)-1: dct.write(' | ')
            dct.write(' ;\n')
            if 'sp' in optList:
                dct.write('$word_sp=$word sp;\n')
                dct.write('(sil {$word_sp} $word sil)\n')
            else:
                dct.write('(sil <$word> sil)\n')
            return self
            
        if 'full' in optList:
            dct.write('$word=')
            for word in self.wordList:
                dct.write(word)
                if self.wordList.index(word) != len(self.wordList)-1: dct.write(' | ')
            dct.write(' ;\n')
            dct.write('(<$word>)\n')
            return self        
        
        sortedwordList = []
        for word in self.wordList:
            sortedwordList.append(word)
        sortedwordList.sort()
        for word in sortedwordList:
            phone = self.word2pone[word]
            pstring =""
            for p in range(len(phone)):
                pstring += phone[p]
                if p!=len(phone)-1:
                    pstring += ' '
            dct.write(word + ' ' + pstring + '\n')
        for extra in optList:
            dct.write(extra + ' ' + extra + '\n')
        return self
            
    def readDCT(self, path, optList=[]):
        text = [t.rstrip('\n').split() for t in open(path).readlines()]
        self.word2pone=({})
        self.pone2word=({})
        self.wordList=[]
        for t in text:
            if t[0] in ['sil', 'sp']:  continue
            pone = tuple(t[1:])            
            word = t[0]
            self.wordList.append(word)
            self.word2pone[word] = pone
            self.pone2word[pone] = word
        return self

    def readMLF_setmode(self, path):
        if path[-4:]== '.txt':
            return 'dump'
        elif path[-4:]== '.mlf':
            text = [t.rstrip('\n').split() for t in open(path).readlines()]
            if len(text[2])>1:
                return 'time'
            if len(text[2])==1:
                return 'word'

    def readMLF_filter(self,text,mode,filterList):
        popList=[]
        for i in range(len(text)):
            #print text[i]
            if not text[i]: 
                popList.append(i)
                continue
            if  text[i][0][0] == '.' or text[i][0][0]== '#':
                popList.append(i)
                continue
            
            for banned in filterList:
                if mode =='word' and '.' not in text[i][0]:
                    if text[i][0] == banned:
                        popList.append(i)
                        continue
                if mode =='time' and '.' not in text[i][0]:
                    if text[i][2] == banned:
                        popList.append(i)
                        continue
        popList.reverse()
        for i in popList:
            text.pop(i)
        return text
        
    def readMLF(self, path, optList=[]):
        "reads in MLF file, set words to filter in optList"
        mode = self.readMLF_setmode(path)
        raw_text = [t.rstrip('\n').split() for t in open(path).readlines()]        
        text = self.readMLF_filter(raw_text,mode,optList)
        
        for t in text:
            if '.' in t[0]:
                if mode =='time':
                    wav = t[0][3:-5]
                if mode =='word':
                    wav = t[0][3:-5]
                if mode =='dump':
                    wav = t[0][:-4]
                self.wavList.append(wav)                                                
            else:
                if mode == 'time':
                    word = t[2]
                    try:
                        contentword = [word, int(t[0]), int(t[1]), float(t[3])]
                        contentwav = [wav, int(t[0]), int(t[1]), float(t[3])]
                    except:
                        contentword = [word, int(t[0]), int(t[1])]
                        contentwav = [wav, int(t[0]), int(t[1])]
                if mode == 'word':
                    word = t[0]
                    contentword = [word,]
                    contentwav = [wav,]
                if mode == 'dump':
                    #pones = tuple(map(lambda x: 'phone' + x, t))
                    pones = tuple(map(lambda x: 'p' + x, t))
                    if pones not in self.pone2word:
                        #wordID = 'word' + str(len(self.word2pone) + 1)
                        wordID = ''
                        for phone in pones:
                            wordID += phone
                        self.pone2word[pones] = wordID
                        self.word2pone[wordID] = pones
                        self.wordList.append(wordID)
                    word = self.pone2word[pones]
                    contentword = [word,]
                    contentwav = [wav,]
                #if word not in optList:
                try:    self.wav2word[wav] += contentword,
                except: self.wav2word[wav] =  contentword,
                try:    self.word2wav[word] += contentwav,
                except: self.word2wav[word] =  contentwav, 
        self.wordList = sorted(self.wordList)
        return self
        
    def writeMLF(self, path, optList=[], dot3='.lab'):
        mlf = open(path,'w')
        if 'lm' in optList:
            for wav in self.wavList:
                mlf.write('<s> ')
                for i in range(len(self.wav2word[wav])):
                    if 'phone' in optList:
                        word = self.wav2word[wav][i][0]
                        phones = self.word2pone[word]
                        for j in range(len(phones)):
                            mlf.write(phones[j] + ' ')
                    else: 
                        mlf.write(self.wav2word[wav][i][0] + ' ')
                mlf.write('</s>\n')
            return self        
            
        mlf.write('#!MLF!#\n')
        for wav in self.wavList:
            mlf.write('"*/' + wav + dot3 + '"\n')
            if 'sil' in optList: mlf.write('sil\n')
            for i in range(len(self.wav2word[wav])):
                if 'phone' in optList:
                    word = self.wav2word[wav][i][0]
                    phones = self.word2pone[word]
                    for j in range(len(phones)):
                        mlf.write(phones[j] + '\n')
                        #if 'sp' in optList and j!= len(phones)-1: mlf.write('sp\n')
                else: 
                    mlf.write(self.wav2word[wav][i][0] + '\n')
                    
                if 'sp' in optList and i!= len(self.wav2word[wav])-1: mlf.write('sp\n')
            if 'sil' in optList: mlf.write('sil\n')                
            mlf.write('.\n')
        return self
        
    def cross(self, result):
        composition = dict({})
        for word in self.word2wav:
            for wav in self.word2wav[word]:
                abeg = wav[1]
                aend = wav[2]
                adur = aend - abeg
                for seg in result.wav2word[wav[0]]:
                    rdur = 0
                    rbeg = seg[1]
                    rend = seg[2]
                    if rbeg <= abeg and rend >= abeg:
                        rdur = rend - abeg
                    if rbeg >= abeg and rend <= aend:
                        rdur = rend - rbeg
                    if rbeg <= aend and rend >= aend:
                        rdur = aend - rbeg
                    if rbeg <= abeg and rend >= aend:
                        rdur = aend - abeg
                    try:    composition[(word,seg[0])] += rdur
                    except: composition[(word,seg[0])]  = 0
        return composition
    def seg_cross(self, result):
        #note: we are A, looking for overlap with R(eference)
        seg_composition = dict({})
        for word in self.word2wav:
            for wav in self.word2wav[word]:
                abeg = wav[1]
                aend = wav[2]
                adur = wav[2]-wav[1]
                if word.count('p') !=0:
                    phone_num  = word.count('p')
                else:
                    phone_num  = word.count('[')
                pattern = []
                for seg in result.wav2word[wav[0]]:
                    rbeg = seg[1]
                    rend = seg[2]
                    phone = seg[0]
                    thresh = 0.5/phone_num
                    if (rbeg < abeg and rend > abeg\
                        and float(rend - abeg)/adur > thresh)\
                    or (rbeg >= abeg and rend <= aend\
                        and float(rend - rbeg)/adur > thresh)\
                    or (rbeg < aend and rend > aend\
                        and float(aend - rbeg)/adur > thresh)\
                    or (rbeg <= abeg and rend >= aend\
                        and float(aend - abeg)/adur > thresh):                    
                        pattern.append(phone)
                pattern = tuple(pattern)
                if word in seg_composition:
                    if pattern in seg_composition[word]:
                        seg_composition [word][pattern]+=1
                    else:
                        seg_composition [word][pattern] =1
                else:
                    seg_composition [word] = dict({})
                    seg_composition [word][pattern] = 1
        return seg_composition

    def sortdict(self, dictionary):
        keyList =[key             for key in dictionary]
        valList =[dictionary[key] for key in dictionary]
        
        keySort = []
        valSort = []
        while valList:
            popIndex = valList.index(max(valList))
            keySort.append( keyList.pop(popIndex) )
            valSort.append( valList.pop(popIndex) )
        return keySort, valSort




    
    def percentage(self, mlf):
        composition = self.cross(mlf)
        percent = dict({})
        for rword in self.word2wav:
            total = 0
            percent[rword] = dict({})
            for aword in mlf.word2wav:
                try:
                    if composition[(rword,aword)]!=0:
                        total += composition[(rword,aword)]
                        percent[rword][aword] = composition[(rword,aword)]
                except:pass
            percent[rword] = self.normalize_dict(percent[rword])
        return percent
    
    def renameMLF2MLF(self, mlf):
        percent = self.percentage(mlf)
        name_change = dict({})
        for rword in percent:
            maxvalue= 0
            for aword in percent[rword]:
                try:
                    if percent[rword][aword] >= maxvalue:
                        name_change[rword] = aword
                        maxvalue = percent[rword][aword]
                except:pass
        #pprint(percent)
        #pprint(name_change)
        
        word2pone = self.word2pone
        pone2word = self.pone2word
        wordList  = self.wordList
        word2wav = self.word2wav
        
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.wordList  = []
        self.word2wav = dict({})

        for word in word2wav:
            try: 
                cword = name_change[word]
            except:
                #note: this case should only happen in the first pass
                #      this means that a word is in the decoded results with 0 duration
                name_change[word] = word
                cword = name_change[word]
            #print cword,word
            try:
                self.word2pone[cword] = word2pone[word]
            except:
                word2pone[word] = word,
                self.word2pone[cword] = word2pone[word]
            self.pone2word[word2pone[word]] = cword
            self.wordList.append(cword)
            self.word2wav[cword] = word2wav[word]
        #pprint(self.wav2word)
        for wav in self.wavList:
            for word in self.wav2word[wav]:
                word[0] = name_change[word[0]]
                
    def deleteMLF2DCT(self, mlf, threshold = 0.9):
        "words with posterior unigram perplexity larger than threshold will be deleted"
        #note: should run "testing" right after this
        composition = self.cross(mlf)
        condition = self.condition_dict(composition)
        deleteList = []
        for word in condition:
            if self.entropy_dict(condition[word]) > threshold:
                deleteList.append(word)
        print(deleteList)
        self.deleteMLF2DCT_delete(deleteList)
        return self
    def condition_dict(self, dictionary):
        #converts dict[(x,y)] to double dict[x][y]
        condition = dict({})
        for pair in dictionary:
            condition[pair[0]] = dict({})                
        for pair in dictionary:
            condition[pair[0]][pair[1]] = dictionary[pair]
        return condition
            
    def deleteMLF2DCT_delete(self, deleteList):
        word2pone = self.word2pone
        pone2word = self.pone2word
        wordList  = self.wordList
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.wordList  = []
        
        for word in self.word2wav:
            if word not in deleteList:
                self.wordList.append(word)
                self.word2pone[word] = word2pone[word]
                pone = word2pone[word]
                self.pone2word[pone] = word
                
    def word_count(self):
        return len(self.word2wav)
        
    def entropy(self):
        composition = self.cross(self)
        return self.entropy_dict(composition)
        
    def joint_entropy(self, mlf):
        composition = self.cross(mlf)
        return self.entropy_dict(composition)
        
    def entropy_dict(self, dictionary):
        prob = self.normalize_dict(dictionary)
        H = 0
        for x in prob:
            H -= np.log2(prob[x]** prob[x])
        return H
        
    def normalize_dict(self, dictionary):
        total  = 0
        normalized = dict({})
        for word in dictionary:
            total += dictionary[word]
        for word in dictionary:
            normalized[word] = float(dictionary[word])/total
        return normalized

    def conditional_entropy(self, mlf):
        composition = self.cross(mlf)
        p_xy = self.normalize_dict(composition)
        p_y_given_x =dict({})
        for wpair in p_xy:
            pdict = dict({})
            for tpair in p_xy:
                if wpair[0] == tpair[0]:
                    pdict[tpair] = p_xy[tpair]
            ndict = self.normalize_dict(pdict)
            for pair in ndict:
                p_y_given_x[pair] = ndict[pair]
        #pprint(p_xy)
        #pprint(p_y_given_x)
        H_Y_given_X = 0
        for pair in p_xy:
            H_Y_given_X -= np.log2(p_y_given_x[pair]** p_xy[pair])
        #print H_Y_given_X
        return H_Y_given_X

