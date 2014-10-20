from pyASR import *
import math


def flatten_pattern(input_dictionary, input_mlf, output_dictionary, output_mlf):
    temp_line = 'temp_line.txt'
    split_token = 'p'
    
    H = HTK()
    H.readMLF(input_mlf,['sil','sp','<s>','</s>'])
    H.readDCT(input_dictionary,['sil','sp','<s>','</s>'])
    
    
    M = open(output_mlf,'w')
    
    M.write('#!MLF!#\n')
    #print H.wavList
    for wav in H.wavList:
        M.write('"*/{}.rec"\n'.format(wav))
        for word in H.wav2word[wav]:
            #print word
            beg = int(word[1])
            end = int(word[2])
            pro = int(word[3])
            p_list = word[0].split(split_token)[1:]
            p_count = len(p_list)
            
            delta = 1.0*(end - beg)/p_count
            p_beg = [int(beg + delta* i)     for i in range(p_count)]
            p_end = [int(beg + delta* (i+1)) for i in range(p_count)]
            p_pro = [pro/p_count for i in range(p_count)]
            for i in range(p_count):
                if p_list[i] =='': continue
                M.write(str(p_beg[i])+' ')
                M.write(str(p_end[i])+' ')
                M.write(split_token + p_list[i]+' ')
                M.write(str(p_pro[i]))
                M.write('\n')
        M.write('.\n')
    
    F = HTK()
    F.wordList = []
    F.word2pone = dict({})
    F.pone2word = dict({})
    for pone in H.pone2word:
        #print pone
        for p in pone:
            if p in F.wordList: continue
            F.wordList.append(p)
            F.pone2word[p,] = p
            F.word2pone[p] = p,
    F.writeDCT(output_dictionary,['sil','sp'])
def parse_pattern_load_count(input_mlf,input_dictionary):

    temp_count = 'temp_count.txt'
    temp_line = 'temp_line.txt'
    N = 100
    A = HTK().readMLF(input_mlf,['sil','sp','<s>','</s>']).writeMLF(temp_line,['lm'])
    SYS().cygwin('ngram-count -text {} -write {} -order {} -vocab {}'.format(
        temp_line, temp_count, str(N), input_dictionary
    ))
    S = open(temp_count).readlines()
    
    #total_count = 0
    #for raw_line in S:
    #    line= raw_line.rstrip('\n').split()
    #    total_count += int(line [-1])
    #f_threshold = float(total_count)/len(S)

    f_threshold = 1
    count_dict = dict({})
    for raw_line in S:
        line= raw_line.rstrip('\n').split()
        if int(line [-1]) < f_threshold: continue
        if '<s>' in line or '</s>' in line: continue
        phrase = []
        for i in range(len(line)-1):
            phrase.append(line[i])
        phrase = tuple(phrase)
        count_dict[phrase] = int(line[-1]) 
    return count_dict
    
    
    
def parse_pattern(input_dictionary, input_mlf, output_dictionary):
    def avg_dict(count_dict):
        tot_con, avg_con = 0, 0
        for phrase in count_dict:
            tot_con += count_dict[phrase]
            avg_con += 1
        print tot_con,avg_con, tot_con/avg_con
        return 1.0*tot_con/avg_con
    def add_phrase(phrase):
        if phrase in p2w: return
        word = ''
        for phone in phrase: 
            word += str(phone)
        w2p[word] = phrase
        p2w[phrase] = word
        wdl.append(word)
    def entropy(v):
        s = sum(v)
        v = [float(p)/s for p in v]
        H=0
        for p in v:
            H -= p * math.log(p ** p,2)
        return H

    count_dict = parse_pattern_load_count(input_mlf,input_dictionary)
    cond_dict_tail=({})
    cond_dict_head=({})
    for phrase in count_dict:
        try:    cond_dict_tail[phrase[:-1]].append(count_dict[phrase])
        except: cond_dict_tail[phrase[:-1]] = [count_dict[phrase]]
        try:    cond_dict_head[phrase[1:]].append(count_dict[phrase])
        except: cond_dict_head[phrase[1:]] = [count_dict[phrase]]

    entropy_head = ({})
    entropy_tail = ({})
    for phrase in count_dict:    
        if phrase not in cond_dict_tail or phrase not in cond_dict_head: continue
        entropy_tail[phrase] = entropy (cond_dict_tail[phrase])
        entropy_head[phrase] = entropy (cond_dict_head[phrase])

    thr_count = avg_dict(count_dict) * 5
    thr_head = avg_dict(entropy_head)
    thr_tail = avg_dict(entropy_tail)
    w2p = dict({})
    p2w = dict({})
    wdl = []
    for phrase in count_dict:
        if phrase in entropy_tail and phrase in entropy_tail:
            if entropy_head[phrase]>thr_head and entropy_tail[phrase]>thr_tail and count_dict[phrase]>thr_count:
                add_phrase(tuple(phrase))
    A = HTK()
    A.word2pone = w2p
    A.pone2word = p2w
    A.wordList = wdl
    A.writeDCT(output_dictionary,['sil', 'sp'])    

def reparse_pattern(input_dictionary, input_mlf, output_mlf):
    A = HTK().readMLF(input_mlf,['sil','sp','<s>','</s>'])
    B = HTK().readDCT(input_dictionary)
    C = open(output_mlf,'w')
    
    C.write('#!MLF!#\n')
    for wav in A.wavList:
        C.write('"*/{}.rec"\n'.format(wav))
        pone = ''
        pro  = 0
        for word in A.wav2word[wav]:
            if A.wav2word[wav].index(word) == 0:
                beg = A.wav2word[wav][0][1]
                end = A.wav2word[wav][0][2]
                pone = word[0]
                pro  = int(word[3])
                continue
            if pone + word[0] not in B.wordList:
                #print B.wordList
                #print tuple(pone + [word[0]])
                C.write(str(beg)+' '+str(end)+' '+pone+' '+str(pro)+'\n')
                beg = word[1]
                end = word[2]
                pone = ''
                pro  = 0
            if A.wav2word[wav].index(word) == len(A.wav2word[wav])-1:
                end = word[2]
                pone = word[0]
                pro  = int(word[3])
                C.write(str(beg)+' '+str(end)+' '+pone+' '+str(pro)+'\n')
            pone += word[0]
            pro  += int(word[3])
            end = word[2]
        C.write('.\n'.format(wav)) 
            
    