from pyULT import * 
from pyASR import * 
from pyHTK import * 
import wave
import sys
import os
def lab2wav_X(label, wav_dir, corpus):
    '''
    input: transcription label
    output: wav folder
    option: wav corpus
    '''
    SYS().cldir(wav_dir)
    A = HTK().readMLF(label)
    waveData=dict({})
    params=[]
    #print 'ready'
    for wavefile in A.wav2word:
        W = wave.open(corpus + wavefile + '.wav')
        scale = float(W.getnframes())/A.wav2word[wavefile][-1][1]
        params = W.getparams()
        for word in A.wav2word[wavefile]:
            framechunk = W.readframes(int(scale*(word[2]-word[1])))
            if word[0] in waveData:
                try:
                    waveData[word[0]] += framechunk
                except:
                    print word[0], 'out of memory'
                    pass
            else:
                waveData[word[0]]  = framechunk
    for word in waveData:
        if "<" in word:
            continue
            #this should only happen for words with illegal characters
        S = wave.open(wav_dir + word + '.wav','w')
        S.setparams(params)
        S.writeframes(waveData[word])
def lab2wav(label, wav_dir, corpus):
    '''
    input: transcription label
    output: wav folder
    option: wav corpus
    '''
    SYS().cldir(wav_dir)
    A = HTK().readMLF(label)
    waveData=dict({})
    params=[]
    #print 'ready'
    for wavefile in A.wav2word:
        W = wave.open(corpus + wavefile + '.wav')
        #scale = double(W.getnframes())/A.wav2word[wavefile][-1][1]
        params = W.getparams()
        scale = float(params[2])/10000000
        end = 0
        for word in A.wav2word[wavefile]:
            framechunk = W.readframes(int(scale*(word[1]-end)))
            framechunk = W.readframes(int(scale*(word[2]-word[1])))
            end = word[2]
            if word[0] in waveData:
                try:
                    waveData[word[0]] += framechunk
                except:
                    print word[0], 'out of memory'
                    pass
            else:
                waveData[word[0]]  = framechunk
    for word in waveData:
        if "<" in word:
            continue
            #this should only happen for words with illegal characters
        S = wave.open(wav_dir + word + '.wav','w')
        S.setparams(params)
        S.writeframes(waveData[word])
def asr_accuracy(comment,i, acc, ans):
    #comment = 'flatphone_acoustic'
    #i=1
    dir = '../{}_{}/'.format(str(i), comment)
    print dir
    res = dir + 'result/flat_result.mlf'
    check = os.listdir(dir + 'result/') 
    #if 'flat_result.mlf' not in check:
    word2phone(dir+'result/result.mlf',dir+'result/flat_result.mlf')    
    
    phn = dir + 'library/dictionary.txt'
    dct = '../../Lee/if.lex.ascii_fix'
    #ans = './data/Corpus_ptv942_character.mlf'
    tra = 'temp.mlf'
    trb = 'temp2.mlf'
    #acc = 'ASR_accuracy.txt'
    R = HTK().readMLF(res, ['sp', 'sil', '<s>', '</s>']).readDCT(phn)
    A = HTK().readMLF(ans, ['sp', 'sil', '<s>', '</s>'])
    R.renameMLF2MLF(A)
    R.writeMLF(tra, dot3='.rec')
    A.writeMLF(trb, dot3='.lab')
    #asc2big(tra,'big'+tra)
    SYS().cygwin('HResults -e "???" sil -e "???" sp -e "???" "<s>" -e "???" "</s>" -I {} {} {} >> {}'.format(trb, dct, tra, acc))

def getF1(ult_mlf, big_mlf, N,comparison = 'temp_F1_self.txt'):
    big2asc(big_mlf,'temp_asc_querycontent.txt')
    text = [t.strip('\n') for t in open('temp_asc_querycontent.txt').readlines()]

    A = HTK().readMLF('data/Corpus_ptv942_hvite.mlf', optList = ['<s>','</s>','sil','sp'])
    B = HTK().readMLF(ult_mlf, optList = ['<s>','</s>','sil','sp'])
    c_string =dict({})
    c_time =dict({})
    for wav in A.wavList:
        c_string[wav] = ''
        c_time[wav] = []
        for word in A.wav2word[wav]:
            c_string[wav] += word[0]
            c = len(word[0])/6
            for i in range(c):
                c_time[wav].append((word[2]-word[1])/c*i +word[1])
        c_time[wav].append(A.wav2word[wav][-1][2])

    p_string =dict({})
    p_time =dict({})
    for wav in B.wavList:
        p_string[wav] = ''
        for word in B.wav2word[wav]:
            p_string[wav] += word[0]
    #print p_string
    ### collect phrase by matching timing information
    phrase_count = dict({})
    for t in text:
        phrase_count[t] = dict({})
        for wav in A.wavList:
            c_pos = c_string[wav].find(t)
            if c_pos ==-1: continue
            beg,end = c_time[wav][c_pos/6],c_time[wav][c_pos/6+len(t)/6]
            phrase = ''
            record = False
            for word in B.wav2word[wav]:
                if (word[2]-beg >= 0.7*(word[2]-word[1])): record = True
                if word[1] <= beg and word[2]>= end: record = True
                if (end-word[1]) <= 0.3*(word[2]-word[1]): record = False
                if word[1] >= end: record = False
                if record: phrase += word[0]
            if phrase in phrase_count[t]:
                phrase_count[t][phrase]+=1
            else:
                phrase_count[t][phrase] =1
        #print phrase_count[t]
    ### pick nbest from phrase
    #N = 4
    nbest = dict({})
    for t in text:
        items = sorted(phrase_count[t].items(), key=lambda (k,v): (v,k),reverse = True)
        n = min(N,len(phrase_count[t]))
        nbest[t] = [i[0] for i in items[:n]]
    #pprint(nbest)
    #compute F1
    result_retrieval = dict({})
    answer_retrieval = dict({})
    for t in text:
        result_retrieval[t] = []
        answer_retrieval[t] = []
        for wav in B.wavList:
            for phrase in nbest[t]:
                #print phrase
                if phrase in p_string[wav]:
                    result_retrieval[t].append(wav)
                    continue
        for wav in B.wavList:
            if t in c_string[wav]:
                answer_retrieval[t].append(wav)
    #print answer_retrieval
    #print result_retrieval
    F = open(comparison,'w')
    FA,CO,FR,recall,precision,F1 = dict({}),dict({}),dict({}),dict({}),dict({}),dict({})
    for t in text:
        FA[t] = 0 
        CO[t] = 0
        FR[t] = 0
        for wav in result_retrieval[t]:
            if wav not in answer_retrieval[t]:
                FA[t]+=1
            else:
                CO[t]+=1
        for wav in answer_retrieval[t]:
            if wav not in result_retrieval[t]:
                FR[t]+=1
        #print FA, CO, FR
        F.write(t + '\n')
        F.write('FA:' +str(FA[t]) + ' FR:'+str(FR[t]) + ' CO:'+str(CO[t]) +'\n')
        recall[t] = float(CO[t])/(FR[t]+CO[t])
        precision[t] = float(CO[t])/(FA[t]+CO[t])
        F1[t] = 2*recall[t]*precision[t]/(recall[t]+precision[t])
        F.write('recall:%.2f precision:%.2f F1:%.2f'%(recall[t],precision[t],F1[t]) +'\n')
    return FA,CO,FR,recall,precision,F1

def avgF1(comment,F1out,query,N=4):
    #get folder list
    raw = os.listdir('../')
    sortraw = dict({})
    for dir in raw:
        if comment in dir:
            if dir.split(comment)[-1] =='':
                try: sortraw[int(dir.split('_')[0])] = dir
                except:pass
        #try: sortraw[int(dir.split('_')[0])] = dir
        #except: pass
    #get information from each folder
    FA,CO,FR,recall,precision,F1 = dict({}),dict({}),dict({}),dict({}),dict({}),dict({})
    for i in range(1,len(sortraw)+1):
        FA[i],CO[i],FR[i],recall[i],precision[i],F1[i] = \
        getF1('../{}/result/result.mlf'.format(sortraw[i]), query,N)
        #asc2big('{}.txt'.format(sortraw[i]),'{}_big.txt'.format(sortraw[i]))
    #get weight
    counter = dict({})
    for i in range(1,len(sortraw)+1):
        counter[i] = [0,0,0,0]
        for t in CO[i]:
            loc_tot = CO[i][t] + FR[i][t]
            counter[i][0] += recall[i][t] * loc_tot
            counter[i][1] += precision[i][t] * loc_tot
            counter[i][2] += F1[i][t] * loc_tot
            counter[i][3] += loc_tot
    #output average F1
    A = open(F1out,'w')
    for j in range(3):
        for i in range(1,len(sortraw)+1):
            A.write('%.2f '%(counter[i][j]/counter[i][-1]))
        A.write('\n')
		