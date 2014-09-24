from pyHTK import *
from pySCP import *

import os
import shutil
import time
import wave
import struct
#from ult_root import *
#import evaluation

class SYS:
    def __init__(self):
        pass
    def mkdir(self, path):
        try:    os.mkdir(path)
        except: pass
        return self
    def rmdir(self, path):
        try:    shutil.rmtree(path)
        except: pass
        return self
    def cldir(self, path):
        self.rmdir(path)
        self.mkdir(path)
        return self
    def mktxt(self, path, text):
        open(path, 'w').write(text)
    def rdtxt(self, path):
        return [t.rstrip('\n') for t in open(path).readlines()]
    def cygwin(self, command):
        "cygwin command: the/usual"
        print command
        os.system(command)
        return self
    def matlab(self, command):
        "matlab command: dotmfile(arg1,arg2)"
        print command
        os.system('matlab -nodesktop -minimize -r ' + command)
        return self
    def python(self, command):
        "python command: dotpyfile"
        print command
        os.system('python ' + command + '.py')
        return self
        
class ASR:
    def __init__(self, corpus = [], target = './', label = [], dump = 'IDump.txt', nState = 3, nFeature = 39,user_feature=False):
        self.corpus = corpus
        self.label = label
        self.target = target
        SYS().mkdir(target)
        self.X = dict({})
        self.setname()
        self.time = 0
        self.offset = 0
        self.dump = dump
        self.nFeature = nFeature
        self.nState = nState
        self.user_feature = user_feature
        os.chdir(self.target)
        os.chdir('..')
    def slice(self, corpus = [], target = [], label = []):
        if not corpus: corpus = self.X['corpus_dir']
        if not label:  label  = self.X['result_mlf']
        if not target: target = self.X['acoust_dir']
        SYS().cldir(target)
        A = HTK().readMLF(label)
        waveData=dict({})
        params=[]
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
            S = wave.open(target + word + '.wav','w')
            S.setparams(params)
            S.writeframes(waveData[word])

    def readASR(self, target):
        #shutil.rmtree(self.target)
        try:   shutil.rmtree(self.target)
        except:pass
        shutil.copytree(target, self.target)
    def writeASR(self, target):
        try:   shutil.rmtree(target)
        except:pass
        shutil.copytree(self.target, target)
    def record_comment(self, tag):
        open(self.X['evalua_txt'],'a').write(tag +'\n')

    def record_time(self):
        open(self.X['evalua_txt'],'a').write('total_time '+str(time.clock() - self.time)+'\n')

    def consistency_check(self):
        pass
    def clean(self):
        SYS()\
        .cldir(self.target)\
        .cldir(self.X['markov_dir'])\
        .cldir(self.X['answer_dir'])\
        .cldir(self.X['libray_dir'])\
        .cldir(self.X['result_dir'])\
        .cldir(self.X['lanmdl_dir'])\
        .cldir(self.X['acoust_dir'])
        
        #initiation
        open(self.X['hcopie_cfg'], 'w').write(hcopy)
        if self.user_feature ==True:
            open(self.X['config_cfg'], 'w').write(config_user)
        else:
            open(self.X['config_cfg'], 'w').write(config)
        #open(self.X['protos_txt'], 'w').write(proto) 
        self.build_proto()
        
        #training
        open(self.X['addsil_scp'], 'w').write(addsil)
        open(self.X['addmix_scp'], 'w').write(addmix)
        addmix_all = """MU 3 \{sil.state[2-4].mix}\nMU +1 {*.state[2-%d].mix}\n"""%self.nState
        open(self.X['addall_scp'], 'w').write(addmix_all)
        
        #reference
        if self.label:
            HTK().readMLF(self.X['refere_mlf'], ['sp', 'sil']).writeMLF(self.X['reflab_mlf'], dot3='.lab')
    def build_proto(self):
        a_state = self.nState
        a_feature = self.nFeature
        state = str(a_state+2)
        A = open(self.X['protos_txt'],'w')
        if self.user_feature==False:
            A.write('~o <VECSIZE> {} <MFCC_Z_E_D_A>\n'.format(str(a_feature)))
        else:
            A.write('~o <VECSIZE> {} <USER>\n'.format(str(a_feature)))            
        A.write('~h "proto"\n')
        A.write('<BeginHMM>\n')
        A.write('<NumStates> '+state+'\n')
        for i in range(2,a_state+2):
            A.write('<State> '+str(i)+'\n')
            
            A.write('<Mean> '+str(a_feature)+'\n')
            for i in range(a_feature): A.write('0.0 ')
            A.write('\n')
            
            A.write('<Variance> '+str(a_feature)+'\n')
            for i in range(a_feature): A.write('1.0 ')
            A.write('\n')

        A.write('<TransP> '+state+'\n')
        for i in range(a_state+2):
            for j in range(a_state+2):
                if i==0 and j==1:
                    A.write('1.0')
                elif i==0 and j==0:
                    A.write('0.0')
                elif i==a_state+1:
                    A.write('0.0')
                elif i+1==j or i==j:
                    A.write('0.5')
                else:
                    A.write('0.0')
                A.write(' ')
            A.write('\n')
        A.write('<EndHMM>\n')
        A.close()
        
    def setname(self):
        self.X['corpus_dir'] = self.corpus
        self.X['featur_dir'] = self.X['corpus_dir'][:-1] + '_MFCC/'
        
        self.X['markov_dir'] = self.target + 'hmm/'
        self.X['answer_dir'] = self.target + 'answer/'
        self.X['libray_dir'] = self.target + 'library/'
        self.X['result_dir'] = self.target + 'result/'
        self.X['lanmdl_dir'] = self.target + 'lm/'
        self.X['acoust_dir'] = self.target + 'ac/'

        #initiation
        self.X['hcopie_cfg'] = self.X['libray_dir'] + 'hcopy.cfg'
        self.X['config_cfg'] = self.X['libray_dir'] + 'config.cfg'
        self.X['protos_txt'] = self.X['libray_dir'] + 'proto'  
        
        #feature
        self.X['wavlst_scp'] = self.X['libray_dir'] + 'list.scp'
        self.X['wavhcp_scp'] = self.X['libray_dir'] + 'hcopy.scp' 

        #training
        self.X['hmmdef_hmm'] = self.X['markov_dir'] + 'hmmdef'
        self.X['macros_hmm'] = self.X['markov_dir'] + 'macros'
        self.X['models_hmm'] = self.X['markov_dir'] + 'models'
        self.X['addsil_scp'] = self.X['libray_dir'] + 'sil1.hed'
        self.X['addmix_scp'] = self.X['libray_dir'] + 'mix2_10.hed'
        self.X['addall_scp'] = self.X['libray_dir'] + 'mixall.hed'
        
        #testing
        self.X['result_mlf'] = self.X['result_dir'] + 'result.mlf'
        self.X['accrcy_txt'] = self.X['result_dir'] + 'accuracy.txt'
        self.X['evalua_txt'] = self.X['result_dir'] + 'evaluation.txt'
        
        #forward
        self.X['answer_mlf'] = self.X['answer_dir'] + 'answer_word.mlf'
        self.X['phonei_mlf'] = self.X['answer_dir'] + 'answer_phone.mlf'
        self.X['phonep_mlf'] = self.X['answer_dir'] + 'answer_phone_sp.mlf'
        self.X['dictry_dct'] = self.X['libray_dir'] + 'dictionary.txt'
        self.X['modeli_dct'] = self.X['libray_dir'] + 'phones.txt'
        self.X['modelp_dct'] = self.X['libray_dir'] + 'phones_sp.txt'
        self.X['grmmri_dct'] = self.X['libray_dir'] + 'grammar.txt'
        self.X['grmmrp_dct'] = self.X['libray_dir'] + 'grammar_sp.txt'
        self.X['wdneti_dct'] = self.X['libray_dir'] + 'wordnet.txt'
        self.X['wdnetp_dct'] = self.X['libray_dir'] + 'wordnet_sp.txt'
        
        #reference
        self.X['refere_mlf'] = self.label
        self.X['transf_mlf'] = self.X['result_dir'] + 'transcription.mlf'
        self.X['reflab_mlf'] = self.X['result_dir'] + 'clean_reference.mlf'
        self.X['consis_txt'] = self.X['result_dir'] + 'ASR_accuracy.txt'

        #language
        self.X['inlang_mlf'] = self.X['lanmdl_dir'] + 'sentences.txt'
        self.X['wrdcnt_txt'] = self.X['lanmdl_dir'] + 'wordcount.txt'
        self.X['outlan_txt'] = self.X['lanmdl_dir'] + 'lm.txt'
        self.X['lanres_mlf'] = self.X['lanmdl_dir'] + 'larvevocab_result.mlf'
        self.X['landct_dct'] = self.X['lanmdl_dir'] + 'dictionary_large.txt'
        self.X['tielst_txt'] = self.X['lanmdl_dir'] + 'tielist.txt'
        
        self.X['biwnet_txt'] = self.X['lanmdl_dir'] + 'biwnet.txt'
        self.X['bigram_txt'] = self.X['lanmdl_dir'] + 'bigram.txt'
        self.X['wdlist_txt'] = self.X['lanmdl_dir'] + 'wdlist.txt'
        self.X['bufdct_dct'] = self.X['lanmdl_dir'] + 'temp_dictionary_large.txt'
        #pattern
        self.X['flattn_mlf'] = self.X['lanmdl_dir'] + 'flat_mlf.mlf'
        self.X['flattn_dct'] = self.X['lanmdl_dir'] + 'flat_dictionary.txt'
        
    def feature(self):
        HTK()\
        .readSCP(self.X['corpus_dir'])\
        .writeSCP(self.X['wavlst_scp'],[]       , self.X['featur_dir'])\
        .writeSCP(self.X['wavhcp_scp'],['hcopy'], self.X['featur_dir'])
     
        try:    os.mkdir(self.X['featur_dir'])
        except: return
        
        SYS().cygwin('HCopy -T 1 -C "{}"  -S "{}" '.format(
            self.X['hcopie_cfg'],self.X['wavhcp_scp']))

    def exp_feature(self, wav_directory, wav_list):
        script = 'temp_hcopy.txt'
        feature_directory = wav_directory[:-1] + '_MFCC/'
        
        HTK()\
        .readSCP(wav_directory)\
        .writeSCP(wav_list,[]       , feature_directory)\
        .writeSCP(script,['hcopy'], feature_directory)
     
        try:    os.mkdir(feature_directory)
        except: return
        
        SYS().cygwin('HCopy -T 1 -C "{}"  -S "{}" '.format(
            self.X['hcopie_cfg'],script))
            
    def training_macros(self):
        macros = open(self.X['macros_hmm'],'w')
        if self.user_feature==True:
            macros.write(macrosf_user)
        else:
            macros.write(macrosf)
        map(lambda x: macros.write(x),open(self.X['markov_dir'] + 'vFloors').readlines())
        
    def training_hmmdef(self):
        models = open(self.X['models_hmm'],'w')
        text = open(self.X['hmmdef_hmm']).readlines()
        modelList =[t.rstrip('\n') for t in open(self.X['modeli_dct']).readlines()]
        for model in modelList:
            if model=='sil': 
                self.training_silence(models)
                return
            models.write('~h "' + model + '"\n')
            transcribe = False
            for line in text:
                if transcribe:  models.write(line)
                if line=='~h "hmmdef"\n': transcribe = True

    def training_silence(self, file):
        file.write('~h \"sil\"\n')
        file.write('<BEGINHMM>\n<NUMSTATES> 5\n')
        text = [t.rstrip('\n')  for t in open(self.X['hmmdef_hmm']).readlines()]
        state_mean_variance_gconst = []
        
        state_pos = [i for i in range(len(text)) if 'STATE' in text[i]]
        for j in [2,3,4]:
            file.write('<STATE> {}'.format(str(j)))
            for i in range(state_pos[1]+1,state_pos[2]):
                file.write(text[i])
        file.write(siltransition)
        
    def training_shortpause(self):
        models = open(self.X['models_hmm'],'a')
        text = open(self.X['models_hmm']).readlines()
        models.write('~h "sp"\n')
        models.write('<BEGINHMM>\n<NUMSTATES> 3\n<STATE> 2\n')
        state_pos = [i for i in range(len(text)) if 'STATE' in text[i]]
        for i in range(state_pos[1]+1,state_pos[2]):
                models.write(text[i])
        models.write(sptransition)

    def training(self, mode = 'default'):
        if mode == 'default':
            iter = [10, 10, 10, 1]
            t_script = self.X['addmix_scp']
            # initiation
            SYS().cygwin('HCompV -T 2 -D -C "{}"  -o "{}"  -f 0.01 -m -S "{}"  -M "{}"  "{}" '.format(\
                self.X['config_cfg'], 'hmmdef', 
                self.X['wavlst_scp'], self.X['markov_dir'], 
                self.X['protos_txt']
            ))
            self.training_macros()
            self.training_hmmdef()
            
            for i in range(iter[0]):
                SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(\
                    self.X['config_cfg'], self.X['phonei_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modeli_dct']
                ))
            # add short pause
            self.training_shortpause()
            SYS().cygwin('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(\
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], self.X['addsil_scp'],
                self.X['modelp_dct']
            ))
            for i in range(iter[1]):
                SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(\
                    self.X['config_cfg'], self.X['phonep_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modelp_dct']
                ))
            # increase mixture
            for i in range(iter[3]):
                SYS().cygwin('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                    self.X['macros_hmm'], self.X['models_hmm'],
                    self.X['markov_dir'], t_script,
                    self.X['modelp_dct']
                ))
                for i in range(iter[2]):
                    SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                        self.X['config_cfg'], self.X['phonep_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modelp_dct']
                    ))            
        elif mode == 'addmix':
            #addmix_all = """MU +1 {*.state[2-%d].mix}\n"""%(self.nState) ### big mistake
            addmix_all = """MU +1 {*.state[2-%d].mix}\n"""%(self.nState+1)
            open(self.X['addall_scp'], 'w').write(addmix_all)
            t_script = self.X['addall_scp']
            SYS().cygwin('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], t_script,
                self.X['modelp_dct']
            ))
            for i in range(10):
                SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                    self.X['config_cfg'], self.X['phonep_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modelp_dct']
                ))
        elif mode == 'keepmix':
            for i in range(10):
                SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                    self.X['config_cfg'], self.X['phonep_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modelp_dct']
                ))           
    def exp_training(self, iter = [10, 10, 10]):
        # initiation
        SYS().cygwin('HCompV -T 2 -D -C "{}"  -o "{}"  -f 0.01 -m -S "{}"  -M "{}"  "{}" '.format(\
            self.X['config_cfg'], 'hmmdef', 
            self.X['wavlst_scp'], self.X['markov_dir'], 
            self.X['protos_txt']
        ))
        self.training_macros()
        self.training_hmmdef()
        
        for i in range(iter[0]):
            SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(\
                self.X['config_cfg'], self.X['phonei_mlf'],
                self.X['wavlst_scp'], self.X['macros_hmm'],
                self.X['models_hmm'], self.X['markov_dir'],
                self.X['modeli_dct']
            ))
        # add short pause
        self.training_shortpause()
        SYS().cygwin('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(\
            self.X['macros_hmm'], self.X['models_hmm'],
            self.X['markov_dir'], self.X['addsil_scp'],
            self.X['modelp_dct']
        ))
        for i in range(iter[1]):
            SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(\
                self.X['config_cfg'], self.X['phonep_mlf'],
                self.X['wavlst_scp'], self.X['macros_hmm'],
                self.X['models_hmm'], self.X['markov_dir'],
                self.X['modelp_dct']
            ))
        # increase mixture
        for i in range(8):
            SYS().cygwin('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], self.X['addmix_scp'],
                self.X['modelp_dct']
            ))
            for i in range(iter[2]):
                SYS().cygwin('HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                    self.X['config_cfg'], self.X['phonep_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modelp_dct']
                ))
    def replace_string(self,path,old,new):
        text = open(path).readlines()
        W = open(path,'w')
        for line in text:
            line = line.replace(old, new)
            W.write(line)
        
    def report(self, t0):
        R = HTK()
        R.readMLF(self.X['result_mlf'], ['sp', 'sil', '<s>', '</s>'])
        R.readDCT(self.X['dictry_dct'], ['sp', 'sil', '<s>', '</s>'])
        W = open(self.X['evalua_txt'],'a')
        W.write('word_count ' + str(R.word_count())+'\n')
        W.write('word_entropy ' + str(R.entropy())+'\n')
        if self.label:
            A = HTK().readMLF(self.X['refere_mlf'], ['sp', 'sil', '<s>', '</s>'])
            R.renameMLF2MLF(A)
            R.writeMLF(self.X['transf_mlf'], dot3='.rec')
            #-e "???" <s> -e "???" </s>
            SYS().cygwin('HResults -e "???" sil -e "???" sp  -I "{}"  "{}"  "{}"  >> "{}" '.format (\
                self.X['reflab_mlf'], self.X['modelp_dct'], 
                self.X['transf_mlf'], self.X['consis_txt']
            ))
            W.write('conditional_entropy ' + str(R.conditional_entropy(A))+'\n')
            W.write('joint_entropy ' + str(R.joint_entropy(A))+'\n')
        # ms windows: t is wall seconds elapsed (floating point)
        # linux: t is CPU seconds elapsed (floating point)
        
        t= time.clock() - t0
        W.write('decode_time ' + str(t)+'\n')
        
    def matlab(self):
        contents = os.listdir(self.X['matlab_dir'])
        #print contents
        if self.dump in contents: return
        SYS().matlab("cd('./{}');addpath('.');clusterDetection({})"
            .format(self.X['matlab_dir'], "'../" + self.X['corpus_dir'] + "'"))
        print "generating IDump.txt, please rerun python after matlab finishes"
        quit()
            
    def bigram2wordnet(self):
        A = open(self.X['wdlist_txt'], 'w') 
        w2w = HTK().readMLF(self.X['answer_mlf']).word2wav 
        for word in w2w:
            A.write(word+'\n')
        A.write('</s>\n<s>\n')
        A.close()
        SYS().cygwin('sort ' + self.X['wdlist_txt'])
        self.lm_dict()
        
        SYS().cygwin('HLStats -s "<s>" "</s>" -b "{}"  -o "{}"  "{}" '.format(\
            self.X['bigram_txt'], self.X['wdlist_txt'], self.X['answer_mlf']
        ))
        
        SYS().cygwin('HBuild -s "<s>" "</s>" -n "{}"  "{}"  "{}" '.format(
            self.X['bigram_txt'], self.X['wdlist_txt'], self.X['biwnet_txt']
        ))
        
    def lm_build(self, order = 3 ):
        HTK().readMLF(self.X['answer_mlf'],['sil','sp']).writeMLF(self.X['inlang_mlf'],['lm'])
        SYS().cygwin('ngram-count -text "{}"  -write "{}"  -order "{}"  -vocab "{}" '.format(
            self.X['inlang_mlf'], self.X['wrdcnt_txt'], str(order), self.X['dictry_dct']
        ))
        SYS().cygwin('ngram-count -read "{}"  -order "{}"  -vocab "{}"  -lm "{}" '.format(
            self.X['wrdcnt_txt'], str(order), self.X['dictry_dct'], self.X['outlan_txt']
        ))
    def lm_dict(self):
        HTK()\
        .readDCT(self.X['dictry_dct'],['sil','sp'])\
        .writeDCT(self.X['landct_dct'])
        B = open(self.X['landct_dct'])
        text = B.readlines()
        A = open(self.X['landct_dct'],'w')
        A.write('<s> sil\n</s> sil\n')
        for line in text:
            A.write(line)
            

    def bi_testing(self, optList = []):
        t0 = time.clock()
        
        self.bigram2wordnet()
        
        SYS().cygwin('HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(\
            self.X['macros_hmm'], self.X['models_hmm'], 
            self.X['wavlst_scp'], self.X['config_cfg'], 
            self.X['biwnet_txt'], self.X['result_mlf'], 
            self.X['landct_dct'], self.X['modelp_dct']
        ))
        
        SYS().cygwin('HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}"  "{}"  "{}"  >> "{}" '.format(\
            self.X['answer_mlf'], self.X['modelp_dct'], 
            self.X['result_mlf'], self.X['accrcy_txt']
        ))
        self.report(t0)
        
    def exp_bi_testing(self, wav_list, result_mlf, nbest):
        self.feedback()
        self.bigram2wordnet()
        
        SYS().cygwin('HVite -D -n 4 "{}"  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(\
            nbest,
            self.X['macros_hmm'], self.X['models_hmm'], 
            wav_list            , self.X['config_cfg'], 
            self.X['biwnet_txt'], result_mlf,
            self.X['landct_dct'], self.X['modelp_dct']
        ))
    def lat_testing(self, optList = []):
        t0 = time.clock()
        self.bigram2wordnet()
        SYS().cygwin('HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l  "{}"  -i "{}"  -p 0.0 -s 0.0 -z txt -n 5 5 "{}"  "{}" '.format(\
            self.X['macros_hmm'], self.X['models_hmm'], 
            self.X['wavlst_scp'], self.X['config_cfg'], 
            self.X['biwnet_txt'], self.X['result_mlf'][:-11],
            self.X['result_mlf'], 
            self.X['landct_dct'], self.X['modelp_dct']
        ))
        self.report(t0)    
    def exp_testing(self, wav_list, result_mlf, nbest):
        SYS()\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmri_dct'],self.X['wdneti_dct']))\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'],self.X['wdnetp_dct']))
        SYS().cygwin('HVite -D -n 4 "{}"  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(\
            nbest,
            self.X['macros_hmm'], self.X['models_hmm'], 
            wav_list            , self.X['config_cfg'], 
            self.X['wdnetp_dct'], result_mlf,
            self.X['dictry_dct'], self.X['modelp_dct']
        ))  
    def external_testing(self, wav_list, result_mlf):
        SYS()\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmri_dct'],self.X['wdneti_dct']))\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'],self.X['wdnetp_dct']))
        SYS().cygwin('HVite -D  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(\
            self.X['macros_hmm'], self.X['models_hmm'], 
            wav_list            , self.X['config_cfg'], 
            self.X['wdnetp_dct'], result_mlf,
            self.X['dictry_dct'], self.X['modelp_dct']
        )) 
    def testing(self, optList = []):
        t0 = time.clock()
        
        SYS().cygwin('HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(\
            self.X['macros_hmm'], self.X['models_hmm'], 
            self.X['wavlst_scp'], self.X['config_cfg'], 
            self.X['wdnetp_dct'], self.X['result_mlf'], 
            self.X['dictry_dct'], self.X['modelp_dct']
        ))
        """wdnet shortpause is too large to generate
        SYS().cygwin("HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l '*' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" ".format(\
            self.X['macros_hmm'], self.X['models_hmm'], 
            self.X['wavlst_scp'], self.X['config_cfg'], 
            self.X['wdneti_dct'], self.X['result_mlf'], 
            self.X['dictry_dct'], self.X['modelp_dct']
        ))
        """ 
        SYS().cygwin('HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}"  "{}"  "{}"  >> "{}" '.format(\
            self.X['answer_mlf'], self.X['modelp_dct'], 
            self.X['result_mlf'], self.X['accrcy_txt']
        ))
        self.report(t0)
        
       
    def am_setup(self):
        HTK()\
        .readMLF (self.dump)\
        .convertMLF2DCT()\
        .writeMLF(self.X['result_mlf'],['sil', 'sp'])\
        .writeDCT(self.X['dictry_dct'],['sil', 'sp'])
    
    def init_setup(self):
        HTK()\
        .readMLF (self.dump)\
        .writeMLF(self.X['result_mlf'],['sil', 'sp'])\
        .writeMLF(self.X['answer_mlf'])\
        .writeMLF(self.X['phonei_mlf'],['phone','sil'])\
        .writeMLF(self.X['phonep_mlf'],['phone','sil','sp'])\
        .convertMLF2DCT(2)\
        .writeDCT(self.X['dictry_dct'],['sil', 'sp'])\
        .writeDCT(self.X['modeli_dct'],['model','sil'])\
        .writeDCT(self.X['modelp_dct'],['model','sp','sil'])\
        .writeDCT(self.X['grmmri_dct'],['grammar','sil'])\
        .writeDCT(self.X['grmmrp_dct'],['grammar','sp', 'sil'])

        SYS()\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmri_dct'],self.X['wdneti_dct']))\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'],self.X['wdnetp_dct']))
        
        self.training()
        self.testing()
        
    def feedback(self):
        HTK()\
        .readMLF (self.X['result_mlf'],['sp', 'sil','<s>','</s>'])\
        .readDCT (self.X['dictry_dct'])\
        .convertMLF2DCT()\
        .writeMLF(self.X['answer_mlf'])\
        .writeMLF(self.X['phonei_mlf'],['phone','sil'])\
        .writeMLF(self.X['phonep_mlf'],['phone','sil','sp'])\
        .writeDCT(self.X['dictry_dct'],['sil', 'sp'])\
        .writeDCT(self.X['modeli_dct'],['model','sil'])\
        .writeDCT(self.X['modelp_dct'],['model','sp','sil'])\
        .writeDCT(self.X['grmmri_dct'],['grammar','sil'])\
        .writeDCT(self.X['grmmrp_dct'],['grammar','sp', 'sil'])
        
        SYS()\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmri_dct'],self.X['wdneti_dct']))\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'],self.X['wdnetp_dct']))

    def lexicon_setup(self):
        #self.recognize_patterns()
        import pySFX
        #suffix.flatten_pattern(self.X['dictry_dct'],self.X['result_mlf'],self.X['flattn_dct'],self.X['flattn_mlf'])
        #suffix.parse_pattern(self.X['flattn_dct'],self.X['flattn_mlf'],self.X['dictry_dct'])
        pySFX.flatten_pattern(self.X['dictry_dct'],self.X['result_mlf'],self.X['dictry_dct'],self.X['result_mlf'])
        pySFX.parse_pattern(self.X['dictry_dct'],self.X['result_mlf'],self.X['dictry_dct'])

        
        HTK()\
        .readDCT(self.X['dictry_dct'],['sil', 'sp'])\
        .writeDCT(self.X['modeli_dct'],['model','sil'])\
        .writeDCT(self.X['modelp_dct'],['model','sil','sp'])\
        .writeDCT(self.X['grmmri_dct'],['grammar','sil'])\
        .writeDCT(self.X['grmmrp_dct'],['grammar','sp', 'sil'])        
        
        SYS()\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmri_dct'],self.X['wdneti_dct']))\
        .cygwin('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'],self.X['wdnetp_dct']))  
    
    def iteration(self, command, comment = '',do_slice = False):
        self.time = time.clock()
        self.record_comment('{}_{}'.format(str(self.offset+1), comment))
        if command == 'a': self.a()
        if command == 'a_mix': self.a_mix()
        #if command == 'a_keep': self.a_mix() ###big mistake
        if command == 'a_keep': self.a_keep()
        if command == 'al': self.al()
        if command == 'al_lat': self.al_lat()
        if command == 'x': self.x()
        if command == 'ax': self.ax()
        if do_slice:  self.slice()
        self.record_time()
        os.chdir(self.target)
        os.chdir('..')
        self.writeASR('{}_{}/'.format(str(self.offset+1), comment))
        self.offset += 1
    
    def initialization(self,comment = '0_initial_condition/'):
        self.clean()
        #self.matlab()
        self.feature()
        
        #self.am_setup() #obsolete, bad for large corpus
        self.init_setup()
        os.chdir(self.target)
        os.chdir('..')        
        self.writeASR('0_{}/'.format(comment))
        
    def a(self):
        self.feedback()
        self.training()
        self.testing()
        
    def a_mix(self):
        self.feedback()
        self.training('addmix')
        self.testing()
        
    def a_keep(self):
        self.feedback()
        self.training('keepmix')
        self.testing()        
               
    def al(self):
        self.feedback()
        self.training()
        self.bi_testing()
        
    def al_lat(self):
        self.feedback()
        self.training('keepmix')
        self.lat_testing()
        
    def x(self):
        self.feedback()
        self.lexicon_setup()
        self.testing()

    def ax(self):
        self.feedback()
        self.training()
        self.lexicon_setup()
        self.testing()
    '''
    def Four_Phase(self):
        self.initialization()
        while True:
            self.iteration('a','acoustic_iteration') 
            if evaluation.pass_consistency(self): break
        while True:
            self.iteration('al','linguistic_iteration') 
            if evaluation.pass_consistency(self): break
        while True:
            self.iteration('ax','lexical_iteration') 
            if evaluation.pass_consistency(self): break
    '''
