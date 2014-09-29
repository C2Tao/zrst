import numpy as np
import pyHTK


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
    '''
    def write(self,path ='temp.mlf',dot='.rec'):
        if dot == '.lab': self.mlf_type = False
        M = open(path,'w')
        M.write('#!MLF!#\n')
        for i in range(len(self.tag_list)):
            M.write('"*/'+self.wav_list[i]+dot+'"\n')
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
        self.path = path
    '''
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
class Query(pyHTK.MLF):
    def __init__():
        None       
 
