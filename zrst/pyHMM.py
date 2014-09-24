from pprint import pprint
import math

def dtw_distance(x,y,dist_metric):
    #xy = [[dist_metric(x[i],y[j]) for i in range(len(x))] for j in range(len(y))]
    if len(x) >len(y):
        x,y = y,x
    dtw = [[0.0 for j in range(len(y)+1)] for i in range(len(x)+1)]
    for i in range(len(x)):
        dtw[i+1][0] = 9999999.9
    parent = [[[] for j in range(len(y)+1)] for i in range(len(x)+1)]
    for i in range(len(x)):
        for j in range(len(y)):
            cost = dist_metric(x[i],y[j])
            
            cindex = (i+1,j),(i,j+1),(i,j)
            #candidate = dtw[i+1][j],dtw[i][j+1],dtw[i][j]
            candidate = [dtw[k[0]][k[1]] for k in cindex]
            mindex = candidate.index(min(candidate))
            #print i,j,cindex[mindex],candidate,len(x),len(y)
            dtw[i+1][j+1] = cost + min(candidate)
            parent[i+1][j+1] = cindex[mindex]
    mdist = min(dtw[len(x)])
    best = dtw[len(x)].index(mdist)
    #print best
    #pprint(dtw)
    path = []
    xindex = len(x)
    yindex = best
    while xindex:
        path.append((xindex,yindex))
        xindex, yindex = parent[xindex][yindex]
    path.reverse()
    path = map(lambda x: (x[0]-1,x[1]-1) ,path)
    return mdist,path
    
def parseHMM(model):
    lines = open(model).readlines()
    M = open(model)
    def getint(string):
        return int(string.split()[1])
    hmm = dict({})
    model_name= ''
    mode = ''
    gauss = ''
    for line in lines:
        line = M.readline()
        line = line.strip('\n')
        if '~h' in line:
            model_name = line.split()[1][1:-1]
            hmm[model_name] = dict({})
            continue
        if not model_name: continue
        if '<NUMSTATES>' in line:
            num_state = getint(line)
            hmm[model_name]['num_state'] = num_state
        elif '<STATE>' in line:
            state = getint(line)
            hmm[model_name][state] = dict({})
        elif '<NUMMIXES>' in line:
            num_gauss = getint(line)
            hmm[model_name][state]['num_gauss'] = num_gauss
        elif '<MIXTURE>' in line:
            gauss = getint(line)
            weight = float(line.split()[2])
            hmm[model_name][state][gauss] = dict({})
            hmm[model_name][state][gauss]['weight'] = weight
        elif '<MEAN>' in line:
            mode = 'mean'
        elif '<VARIANCE>' in line:
            mode = 'variance'
        elif '<TRANSP>' in line:
            assert int(line.split()[1]) == hmm[model_name]['num_state']
            hmm[model_name]['trans'] = dict({})
            mode = 'trans'
            trans_row = 0
        else:
            if mode == 'mean':
                try:
                    hmm[model_name][state][gauss]['mean'] = map(float,line.split())
                    #hmm[model_name][state][gauss]['mean'] = line.split()
                except:
                    hmm[model_name][state]['num_gauss'] = 1
                    gauss,weight = 1,1
                    hmm[model_name][state][gauss] = dict({})
                    hmm[model_name][state][gauss]['weight'] = weight
                    hmm[model_name][state][gauss]['mean'] = map(float,line.split())
                mode = ''
            elif mode == 'variance':
                hmm[model_name][state][gauss]['variance'] = map(float,line.split())
                mode = ''
            elif mode == 'trans':
                hmm[model_name]['trans'][trans_row] = map(float,line.split())
                trans_row += 1
                if trans_row == hmm[model_name]['num_state']-1:   mode = ''
    return hmm
#H = parseHMM('64_l5034_lab/hmm/models')
#pprint(H['p1'])

def parseBI(bi):
    lines = open(bi).readlines()
    id2word = dict({})
    bigram = dict({})
    null_list_end = dict({})
    null_list_beg = dict({})
    null_id = 0
    for line in lines:
        line = line.strip('\n')
        if line[0]=='I':
            id = int(line.split()[0].split('=')[1])
            word = line.split()[1].split('=')[1]
            id2word[id] = word
            
        if line[0]=='J':
            begi, endi = int(line.split()[1].split('=')[1]), int(line.split()[2].split('=')[1])
            beg, end = id2word[begi], id2word[endi]
            log10likelihood = float(line.split()[3].split('=')[1])
            if begi==null_id:
                null_list_end[end] = log10likelihood
            if endi==null_id:
                null_list_beg[beg] = log10likelihood
                continue
            bigram[beg,end] = log10likelihood
    for beg in null_list_beg:
        for end in null_list_end:
            if (beg,end) in bigram:
                bigram[beg,end] += null_list_beg[beg] + null_list_end[end]
            else:
                bigram[beg,end] = null_list_beg[beg] + null_list_end[end]
    return bigram
#B = parseBI('64_l5034_lab/lm/biwnet.txt')
#print B

def parseLEX(lex):
    lines = open(lex).readlines()
    lex = dict({})
    for line in lines:
        line = line.strip('\n')
        lex[line.split()[0]] = line.split()[1:]
    return lex
#L = parseLEX('64_l5034_lab/library/dictionary.txt')    
#print L

def HTK2CHAN(input_dir, output_parameter):
    chan = open(output_parameter, 'w')
    lex = parseLEX(input_dir + 'library/dictionary.txt')
    bigram = parseBI (input_dir + 'lm/biwnet.txt')
    hmm = parseHMM(input_dir + 'hmm/models')
    rsn = 13#real_state_number = 13
    dim = 39
    phone_vector = []
    pos2word = dict({})
    pos2phone = dict({})
    # maps state back to original word
    i = 0 
    for word in lex:
        if word[0] != 'p': continue
        for phone in lex[word]: 
            phone_vector.append(phone)
            
            for j in range(rsn):
                pos2word[i*rsn+j] = word
                pos2phone[i*rsn+j] = phone
            i += 1
    assert len(pos2word)==len(phone_vector)*rsn
    
    phone_uniq = list(set(phone_vector))
    
    phone_map = dict({})
    state_map = dict({})
    #maps phone to state_sequence 
    #and state to corresponding phone
    i = 0
    for phone in phone_uniq:
        phone_map[phone] = range(rsn*i,rsn*(i+1))
        for j in range(rsn*i,rsn*(i+1)):
            state_map[j] = phone
        i += 1

    state_vector = []
    for phone in phone_vector:
        state_vector += phone_map[phone]
    
    #pi_map = ['*' for i in range(rsn*len(state_vector))]
    pi_vector = ['*' for i in range(len(state_vector))]
    for arc in bigram:
        if arc[0] == '</s>' or arc[1] == '</s>': continue
        if arc[0] == '<s>':
            pi_vector[phone_vector.index(lex[arc[1]][0]) * rsn] = bigram[arc]
    pi_sum = 0
    for p in pi_vector:
        if p != '*':  pi_sum += 10**p
    print 'pi_sum',pi_sum
    for p in pi_vector:
        if p != '*':  p = math.log10((10**p)/pi_sum)
    
    pair_dict = set([]) # record all possible phone pairs in the lexicon
    for word in lex:
        if word[0] != 'p': continue
        for i in range(len(lex[word])-1):
            pair_dict.add((lex[word][i],lex[word][i+1]))
    '''       
    state2word = set([]) # maps state back to original word
    for word in lex:
        if word[0] != 'p': continue
        .append(word)
            state2word[state] = word
    '''
    
    chan.write('HMMSet 1\n')
    chan.write('HMM ascii\n')
    chan.write('nstate: '+str(rsn * len(phone_map))+'\n')
    chan.write('pdf_weight: 1\n')
    chan.write('gmindex: ')
    for state in state_vector:
        chan.write(str(state) + ' ')
    chan.write('\n')
    
    chan.write('pi: ')
    for pi in pi_vector:
        if pi == '*':  chan.write('0 ')
        else:          chan.write(str(10**pi)+' ')
    chan.write('\n')
    
    chan.write('trans:\n')
    for i in range(len(state_vector)):
        #local position in HMM
        s_from = state_vector[i]
        l_from = s_from % rsn
        #p_from = state_map[s_from]
        p_from = pos2phone[i]
        w_from = pos2word[i]
        row_output = []
        for j in range(len(state_vector)):
            s_to = state_vector[j]
            l_to = s_to % rsn
            #p_to = state_map[s_to]
            p_to = pos2phone[j]
            w_to = pos2word[j]
            #chan.write(' ')
            
            if p_from == p_to and w_from == w_to:
                row_output.append(hmm[p_from]['trans'][l_from + 1][l_to + 1])
                #chan.write('h ')
                #write the hmm transition prob
            elif l_from == rsn-1 and l_to == 0 and w_from == w_to:
                row_output.append(hmm[p_from]['trans'][l_from + 1][rsn + 1])
                #chan.write('p ')
                #write the hmm transition prob of last state to state rsn+1
            elif l_from == rsn-1 and l_to == 0 and w_from != w_to:
                row_output.append(10**bigram[pos2word[i],pos2word[j]])
                #chan.write('b ')
                #write bigram transition prob
            else:
                row_output.append(0)
                
        #row_sum = sum(row_output)
        for row_element in row_output:
        #    row_element = row_element/row_sum
            chan.write(str(row_element)+' ')
        chan.write('\n')
    chan.write('GaussinaMixtureSet '+str(rsn * len(phone_map))+'\n')
    
    for s in state_vector:
        chan.write('GaussianMixture ascii\n')
        chan.write('dim: ' + str(dim) + '\n')
        chan.write('nmix: 1\n')
        chan.write('weight: 1\n')
        chan.write('gaussidx: ' + s + '\n')
        chan.write('EndGaussianMixture\n')
    
    chan.write('GaussianSet '+ str(rsn * len(phone_map)) + '\n')
    for phone in phone_map:
        #for state in phone_map[phone]:
        chan.write('Gaussian ascii\n')
        chan.write('dim: '+str(dim)+'\n')
        chan.write('mean: ')
        #for 
            
        chan.write('\n')
        chan.write('diag+\n')
        chan.write('cov:\n')
        #for 

        chan.write('EndGaussian\n')
   
#HTK2CHAN('64_l5034_lab/','test.txt')
def mylog(x):
    try:
        return math.log(x)
    except:
        return -999.9
def KLD_gss(A,B):
    varA = A['variance']
    varB = B['variance']
    invB = map(lambda x: 1/x,B['variance'])
    vAiB = []
    for i in range(len(varA)):
        vAiB.append(varA[i]*invB[i])
    term1 = sum(vAiB)
    
    uA   = A['mean']
    uB   = B['mean']
    #print uA,uB
    ud,udTcovud = [],[]
    for i in range(len(uA)):
        ud.append(uB[i]-uA[i])
    for i in range(len(ud)):
        udTcovud.append(ud[i]*invB[i]*ud[i])
    term2 = sum(udTcovud)
    
    logdetA, logdetB = 0,0
    for v in varA:  logdetA += mylog(v)
    for v in varB:  logdetB += mylog(v)
    term3 = logdetB-logdetA
    term4 = -len(uA)
    #print term1,term2,term3,term4
    return 0.5*(term1+term2+term3+term4)



def KLD_mix_bug(A,B):
    nA = A['num_gauss']
    nB = B['num_gauss']
    #print B['num_gauss'],B[1]
    #print A['num_gauss'],A[1]
    
    term = []
    if nA==1 and nB==1:
        return KLD_gss(A[1],B[1])
    for j in range(1,nA+1):
        a = A[j]
        top,bot = [],[]
        for i in range(1,nA+1):
            top.append(A[i]['weight']*math.exp(-KLD_gss(a,A[i])))
        for i in range(1,nB+1):
            bot.append(B[i]['weight']*math.exp(-KLD_gss(a,B[i])))
        term1,term2 = mylog(sum(top)), mylog(sum(bot))
            
        #print term1,term2
        term.append(a['weight']*(term1-term2))
    return 1.0*sum(term)
def KLD_mix(A,B):
    #nA = A['num_gauss']
    #nB = B['num_gauss']
    nA, nB = len(A)-1, len(B)-1
    ka,kb = [],[]#to skip the missing gaussians
    def skip_index(A):
        vA = sorted(A.keys())
        vA.remove('num_gauss')
        vA.insert(0,0)#insert dummy index
        return vA
    vA, vB = skip_index(A), skip_index(B)
    #print vA,vB
    
    term = []
    if nA==1 and nB==1:
        return KLD_gss(A[vA[1]],B[vB[1]])
    for j in range(1,nA+1):
        a = A[vA[j]]
        top,bot = [],[]
        for i in range(1,nA+1):
            top.append(A[vA[i]]['weight']*math.exp(-KLD_gss(a,A[vA[i]])))
        for i in range(1,nB+1):
            bot.append(B[vB[i]]['weight']*math.exp(-KLD_gss(a,B[vB[i]])))
        term1,term2 = mylog(sum(top)), mylog(sum(bot))
        #print term1,term2
        term.append(a['weight']*(term1-term2))
    return 1.0*sum(term)

def KLD_bid(A,B):
    return (KLD_mix(A,B) + KLD_mix(B,A))/2
