def ap(p_list):          
    #average precision
    #assert(p_list[0]==0)
    #p_list.pop(0)

    pos = 0.0
    if not p_at: return 0
    for i in range(len(p_list)):
        pos += 1.0*(i+1)/p_list[i]
        #print (i+1),p_list[i]
    return pos/len(p_list)

def p_at(p_list,at = 99999):
    #p@i
    #assert(p_list[0]==0)
    #p_list.pop(0)
    at =  min(at,max(p_list))
    
    hit = 0
    for i in range(len(p_list)):
        if p_list[i] >at: break
        hit+=1
    return 1.0*hit/at

def eer(p_list):
    #equal error rate
    if not p_list: return 0
    N = len(p_list)
    pat = p_list
    while True:
        recall = 1.0*len(pat)/N
        precis = 1.0*len(pat)/(pat[-1])
        #print recall,precis
        if len(pat)==1 :break
        if recall <= precis:break
        pat = pat[:-1]
    return 1-recall

#note that first entry '0' has been removed from the test set
#print EER([1,3,7,11])
#print p_at([0,1,7,9,11,13,18],5)
#print ap([0,1,2,4,5])