# -*- coding: utf-8 -*-
"""
Spyder Editor

SUBMITTED BY - SATYAM SEVANYA (15CS10040)
"""
import nltk
from nltk.corpus import brown

###############################################################################

########### MAKING DICTIONARY OF WORDS, TAGS and their PAIRS ##################
w_dict={}
t_dict={}
brown_tags_words = [ ]
for sent in brown.tagged_sents()[:-100]:
    brown_tags_words.append( ("START", "START") )
    brown_tags_words.extend([ (tag, word) for (word, tag) in sent ])
    brown_tags_words.append( ("END", "END") )
    for ele in sent:
        w=ele[0]
        t=ele[1]
        if w not in w_dict:
            w_dict[w]=0
        if t not in t_dict:
            t_dict[t]=0
        w_dict[w]+=1
        t_dict[t]+=1

#print(brown_tags_words)
#print(len(brown_tags_words))
#print(w_dict)
#print(len(w_dict))
#print(t_dict)
#print(len(t_dict))

############################ EMISSION MATRIX ##################################
# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

emission={}
for t in t_dict:
    m={}
    for w in w_dict:
        m[w]=cpd_tagwords[t].prob(w)
    emission[t]=m

# uncomment this to print emission matrix for words w in some tag t
#for t in t_dict:
#    print(t,'->\n',emission[t],'\n\n')
###############################################################################

############################ TRANSMISSION MATRIX ##############################
brown_tags = [tag for (tag, word) in brown_tags_words ]

cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)


transmission={}
for tag1 in t_dict:
    m={}
    for tag2 in t_dict:
        m[tag2]=cpd_tags[tag1].prob(tag2)
    transmission[tag1]=m

# uncomment this to print transmission matrix from tag1 to tag2
#for tag1 in t_dict:
#    print(tag1,'->\n',transmission[tag1],'\n\n')

###############################################################################

############################## START MATRIX ###################################
start={}

for tag in t_dict:
    start[tag]=cpd_tags['START'].prob(tag)

#uncomment this to print start matrix from START to any tag
#print(start)

###############################################################################

############################## END MATRIX #####################################
end={}

for tag in t_dict:
    end[tag]=cpd_tags[tag].prob('END')

#uncomment this to print transition probability matrix from any tag to END
#print(end)

###############################################################################

############################ VITERBI ALGORITHM ################################
def viterbi (test_data=[], t_dict={}, w_dict={}, start={}, end={}, transmission={}, emission={}):
    
    for sentence in test_data:
        
        words=[]
        tags=[]
        
        for p in sentence:
            if p[0] not in w_dict:
                for t in t_dict:
                    emission[t][p[0]]=0.0000000001
            words.append(p[0])
            tags.append(p[1])
        
        sentlen = len(sentence)
            
        
        viterbi = [ ]
        backptr = [ ]
        
        this_viterbi = { }
        this_backptr = { }
        for tag in t_dict:
            this_viterbi[ tag ] = start[tag] * emission[tag][ words[0] ]
            this_backptr[ tag ] = 'START'
        
            
        viterbi.append(this_viterbi)
        backptr.append(this_backptr)
        
        
        for i in range(1, len(sentence)):
            this_viterbi = { }
            this_backptr = { }
            prev_viterbi = viterbi[-1]
    
            for tag in t_dict:
                best_previous = max(prev_viterbi.keys(),key = lambda prevtag: prev_viterbi[ prevtag ] * transmission[prevtag][tag] * emission[tag][words[i]])
        
                this_viterbi[tag] = prev_viterbi[best_previous] * transmission[best_previous][tag] * emission[tag][words[i]]
                this_backptr[ tag ] = best_previous
        
            viterbi.append(this_viterbi)
            backptr.append(this_backptr)
        
        
        prev_viterbi = viterbi[-1]
        best_previous = max(prev_viterbi.keys(),key = lambda prevtag: prev_viterbi[ prevtag ] * end[prevtag])
        
        prob_tagsequence = prev_viterbi[ best_previous ] * end[ best_previous]
        
        best_tagsequence = [ "END", best_previous ]
        backptr.reverse()
        
        current_best_tag = best_previous
        for bp in backptr:
            best_tagsequence.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        
        best_tagsequence.reverse()
        print( "The sentence was:", end = " ")
        for w in sentence: print( w, end = " ")
        print("\n")
        print( "The best tag sequence is:", end = " ")
        for t in best_tagsequence: print (t, end = " ")
        print("\n")
        print( "The viterbi probability of the best tag sequence is:", prob_tagsequence)
        
        j=0
        c=0
        for t in sentence:
            if t==sentence[j]:
                c+=1
            j+=1
        print( "The Accuracy was :", c,'/',sentlen,'=',c/sentlen)
        
        print('\n\n')
    
    
###############################################################################
        
########################### TESTING SENTENCES #################################
test_data= brown.tagged_sents()[-10:]
viterbi(test_data,t_dict,w_dict,start,end,transmission,emission)

###############################################################################




