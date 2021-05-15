
import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords

TEXT = ''
WINDOW_SIZE = 5
DAMPING_CONSTANT = 0.85
SENTENCE_TOKENS = list()
WORD_TOKENS = list()
ALPHABET = 'abcdefghijklmnopqrstuvwxyz-'
STOPWORDS = set(stopwords.words('english'))
VOCABULARY = set()
CONTEXT_GRAPH = None
WORD_RANK = dict()
WORD_RANK_PRECISION = 3

def LoadText ( path ) :
    global TEXT
    try : text_file = open(path, 'r')
    except : return False
    TEXT = text_file.read()
    text_file.close()
    return True

def PreprocessText ( ) :
    global SENTENCE_TOKENS, WORD_TOKENS, VOCABULARY
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    sent_tokens = nltk.tokenize.sent_tokenize(TEXT.lower())
    for sent in sent_tokens :
        word_tokens = nltk.tokenize.word_tokenize(sent)
        tagged = nltk.pos_tag(word_tokens)
        refined_word_tokens = []
        for word , tag in tagged :
            refined_token = ''.join(sym for sym in word if sym in ALPHABET)
            if ( refined_token != "" and refined_token[0] != '-' and refined_token[-1] != '-' and not refined_token in STOPWORDS ) :
                if tag[0] in ['N'] :
                    refined_word_tokens.append(refined_token)
        if len(refined_word_tokens) > 0 :
            SENTENCE_TOKENS.append(refined_word_tokens)
        WORD_TOKENS += refined_word_tokens
    VOCABULARY = set(WORD_TOKENS)

def MakeGraph ( ) :
    global CONTEXT_GRAPH
    CONTEXT_GRAPH = nx.Graph()
    for word in VOCABULARY :
        CONTEXT_GRAPH.add_node(word)
    
    for sent in SENTENCE_TOKENS :
        if len(sent) <= WINDOW_SIZE :
            for i in range(len(sent)) :
                for j in range(i+1, len(sent)) :
                    CONTEXT_GRAPH.add_edge(sent[i], sent[j])
        else :
            for start in range(len(sent)-WINDOW_SIZE+1) :
                window = sent[start:start+WINDOW_SIZE]
                for i in range(len(window)) :
                    for j in range(i+1, len(window)) :
                        CONTEXT_GRAPH.add_edge(window[i], window[j])

def PerformPageRankStyleAlgo ( ) :
    global WORD_RANK
    WORD_RANK = dict()
    for word in VOCABULARY :
        WORD_RANK[word] = 1 / len(VOCABULARY)
    
    max_difference = float('inf')
    while max_difference > 10**(-1*WORD_RANK_PRECISION) :
        WORD_RANK_new = dict()
        max_change_in_rank = 0.0
        for word in VOCABULARY :
            total = 0.0
            neighbours = list(CONTEXT_GRAPH.neighbors(word))
            for ng in neighbours :
                total += WORD_RANK[ng] / len(list(CONTEXT_GRAPH.neighbors(ng)))
            WORD_RANK_new[word] = 1 - DAMPING_CONSTANT + DAMPING_CONSTANT * total
            diff = abs(WORD_RANK_new[word] - WORD_RANK[word])
            if ( max_change_in_rank < diff ) :
                max_change_in_rank = diff
        WORD_RANK = WORD_RANK_new
        max_difference = max_change_in_rank

    WORD_RANK_copy = WORD_RANK.copy()
    WORD_RANK = dict()
    for word, rank in sorted(WORD_RANK_copy.items(), key=lambda x: x[1])[::-1] :
        WORD_RANK[word] = round(rank, WORD_RANK_PRECISION)


print('\n\n', end='')
while True :
    file_path = input('\n\tENTER TEXT DOCUMENT PATH : ')
    if ( not LoadText(file_path) ) :
        print( "\n\t\t [ TEXT DOCUMENT COULD NOT BE READ SUCCESFULLY ! ] " )
        continue
    PreprocessText()
    MakeGraph()
    PerformPageRankStyleAlgo()
    try : count = round(eval(input('\tENTER NO. OF KEYWORDS : ')))
    except :
        print( "\n\t\t [ INCORRECT FORMAT FOR TOTAL KEYWORDS COUNT ! ] " )
        continue
    if ( count < 0 ) :
        print( "\n\t\t [ TOTAL KEYWORDS COUNT CANNOT BE NEGATIVE ! ] " )
        continue
    if ( count > round(0.25 * len(VOCABULARY)) ) :
        print( "\n\t\t [ TOO MANY KEYWORDS ! ] " )
        continue
    ranked_words = list(WORD_RANK.keys())
    print('\n', end='')
    to_be_displayed = list()
    for rank in range(min(count, len(ranked_words))) :
        to_be_displayed.append(ranked_words[rank])
    to_be_displayed = sorted(to_be_displayed, key = lambda s: s.casefold())
    for word in to_be_displayed :
        print( "\t\t>> " + word[0].upper() + word[1:].lower() )
    if ( len(to_be_displayed) > 0 ) :  print('\n', end='')
    
