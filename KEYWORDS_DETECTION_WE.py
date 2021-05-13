
import nltk
import numpy as np
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from scipy.cluster.vq import whiten, kmeans

WORD_2_VEC_MODEL = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)
TEXT = ''
STOPWORDS = set(stopwords.words('english'))
SENTENCE_TOKENS = list()
WORD_TOKENS = list()
ALPHABET = "abcdefghijklmnopqrstuvwxyz-"
VOCAB_VECTORS = dict()
VOCAB = set()
CLUSTERING_DATA = None
TOTAL_KEYWORDS_COUNT = 0
TOTAL_KEYWORDS_COUNT_TOLERANCE = 0.3
CLUSTER_COUNT_FRACTION = 0.25
PROPER_NOUNS_FREQ = dict()

def LoadText ( path ) :
    global TEXT
    try : text_file = open(path, 'r')
    except : return False
    TEXT = text_file.read()
    text_file.close()
    return True

def NotAllCaps ( word ) :
    CAPITALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in word :
        if ( not letter in CAPITALS ) : return True
    return False

def PreprocessText ( ) :
    global SENTENCE_TOKENS, WORD_TOKENS, VOCAB, PROPER_NOUNS_FREQ
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    PROPER_NOUNS_FREQ = dict()
    sent_tokens = nltk.tokenize.sent_tokenize(TEXT)
    for sent in sent_tokens :
        word_tokens = nltk.tokenize.word_tokenize(sent)
        tagged = nltk.pos_tag(word_tokens)
        refined_word_tokens = []
        for word , tag in tagged :
            refined_token = ''.join(sym for sym in word if sym.lower() in ALPHABET)
            if ( refined_token != "" and refined_token[0] != '-' and refined_token[-1] != '-' and not refined_token.lower() in STOPWORDS ) :
                if tag in ['NN', 'NNS'] :
                    refined_word_tokens.append(refined_token)
                    refined_word_tokens.append(refined_token.lower())
                elif tag in ['NNP', 'NNPS'] :
                    if ( not '.' in word and NotAllCaps(word) ) :
                        refined_token = word[0].upper() + word[1:].lower()
                    else : refined_token = word
                    if refined_token in PROPER_NOUNS_FREQ :  PROPER_NOUNS_FREQ[refined_token] += 1
                    else : PROPER_NOUNS_FREQ[refined_token] = 1
        if len(refined_word_tokens) > 0 :
            SENTENCE_TOKENS.append(refined_word_tokens)
        WORD_TOKENS += refined_word_tokens
    VOCAB = set(WORD_TOKENS)

def GenerateDataForClustering ( ) :
    global VOCAB_VECTORS, CLUSTERING_DATA, VOCAB
    VOCAB_VECTORS = dict()
    CLUSTERING_DATA = list()
    for word in VOCAB :
        try :
            WORD_2_VEC_MODEL[word]
            VOCAB_VECTORS[word] = WORD_2_VEC_MODEL[word]
            CLUSTERING_DATA.append(list(WORD_2_VEC_MODEL[word]))
        except :
            continue
    CLUSTERING_DATA = whiten(np.array(CLUSTERING_DATA))


def GetKeywords ( ) :
    global TOTAL_KEYWORDS_COUNT, PROPER_NOUNS_FREQ
    if ( TOTAL_KEYWORDS_COUNT == 0 ) :
        return set()
    
    PROPER_NOUNS_FREQ_copy = PROPER_NOUNS_FREQ.copy()
    PROPER_NOUNS_FREQ = dict()
    for word, freq in sorted(PROPER_NOUNS_FREQ_copy.items(), key=lambda x: x[1])[::-1] :
        PROPER_NOUNS_FREQ[word] = freq

    PROPER_WORD_SET = set()
    PROPER_WORD_IMPORTANCE_THRESHOLD = 4
    for word in PROPER_NOUNS_FREQ :
        if PROPER_NOUNS_FREQ[word] >= PROPER_WORD_IMPORTANCE_THRESHOLD : 
            PROPER_WORD_SET.add(word)
            if ( len(PROPER_WORD_SET) == TOTAL_KEYWORDS_COUNT ) :
                return PROPER_WORD_SET
    
    TOTAL_KEYWORDS_COUNT = max(0, TOTAL_KEYWORDS_COUNT-len(PROPER_WORD_SET))
    if ( TOTAL_KEYWORDS_COUNT == 0 ) :
        return PROPER_WORD_SET
    
    CURRENT_KEYWORD_SET = set(VOCAB_VECTORS.keys())
    CLUSTER_COUNT = round(CLUSTER_COUNT_FRACTION * len(VOCAB_VECTORS))
    iteration = 0
    MAX_ITERATIONS = 20
    tolerance = round(TOTAL_KEYWORDS_COUNT_TOLERANCE * TOTAL_KEYWORDS_COUNT)
    while abs(len(CURRENT_KEYWORD_SET)-TOTAL_KEYWORDS_COUNT) > tolerance :
        iteration += 1
        if ( iteration > MAX_ITERATIONS ) : break
        upper_limit = TOTAL_KEYWORDS_COUNT + tolerance
        lower_limit = TOTAL_KEYWORDS_COUNT - tolerance

        if len(CURRENT_KEYWORD_SET) > upper_limit :
            centroids = kmeans(CLUSTERING_DATA, CLUSTER_COUNT)[0]
            keywords = set()
            for centroid in centroids :
                candidates = WORD_2_VEC_MODEL.most_similar([np.array(centroid)])
                for candidate in candidates :
                    if ( candidate[0] in VOCAB_VECTORS.keys() ) :
                        word = candidate[0][0].upper() + candidate[0][1:].lower()
                        keywords = keywords.union(set([word]))
                        break
            CURRENT_KEYWORD_SET = CURRENT_KEYWORD_SET.intersection(keywords)
        
        if len(CURRENT_KEYWORD_SET) < lower_limit :
            while len(CURRENT_KEYWORD_SET) < lower_limit :
                centroids = kmeans(CLUSTERING_DATA, CLUSTER_COUNT)[0]
                keywords = set()
                for centroid in centroids :
                    candidates = WORD_2_VEC_MODEL.most_similar([np.array(centroid)])
                    for candidate in candidates :
                        if ( candidate[0] in VOCAB_VECTORS.keys() ) :
                            word = candidate[0][0].upper() + candidate[0][1:].lower()
                            keywords = keywords.union(set([word]))
                            break
                CURRENT_KEYWORD_SET = CURRENT_KEYWORD_SET.union(keywords)
    
    CURRENT_KEYWORD_SET = CURRENT_KEYWORD_SET.union(PROPER_WORD_SET)
    return CURRENT_KEYWORD_SET

print('\n\n', end='')
while True :
    file_path = input('\n\tENTER TEXT DOCUMENT PATH : ')
    if ( not LoadText(file_path) ) :
        print( "\n\t\t [ TEXT DOCUMENT COULD NOT BE READ SUCCESFULLY ! ] " )
        continue
    PreprocessText()
    GenerateDataForClustering()
    try : TOTAL_KEYWORDS_COUNT = round(eval(input('\tENTER NO. OF KEYWORDS (APPROX.) : ')))
    except :
        print( "\n\t\t [ INCORRECT FORMAT FOR TOTAL KEYWORDS COUNT ! ] " )
        continue
    if ( TOTAL_KEYWORDS_COUNT < 0 ) :
        print( "\n\t\t [ TOTAL KEYWORDS COUNT CANNOT BE NEGATIVE ! ] " )
        continue
    if ( TOTAL_KEYWORDS_COUNT > round(0.25 * len(VOCAB)) ) :
        print( "\n\t\t [ TOO MANY KEYWORDS ! ] " )
        continue
    keywords = list(GetKeywords())
    keywords = sorted(keywords, key = lambda s: s.casefold())
    print('\n', end='')
    for word in keywords :
        print( "\t\t>> " + word[0].upper() + word[1:] )
    if ( len(keywords) > 0 ) :  print('\n', end='')
