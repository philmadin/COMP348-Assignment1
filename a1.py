import nltk
nltk.download('punkt')
nltk.download('reuters')
nltk.download('gutenberg')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Task 1 (1 mark)
from collections import Counter
def stem_counter(text):
    """Return a Python Counter of stems
    >>> c1 = stem_counter("Here is sentence 1. Here is sentence 2.")
    >>> sorted(c1.most_common())
    [('.', 2), ('1', 1), ('2', 1), ('here', 2), ('is', 2), ('sentenc', 2)]
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> c2 = stem_counter(emma[:1000])
    >>> sorted(c2.most_common(4))
    [(',', 13), ('had', 7), ('of', 12), ('the', 7)]
    >>> c2['had']
    7
    """

    text=nltk.sent_tokenize(text)

    wordList=[]
    for sen in text:
        for word in nltk.word_tokenize(sen):

            wordList.append(nltk.PorterStemmer().stem(str.lower(word)))
    counter = Counter(wordList)
    return counter

# Task 2 (1 mark)
def distinct_words_of_pos(text, pos):
    """Return the sorted list of distinct words with a given part of speech
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> d = distinct_words_of_pos(emma[:1000], 'NOUN')
    >>> len(d)
    42
    >>> d[:10]
    ['[', ']', 'affection', 'austen', 'between', 'blessings', 'caresses', 'clever', 'consequence', 'daughters']
    """
    text=nltk.sent_tokenize(text)
    textStructure=[]
    for sentence in text:
        temp=nltk.word_tokenize(sentence)
        textStructure.append(temp)

    tempword = nltk.pos_tag_sents(textStructure, tagset="universal")
    sorted_list = []
    for x in tempword:
        for y in x:
            if(str.lower(y[1])==str.lower(pos)):
                sorted_list.append(str.lower(y[0]))
    sorted_list = sorted(set(sorted_list))
    return sorted_list

# Task 3 (1 mark)
def most_common_pos_bigram(text):
    """Return the most common PoS bigram
    >>> most_common_pos_bigram("I saw the man with a telescope")
    ('DET', 'NOUN')
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> most_common_pos_bigram(emma[:1000])
    ('NOUN', '.')
    """

    text=nltk.sent_tokenize(text)
    textStructure=[]
    for sentence in text:
        temp=nltk.word_tokenize(sentence)
        textStructure.append(temp)

    tempword = nltk.pos_tag_sents(textStructure, tagset="universal")
    #print(tempword)
    tempword = [y for x,y in tempword[0]]
    tempword = nltk.bigrams(tempword)
    counter = Counter(tempword)
    return counter.most_common(1)[0][0]


# Task 4 (2 marks)
import re
def my_tokeniser(text):
    """Return the tokens
    #>>> my_tokeniser("This is a sentence")
    >>> my_tokeniser("This is a sentence")
    ['This', 'is', 'a', 'sentence']
    """
    regexp = "[a-z|A-Z|\-|:|\/]+|'\w+|[\d|,|\.|\-|\/]+|\(|\)"
    return re.findall(regexp,text)

# DO NOT MODIFY THE CODE BELOW
def baseline_tokeniser(text):
    "A baseline tokeniser"
    regexp = r'''[^\s]+'''
    return re.findall(regexp,text)

def false_negatives(tokens,target):
    """Return false negatives
    False negatives are items from the target that were not detected 
    as tokens"""
    return list(set(target)-set(tokens))

def false_positives(tokens,target):
    """Return the false positives
    False positives are the items that were wrongly identified as tokens"""
    return list(set(tokens)-set(target))

def my_score(raw,tokens,target):
    fn = false_negatives(tokens,target)
    fp = false_positives(tokens,target)
    score = len(fn)/len(target) + len(fp)/len(target)
    baseline_results = baseline_tokeniser(raw)
    fn_baseline = false_negatives(baseline_results,target)
    fp_baseline = false_positives(baseline_results,target)
    score_baseline = len(fn_baseline)/len(target) + len(fp_baseline)/len(target)
    return max(0,2*(score_baseline - score)/score_baseline)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    raw_reuters = nltk.corpus.reuters.raw(categories="corn")
    words_reuters = [w for s in nltk.sent_tokenize(raw_reuters)
                       for w in nltk.word_tokenize(s)]
    
    score = my_score(raw_reuters, my_tokeniser(raw_reuters), words_reuters)
    if score > 0 and score <= 0.5:
        rounded_score = 0.5
    elif score > 0.5 and score <= 1:
        rounded_score = 1
    elif score > 1 and score <= 1.5:
        rounded_score = 1.5
    else:
        rounded_score = 2
        
    print("Score of your tokeniser: %1.3f Rounded: %1.1f" % (score,
                                                             rounded_score))
