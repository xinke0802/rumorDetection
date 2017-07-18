"""
Util functions for clustering and feature extraction
Author: Yamin Tun, Rose John
"""

import string, re
import nltk

from nltk import word_tokenize
from nltk.util import ngrams

table = string.maketrans("","")

regex_str = [
    r'<[^>]+>', # HTML tags
    r'RT\s*(?:@[\w_]+)', #RT @-Retweets
    r'(?:@[\w_]+)', #@-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]


class Data(object):
    def __init__(self, name, tweet):
        self.__name  = name
        self.__tweet = tweet
        self.__links = set()
        
    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)
    
    @property
    def tweet(self):
        return self.__tweet
    
    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)

# The function to look for connected components.
def connected_components(nodeID_list,node_database):

    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set([node_database[idno] for idno in nodeID_list])

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:

            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result

def tokenize(s):
    #do traditional tokenizing
   # s=tokens_re.findall(s)
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

    #strip all punctuations
    
    s = s.translate(table, string.punctuation) #http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

    return tokens_re.findall(s)

def jaccard(s1,s2):
#     print "intersect: ",(s1.intersection(s2))
#     print "union: ",((s1.union(s2)))
    return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    
def get_trigram(tweet):
    
    word_list=tokenize(tweet)
    trigrams = set(nltk.trigrams(word_list))

    return trigrams

def get_minHash(wordlist1, wordlist2):
    
    t1=time.time()
    m1, m2 = MinHash(), MinHash()
    for d in wordlist1:
        m1.digest(sha1(d.encode('utf8')))
    for d in wordlist2:
        m2.digest(sha1(d.encode('utf8')))

    #print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))
    return m1.jaccard(m2)
