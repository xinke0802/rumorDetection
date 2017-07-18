"""
Create Cluster
Author: Yamin Tun
"""


from datasketch import MinHash
import itertools
from sklearn.metrics import jaccard_similarity_score
from collections import defaultdict
import csv
from graph_clusterEdge_funs import *

import pickle as pkl
import collections #import default_dict
import re


#Expected output
##cluster id:  3041611.0  num of tweets:  120
##cluster id:  1041601.0  num of tweets:  72
##cluster id:  2042017.0  num of tweets:  375

outputfile="/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Data"
#filename="/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Code/my_visual/word_cloud/cluster_text2.p"
inputfile= "my_cluster_database.p"


rumor_id_list=[3041611.0, 1041601.0, 2042017.0]


tweet_list=["I'm super machine. Robot indeed. I'm tr'ing",
            "I'm super machine. Robot indeed. I'm supering",
            "I'm Ruby. Robot indeed. I'm stupid" ,
            "You should go home far come on.",
            "I am stuck you know that right?", 
            "I am beautiful you know that right?", 
            "I am perfect you know that right?",
            "You tell me and go home far really"]


newFileName='tweet_cluster_edges_3041611_1041601_2042017_0.6Thres_50nodes.csv'


nTweets=50 #number of tweets ecpected for each clusters

#[3041611.0, 1041601.0, 2042017.0]

# The test code...
if __name__ == "__main__":

    with open(inputfile, 'rb') as f:
        my_cluster_database=pkl.load(f)

    tweet_list=[]

    for i in range(len(rumor_id_list)):

        tweet_list=tweet_list+ my_cluster_database[rumor_id_list[i]][:nTweets]
    
    node_database=defaultdict(int)
   
    idno=0
    
    #create node database dictionary- 
    # -key:id from 0 to #nodes-1,
    # -values: string tweets
    for tweet in tweet_list:
        node_database[idno]=Data(idno,tweet)
        idno=idno+1
    
    #create list of node id pair (used combination)
    nodePair_List=list(itertools.combinations([node.name for node in node_database.values()], r=2))
    #print "pairs of node names:",nodePair_List#[(p[0].name,p[1].name) for p in nodePair_List]


    print len(nodePair_List)
    
    counter=0
    
    #create csv file with two columns of vertices that specify all edges in graph:
    # column 1: vertex 1
    # column 2: vertex 2
    with open(newFileName, 'wb') as csvfile:
        graphwriter = csv.writer(csvfile, delimiter=',')
        thres=0.6 #threshold of similiarity score for creating edges
        #store similarity score in node Pair dictionary
        for nodeIDP in nodePair_List:
        
            similarity=jaccard(get_trigram(node_database[nodeIDP[0]].tweet), get_trigram(node_database[nodeIDP[1]].tweet))

            counter=counter+1
            
            #create edge if similiarity of nodes are above thresohld
            if similarity>=thres:        
                #add link (optional)
                node_database[nodeIDP[0]].add_link(node_database[nodeIDP[1]]  )
                #write csv file
                graphwriter.writerow( (nodeIDP[0],nodeIDP[1])  )

                
            else:
                print counter
        
