"""
Save tweet cluster into pickle
Author: Yamin Tun
"""


import pickle as pkl
import collections 
import re

outputfile="/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Data/Phase2/merged_data"
inputfile= "my_cluster_database.p"


rumor_id_list=[3041611.0, 1041601.0, 2042017.0]
non_rumor_id_list=[3041602.0,  1041710.0,  2041707.0]


my_cluster_database=collections.defaultdict()

for i in range(len(rumor_id_list)) :
    my_cluster_database[rumor_id_list[i]]=[]
    
with open(outputfile, 'rb') as f:
        words_activation_list = pkl.load(f)


for i in range(len(words_activation_list)) :

    if words_activation_list.iloc[i]['cId']==rumor_id_list[0] or words_activation_list.iloc[i]['cId']==rumor_id_list[1] or words_activation_list.iloc[i]['cId']==rumor_id_list[2]: 
    #append to the list
       my_cluster_database[words_activation_list.iloc[i]['cId']]= my_cluster_database[words_activation_list.iloc[i]['cId']]+ [words_activation_list.iloc[i]['tweet']]

#printing info what I have saved
for i in range (len(rumor_id_list)):        
    print "cluster id: ",rumor_id_list[i], " num of tweets: ",len(my_cluster_database[rumor_id_list[i]] )

    
#saving values
pkl.dump(my_cluster_database, open(inputfile,"wb"))


