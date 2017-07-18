"""
Writing and reading tweet word clouds into pickles
Author: Yamin Tun
"""

import pickle as pkl
import collections #import default_dict
import re

outputfile="/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Data/Phase2/merged_data"
filename="/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Code/my_visual/word_cloud/cluster_text2.p"
inputfile= filename

mode= "w" #"w"= building and pickling dictionary, "r"=reading dictionary

if mode=="w":

        cluster_text=collections.defaultdict()

        with open(outputfile, 'rb') as f:
                words_activation_list = pkl.load(f)


        print words_activation_list['tId']
        
        for i in range(len(words_activation_list)) :


            #print words_activation_list.iloc[i]['cId'], " ",words_activation_list.iloc[i]['tweet']

            #not working well
            #removing usernames in the tweets
            tweet=re.sub('@.*.:', '', words_activation_list.iloc[i]['tweet']) 

            try:
                    cluster_text[words_activation_list.iloc[i]['cId']] =cluster_text[words_activation_list.iloc[i]['cId']]+ tweet

            except:
                    cluster_text[words_activation_list.iloc[i]['cId']] = tweet



        print len(cluster_text)

        pkl.dump(cluster_text, open(filename,"wb"))

        
elif mode=="r":

        with open(inputfile, 'rb') as f:
                cluster_text = pkl.load(f)

        #eg.
        print cluster_text[117041519]
        
