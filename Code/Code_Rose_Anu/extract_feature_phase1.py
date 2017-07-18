"""
Extract features of tweets and save them in a npy file
Author: Yamin Tun, Rose John, Anushree Ghosh

"""

import numpy as np
import pandas as pd
#import nltk as nltk
import cust_tokenizer as tk
import re
import sys
import os
import random
data_root = "../Data/Phase1/"

def load_tweets_label(filepath) :
	#read the file
    df = pd.read_csv(filepath,sep='\t')
    #extract only the tweet and label for now
    columns_of_interest = ['tweet','label']
    df = df.reindex(columns=columns_of_interest)
    #we are interested in only if the tweet is a rumor or not - so, 
    #we give it two classes - 0 - for not rumor, 1 - for rumor
    df.loc[df['label'] > 0, 'label'] = 1
    return df

#Added by Yamin
def load_tweets_label_boston(filepath) :
	#read the file
    df = pd.read_csv(filepath,sep='\t')
    #extract only the tweet and label for now
    columns_of_interest = ['rumor_cluster_id','label','tweet']
    df = df.reindex(columns=columns_of_interest)
    #we are interested in only if the tweet is a rumor or not - so, 
    #we give it two classes - 0 - for not rumor, 1 - for rumor
    df.loc[df['label'] > 0, 'label'] = 1
    return df

#feature extraction
#create empty array -- 7 columns - 1st 6 are features, 7th column is label
# col 0- count of question marks
# col 1- count of exclamation marks
# col 2- count of hashtags
# col 3- count of urls
# col 4- count of @ symbols
# col 5- count of question phrases
def extract_features(dataset) :
    num_of_tweets = dataset.shape[0]
    num_of_features = 7
    features = np.zeros((num_of_tweets, num_of_features))
    for i in range(0, num_of_tweets) :

        tweet = dataset.iloc[i]['tweet']
        label = dataset.iloc[i]['label']

        # set label column to tweets label
        features[i][6] = label

        
        # set other features after tokenizing and counting through all of them
        try:
            tweet_tokens = tk.preprocess(tweet)
        except:
            print "error: preprocess"#tweet_tokens
        
        for token in tweet_tokens:
            # count num of question marks
            if token=='?' :
                features[i][0]+=1
            # count num of exclamation marks
            if token=='!' :
                features[i][1]+=1
            # count num of hashtags
            if re.search( r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", token) :
                features[i][2]+=1
            # count num of urls
            if re.search( r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', token) :
                features[i][3]+=1
            # count retweet
            if re.search( r'RT\s*(?:@[\w_]+)', token) :
                features[i][4]+=1
        # count question phrases
        tweet=tweet.lower()
        # question phrase - is(that | this | it ) true
        if(re.search( r"is\s+(that|this|it)\s+\w*\s*(real|rl|true)", tweet)):
           features[i][5]+=1
        # question phrase - omg, o my god, oh my god, oh my gawd
        if(re.search( r"(omg)|(o)(h)*\s*(my)\s*(god|gawd)", tweet)):
           features[i][5]+=1
        if(re.search( r"(are|is)\s*(true)", tweet )):
           features[i][5]+=1
        # question phrase - really? real etc..
        if(re.search( r"real(l|ll|lly|ly)*(\?|\!)+", tweet)):
           features[i][5]+=1
        # tweet contains the word unconfirmed or debunked
        if(re.search(r"(unconfirm)(ed)*", tweet)) :
           features[i][5]+=1
        if(re.search(r"(looks)\s*(like)", tweet)):
           features[i][5]+=1
        if(re.search(r"(debunk)(ed)*|(dismiss)(ed)*", tweet)):
           features[i][5]+=1
        # question phrase - what?? or something similar
        if(re.search(r"wh[a]*t[?!][?1]*", tweet)):
           features[i][5]+=1
        # question phrase - rumor?
        if(re.search(r"(rumor\s*)(\?)|(hoax)|(gossip)|(scandal)", tweet)):
           features[i][5]+=1
        # question phrase - truth or true
        if(re.search(r"(tru)(e|th)|(false)|(fake)", tweet)):
           features[i][5]+=1
        # question phrase - truth or true
        if(re.search(r"(den)(ial|y|ied|ies)|(plausible)", tweet)):
           features[i][5]+=1
        if(re.search(r"(plausible)", tweet)):
           features[i][5]+=1
        # question phrase - truth or true
        if(re.search(r"belie(f|ve|ving)", tweet)):
           features[i][5]+=1
        # question phrase - truth or true
        if(re.search(r"(why)|(what)|(wht)|(when)|(where)|(whr)", tweet)):
           features[i][5]+=1
    return features

def trim_dataset_label(data):
 #   result = filter(lambda x: x[2] ==8, x)

    zeros = filter(lambda x: x[len(x)-1] ==0, data)
    zeros = np.vstack(zeros) #convert from list of array to 2d array- stack all arrays in the list

    print "\n before trimming:\n #nonrumors:", zeros.shape

    ones = filter(lambda x: x[len(x)-1] ==1, data)[:zeros.shape[0]]
 #   random.shuffle(ones)
    ones = np.vstack(ones)

    trimmed_data = np.vstack((zeros,ones))
    
    return trimmed_data

if __name__=='__main__':

    ## STEP 1 - LOAD DATA ##

    #load datafiles
    
    #we will use palin for training
    palin_raw = load_tweets_label(data_root+"palin.txt")
    
    #we will use airfrance and michelle for testing
    cell_raw = load_tweets_label(data_root+"cell.txt")
    obama_raw = load_tweets_label(data_root+"obama.txt")
    airfrance_raw = load_tweets_label(data_root+"airfrance.txt")
    michelle_raw = load_tweets_label(data_root+"michelle.txt")

    test_raw=pd.concat([cell_raw,obama_raw,airfrance_raw,michelle_raw])#cell_raw+obama_raw+airfrance_raw+michelle_raw
 #   michelle_airfrance_raw=michelle_raw
    #join the two dataframes
 #   michelle_airfrance_raw = michelle_raw.append(airfrance_raw)

    n_rumor=np.count_nonzero(obama_raw['label'])
    n_samples=len(obama_raw['label'])
    print "n_samples",n_samples
    print "#non_rumor", n_samples-n_rumor
    print "#rumor", n_rumor
    print "ratio of rumor", n_rumor*1.0/n_samples,'\n'

    sys.exit(0)


    #stats of the dataset
    print(type(cell_raw))
    print(cell_raw.shape)
    print(obama_raw.shape)
    print(airfrance_raw.shape)
    print(michelle_raw.shape)
    print ("PALIN DATASET : ", palin_raw.shape)
    print ("MICHELLE - AIRFRANCE - CELL - OBAMA DATASET : ", test_raw.shape)
    
    ## STEP 2 - FEATURE EXTRACTION ##
    train_set = extract_features(palin_raw)
    test_set = extract_features(test_raw)

    #np.save("test_set_obama_cell.npy",test_set)

    n_rumor=np.count_nonzero(test_set[:,6])
    n_samples=test_set.shape[0]
    print "#non_rumor", n_samples-n_rumor
    print "#rumor", n_rumor
    print "ratio of rumor", n_rumor*1.0/n_samples,'\n'

   # np.save(data_root+"test_michelle_cell_airfrance_obama.npy", test_set)
    
    #print "before saving..."
    #print train_set[12]
    #print test_set[12]



    #-----------------#added by Yamin:
    #Add Boston bombing tweets in the training data
    data_root2='/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Data/rumor_labeled/'
    ##    print(os.path.exists(data_root))
    ##    print(os.path.exists(data_root+"boston_baseline_raw.txt"))
    ##    print(os.getcwd())

    boston_raw=load_tweets_label_boston(data_root2+"boston_baseline_raw")
    print ("BOSTON DATASET : ", boston_raw.shape)

    print("train_set: ",train_set.shape)

    boston_featMat=extract_features(boston_raw)
    print("boston_featMat: ",boston_featMat.shape)
    train_set = np.vstack((train_set,boston_featMat))
    print("combined: ",train_set.shape)

    n_rumor=np.count_nonzero(train_set[:,6])
    n_samples=train_set.shape[0]
    print "#non_rumor", n_samples-n_rumor
    print "#rumor", n_rumor
    print "ratio of rumor", n_rumor*1.0/n_samples,'\n'

    #-----------------#added by Yamin:
    #Add nonrumors fetched from twitter into the training data
    data_root1='/Users/yamintun/Google Drive/NLP_Project/FINAL_NLP/Data/Nonrumors/'
    ##    print(os.path.exists(data_root))
    ##    print(os.path.exists(data_root+"boston_baseline_raw.txt"))
    ##    print(os.getcwd())

    nonrumor_raw=load_tweets_label(data_root1+"nonRumor_1200.txt")
    print ("NONRUMORS DATASET : ", nonrumor_raw.shape)

    nonrumor_featMat=extract_features(nonrumor_raw)
    print("nonrumor_featMat: ",nonrumor_featMat.shape)
    train_set = np.vstack((train_set,nonrumor_featMat))
    print("combined: ",train_set.shape)

    n_samples=train_set.shape[0]
    print "#non_rumor", n_samples-n_rumor
    print "#rumor", n_rumor
    print "ratio of rumor", n_rumor*1.0/n_samples,'\n'

    #-----------------
    #trim out some rumor tweets to be 50-50 for train data
    train_set=trim_dataset_label(train_set)

    np.save(data_root+"train_palin_boston_nonrumors.npy",train_set)


    ## STEP 3 - SAVE THE TRAINING AND TEST SET FILES ##
    ##    np.save("train_palin_boston_nonrumors_morequestP.npy", train_set)
    ##    n_rumor=np.sum(train_set[:,6])
    ##    n_samples=train_set.shape[0]
    ##    print "\nafter shuffling: \n#non_rumor", n_samples-n_rumor
    ##    print "#rumor", n_rumor
    ##    print "ratio of rumor", n_rumor*1.0/n_samples
    ##    #np.save(data_root+"test_with_boston.npy", test_set)

    n_rumor=np.sum(train_set[:,6])
    n_samples=train_set.shape[0]
    print "\nafter shuffling: \n#non_rumor", n_samples-n_rumor
    print "#rumor", n_rumor
    print "ratio of rumor", n_rumor*1.0/n_samples
    #np.save(data_root+"test_with_boston.npy", test_set)    


    #f1 = np.load(data_root+"train.npy")
    #f2 = np.load(data_root+"test.npy")
    #print "after saving..."
    #print f1[12]
    #print f2[12]

    
