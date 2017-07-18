"""
Plot wordcloud out of tweets
Author: Yamin Tun
Reference: http://sebastianraschka.com/Articles/2014_twitter_wordcloud.html
"""

#%matplotlib inline
import pickle as pkl

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


with open("cluster_text2.p", 'rb') as f:
        dict_Word = pkl.load(f)

#Rumor Cluster ID list
#id_list=[117041519.0, 6042217.0, 1042211.0, 167041521.0, 5041708.0, 56041520.0, 13041519.0, 45041519.0, 76041522.0, 1042203.0, 99041520.0, 89041520.0, 1042217.0, 2041523.0, 1042200.0, 6041608.0, 1041601.0, 108041522.0, 5042215.0, 91041520.0, 10041518.0, 16041518.0, 49041519.0, 4042123.0, 3041614.0, 30041519.0, 75041520.0, 48041523.0, 2041518.0, 2042215.0, 6041612.0, 4041522.0, 23041520.0]

#Non-rumor Cluster ID list
#id_list =[3041602,  1041710,  2041707,  1041603,   5042203,  10042214, 14041701,  304170,   1041701,   6041610]

id_list=[3041611]
#100041521- first 

for myid in id_list:
	words=dict_Word[myid]
	print "myid: ",myid
	
        # join tweets to a single string
        #words = cluster_text[words_activation_list.iloc[i][100041521]]# "good good good bad artistic artistic games games games games"
        #' '.join(tm.df['tweet'])

        # remove URLs, RTs, and twitter handles
        # no_urls_no_tags = " ".join([word for word in words.split()
        #                             if 'http' not in word
        #                                 and not word.startswith('@')
        #                                 and word != 'RT'
        #                             ])

	no_urls_no_tags = " ".join([word for word in words.split() ])

	wordcloud = WordCloud(
	                      font_path = os.environ.get("FONT_PATH", "/Library/Fonts/Times New Roman.ttf"),
	                      ##font_path='/Users/ytun/Library/Fonts/CabinSketch-Bold.ttf',CabinSketch-Bold.ttf"),  #
	                      stopwords=STOPWORDS,
	                      background_color='black',
	                      width=1800,
	                      height=1400
	                     ).generate(no_urls_no_tags)

	plt.imshow(wordcloud)
	plt.axis('off')
	plt.savefig('./my_twitter_wordcloud_1.png', dpi=300)
	plt.show()

