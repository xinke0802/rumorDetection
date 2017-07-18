"""
Collect tweets
Author: Anushree Ghosh
"""

import oauth2 as oauth
import urllib2 as urllib
import sys
import json

# See assignment1.html instructions or README for how to get these credentials

api_key = "gHRSRSn6oN2Vfw5OzyKNUI5fK"
api_secret = "GEJIk7O1mGVq6wRTOHDSXKFoK5i75LibPph51N73WxHqQ6wOoH"
access_token_key = "155203347-KHhbqxgBHVwXUtkHssrkPKq4JN99c77KgS3A07qS"
access_token_secret = "bk9aq3q2Uak0Wf0SJJ7t1a6ee3NoiNeTL8Esaib3k7lDn"

_debug = 0

oauth_token    = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
def twitterreq(url, method, parameters):
  req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

  req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

  headers = req.to_header()

  if http_method == "POST":
    encoded_post_data = req.to_postdata()
  else:
    encoded_post_data = None
    url = req.to_url()

  opener = urllib.OpenerDirector()
  opener.add_handler(http_handler)
  opener.add_handler(https_handler)

  response = opener.open(url, encoded_post_data)

  return response

def fetchsamples(name):

  # url = "https://stream.twitter.com/1/statuses/sample.json"
  url="https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=%s&count=440"%name

  #?screen_name=anushree99&count=2"

  # parameters = [{"screen_name": name},{"count":1}]
  parameters=[]
  response = json.load(twitterreq(url, "GET", parameters))

  texts = [(tweet["text"]).encode("utf-8") for tweet in response]

  # for line in response:
    # print line.strip()

  print texts 

  # statuses = response.json()
  # print "\n".join([status["text"] for status in statuses])


  
if __name__ == '__main__':
  # print sys.argv[1]
  fetchsamples(sys.argv[1])
