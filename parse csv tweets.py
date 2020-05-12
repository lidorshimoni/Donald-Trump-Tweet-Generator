import pandas as pd
from utils import save_text
import os, sys, string

DATA_DIR = 'training_data'

data = pd.read_csv(DATA_DIR+'/'+'realdonaldtrump.csv')

data = data["content"]

# data = [str(str(x).encode('utf-8'))[2:-1] for x in data]

print("- total number of tweets: ", len(data))

filtered = []

for tweet in data:
    check_tweet = str(str(str(tweet).lower().split()).encode('utf-8'))[2:-1]
    if '#' not in tweet and 'http' not in check_tweet:
        filtered.append(check_tweet)

print("- number of filtered tweets: ", len(filtered))
# print(comment)
input()
tweets = []

for tweet in filtered:
    tokens = tweet.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tweets.append(' '.join(tokens))

save_text(tweets, DATA_DIR + '/tweets.txt')

print("tweets saved to tweets.txt")
