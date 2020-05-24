#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import re
from collections import defaultdict

warnings.filterwarnings("ignore")

# Read data from the input file to a pandas dataframe
data = pd.read_csv("political_media.csv", delimiter=',', encoding = "ISO-8859-1")

## Preliminary Analysis of the data
#Identified the tweets in other languages. Observed those tweets have the letter Ì
#data[data.text.str.contains('Ì')].index

data = data.drop(index=[55, 395, 455, 1294, 1694, 2991, 2998, 3001, 3360, 4238, 4240, 4241, 4243, 4244, 4245, 4332, 4335, 4339, 4344, 4358, 4359, 4812, 4890])

# Add possible new features

# Compute the length of the tweet
data['length'] = data['text'].apply(lambda x: len(x))

# Create Retweet and Mention columns (with 0/1 values)
data['Retweet'] = data.text.apply(lambda x: 1 if x.startswith('RT') else 0)
data['Mention'] = data.text.apply(lambda x: 1 if x.__contains__('@') else 0)

# Remove lengthy tweets exceeding the twitter character limit
data = data[data.length <= 280]
print(data.shape)

# Plot the bar graph of tweets based on the bias 
data.groupby('bias', as_index=False).count().plot(kind='bar', x='bias', y='text')
plt.text(-0.1, data[data.bias=='neutral'].shape[0]+30, data[data.bias=='neutral'].shape[0])
plt.text(0.9, data[data.bias=='partisan'].shape[0]+30, data[data.bias=='partisan'].shape[0])
plt.ylabel('No. of tweets')
plt.show()

# Plot a bar chart of tweets based on message type
data.groupby('message', as_index=False).count().sort_values(by='text', ascending = False).plot(kind = 'bar', x='message', y='text', sort_columns=True)

# Show the distribution of tweets based on message types and bias
temp = data.groupby(['message', 'bias'], as_index=False).count()[['message', 'bias', 'text']]
print(temp.pivot(index='message', values='text', columns='bias'))


# Show the distribution of tweets based on audience type and bias
temp1 = data.groupby(['audience', 'bias'], as_index=False).count()[['audience', 'bias', 'text']]
temp1.pivot(index='audience', values='text', columns='bias')

# Plot the distribution of tweets based on message type and bias
ax = data.groupby('message', as_index=False).count().sort_values(by='text', ascending=False).plot(x='message', y='text', kind='bar', figsize=[10,5])
data[data.bias == 'partisan'].groupby('message', as_index=False).count().sort_values(by='text', ascending=False).plot(x='message', y='text', kind='bar', figsize=[10,5], color='orange', ax = ax)
plt.ylabel('No. of tweets', fontsize=10)
plt.xlabel('Message', fontsize=10)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
ax.legend(['neutral', 'partisan'])


# Plot the distribution of tweets based on audience type and bias
ax = data.groupby('audience', as_index=False).count().sort_values(by='text', ascending=False).plot(x='audience', y='text', kind='bar', figsize=[4,5])
data[data.bias == 'partisan'].groupby('audience', as_index=False).count().sort_values(by='text', ascending=False).plot(x='audience', y='text', kind='bar', figsize=[4,5], color='orange', ax = ax)
plt.ylabel('No. of tweets', fontsize=10)
plt.xlabel('Audience', fontsize=10)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
ax.legend(['neutral', 'partisan'])

# Plot the distribution of tweets based on message type and bias for the tweets that are flagged as retweets
print("Distribution of Retweets based on message type and bias")
retweets = pd.merge(data[data.Retweet == 0].groupby(['message'], 
                                                as_index=False).count()[['message', 'bias', 'text']], 
                data[(data.bias == 'partisan')&(data.Retweet == 0)].groupby([
                    'message', 'bias'], as_index=False).count()[['message', 'bias', 'audience']], 
                on=['message'], how='left').sort_values(by='text', ascending=False)

ax = retweets.plot(x='message', y='text', kind='bar', figsize=[10,4])
retweets.plot(x='message', y='audience', kind='bar', figsize=[10,4], color='orange', ax=ax)
plt.ylabel('No. of Retweets', fontsize=10)
plt.xlabel('Message', fontsize=10)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
ax.legend(['neutral', 'partisan'])

# Show the distribution of retweets based on message types and bias
temp1 = data[data.Retweet == 1].groupby(['message', 'bias'], as_index=False).count()[['message', 'bias', 'text']]
temp1 = temp1.append({'message':'attack', 'bias':'neutral', 'text':0}, ignore_index=True)
temp1 = temp1.append({'message':'information', 'bias':'partisan', 'text':0}, ignore_index=True)
temp1 = temp1.append({'message':'other', 'bias':'partisan', 'text':0}, ignore_index=True)
print("Distribution of retweets based on message and bias type")
temp1.pivot(index='message', values='text', columns='bias')


# Plot the distribution of tweets based on message type and bias for the tweets that are flagged to have mentions
print("Distribution of tweets based on message type and bias for the tweets with mentions")
mentions = pd.merge(data[data.Mention == 0].groupby(['message'], 
                                                as_index=False).count()[['message', 'bias', 'text']], 
                data[(data.bias == 'partisan')&(data.Mention == 0)].groupby([
                    'message', 'bias'], as_index=False).count()[['message', 'bias', 'audience']], 
                on=['message'], how='left').sort_values(by='text', ascending=False)

ax = mentions.plot(x='message', y='text', kind='bar', figsize=[10,4])
mentions.plot(x='message', y='audience', kind='bar', figsize=[10,4], color='orange', ax=ax)
plt.ylabel('No. of Mentions', fontsize=10)
plt.xlabel('Message', fontsize=10)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
ax.legend(['neutral', 'partisan'])


# Show the distribution of tweets based on message types and bias
temp2 = data[data.Mention == 1].groupby(['message', 'bias'], as_index=False).count()[['message', 'bias', 'text']]
print("Distribution of tweets with mentions based on message and bias type")
temp2.pivot(index='message', values='text', columns='bias')

# Some Descriptive Stats
print("No. of Retweets =", len(data[data['text'].str.startswith('RT')]))
print("No. of Tweets with @'s =", len(data[data['text'].str.contains('@')]))
print("No. of Tweets with hashtags =", len(data[data['text'].str.contains('#')]))
print("No. of Tweets with URLs =", len(data[(data['text'].str.contains('http'))|(data['text'].str.contains('www'))]))
print("No. of Tweets with less confidence in classification =", len(data[data['bias:confidence'] < 1]))
print("Average Tweet Length = ", data.length.mean())

# Identify the hashtags used in partisan and neutral tweets and the count of their occurances
def top_hashtags(bias, data):
    bias_tweets = data[data.bias == bias]
    bias_hashtags = defaultdict(int)
    for i in range(len(bias_tweets)):
        htags = [x for x  in re.sub('[!?.,:;()]', '', bias_tweets.text.values[i].lower()).split(' ') if x.startswith('#')]
        for tag in htags:
            bias_hashtags[tag] += 1
    bias_hashtags_sorted = sorted(bias_hashtags.items(), key=lambda k_v: k_v[1], reverse=True)
    print("Total # of Hashtags in the tweets =", len(bias_hashtags))
    print("Top 10 ", bias, "Hashtags:")
    for num, h in enumerate(bias_hashtags_sorted[:10]):
        print(num," ",h)
    
    print(type(bias_hashtags_sorted[:10]))
    return bias_hashtags_sorted[:10]

partisan_hashtags = top_hashtags('partisan', data)
neutral_hashtags = top_hashtags('neutral', data)

print("Top 10 Hastags in partisan tweets")
pd.DataFrame(partisan_hashtags, columns=['Hashtag', 'Count'])

print("Top 10 Hastags in neutral tweets")
pd.DataFrame(neutral_hashtags, columns=['Hashtag', 'Count'])

# Define function to plot the distribution of top 10 hashtags in neutral and partisan tweets
def plot_bar_x(htags, bias):
    index = np.arange(len(htags))
    counts = [num[1] for num in htags]
    label = [h[0] for h in htags]
    plt.bar(index, counts)
    plt.xlabel(bias + ' Hashtags', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=45)
    #plt.title('Top 10', bias, 'Hashtags')
    plt.show()

print("Bar chart of hashtags in partisan and neutral tweets")
plot_bar_x(partisan_hashtags, 'Partisan')
plot_bar_x(neutral_hashtags, 'Neutral')

# Plot Distribution of tweets based on their length for neutral and partisan tweets
print("Plot showing the distribution of tweets based on their length for neutral and partisan tweets")
data.hist(column='length', by='bias', figsize=[15, 5])

# Plot Distribution of tweets based on the no. of hastags in mentioned in it for neutral and partisan tweets
print("Plot showing the distribution of tweets based on the no. of hastags in mentioned in it for neutral and partisan tweets")
data['hashtags'] = data.text.apply(lambda x:x.count('#'))
data.hist(column='hashtags', by='bias', figsize=[10,4])

# Plot Distribution of tweets based on the no. of mentions in it for neutral and partisan tweets
print("Plot showing the distribution of tweets based on the no. of mentions in it for neutral and partisan tweets")
data['mentions'] = data.text.apply(lambda x:x.count('@'))
data.hist(column='mentions', by='bias', figsize=[10,4])


## Pre Processing
# Load necessary packages for pre-processing
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from string import punctuation
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Defina a class to remove repeated characters in a word
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
        
    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

# Define a function to pre-process a tweet        
def pre_process_text(tweet):
    replacer = RepeatReplacer()
    tknzr = TweetTokenizer()
    ps = PorterStemmer()
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('http', ' ', tweet) # remove http, but keep the text in the url
    tweet = re.sub('www', ' ', tweet) # remove http, but keep the text in the url
    tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
    tweet = re.sub('[^\s]+.com', '', tweet) # remove email addresses
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = re.sub('&amp;', '', tweet) #removing & (ampersand)
    tweet = re.sub("[!?.,:;'-/]", ' ', tweet) #split based punctuations
    # Removing special characters converted to special chars(as below) when 
    # loaded using ISO-8859-1 encoding)
    tweet = re.sub('\x89', ' ', tweet) 
    tweet = re.sub('ûï', ' ', tweet)
    tweet = re.sub('ûª',' ',tweet)
    tweet = re.sub('û_',' ',tweet)
    tweet = re.sub('ûó',' ',tweet)
    tweet = re.sub('[0-9]', ' ', tweet) #removing numbers
    tweet = replacer.replace(tweet) # remove repeated characters (helloooooooo into hello)
    tweet = tknzr.tokenize(tweet) #tokenize words in the tweet
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    tweet = [ps.stem(word) for word in tweet if len(word) >= 3] #Stemming 
    return ' '.join(tweet).strip()

data['processed'] = data.text.apply(pre_process_text)

# Delete all the tweets which are empty (strings) after pre-processing
data['length'] = data.processed.apply(lambda x: len(x))
data = data[data.length >= 1]

## Build word cloud for each label 
# Import the packages to view word clouds
from wordcloud import WordCloud, STOPWORDS
stop_words = set(STOPWORDS)

# Define function to show word clouds 
def show_wordcloud(col, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=500,
        max_font_size=40, 
        scale=3,
        random_state=1,
        #colormap='magma'
    ).generate(str(col))

    fig = plt.figure(1, figsize=(14, 14))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

# Word cloud for partisan tweets
show_wordcloud(" ".join(data[data.bias == 'partisan']['processed']))

# word cloud for neutral tweets
show_wordcloud(" ".join(data[data.bias == 'neutral']['processed']))


# Latent Dirichlet Allocation
# Import necesasry packages for LDA
from gensim import corpora, models
texts = [text.split(' ') for text in data.processed.values]
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.6)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaMulticore(corpus, num_topics=2, id2word=dictionary, chunksize=10000, passes=50, 
                          workers=4)

# Identify words in the two topics
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)
np.array([np.array([x[0], x[1]]) for x in topics_matrix[:,1][:][0]])[:,0], np.array([np.array([x[0], x[1]]) for x in topics_matrix[:,1][:][1]])[:,0]


# The words obtained from two topics using LDA are similar to the words in the word clouds of partisan and neutral tweets
# Identify unique words with respect to partisan and neutral tweets and remove the words common to both categories
neutral_words = set(" ".join(data[data.bias == 'neutral']['processed']).split(' '))
partisan_words = set(" ".join(data[data.bias == 'partisan']['processed']).split(' '))
only_neutral_words = neutral_words - partisan_words
only_partisan_words = partisan_words - neutral_words
only_neutral_text = " ".join([word for word in " ".join(data[data.bias == 'neutral']['processed']).split(' ') if word in only_neutral_words])
only_partisan_text = " ".join([word for word in " ".join(data[data.bias == 'partisan']['processed']).split(' ') if word in only_partisan_words])

# Identify the unique words in tweets and the count of their occurances
most = pd.Series(' '.join(data.processed).split()).value_counts()
print("Total no.of unique words = ", len(most))
print("Top 10 words used in the tweets:")
most[:10]

# Remove the top 30 common words from the processed texts and also the ones that appeared only once in the entire corpus
common_words = neutral_words.intersection(partisan_words)
common_words = common_words.intersection(list(most[:30].keys()))
data.processed = [' '.join([word for word in text.split(' ') if word not in common_words]) for text in data.processed.values]
data.processed = [' '.join([word for word in text.split(' ') if word in most and most[word]>1]) for text in data.processed.values]
data.head(2)

# Show word clouds of neutral tweets after removing common words
show_wordcloud(only_neutral_text)

# Show word clouds of partisan tweets after removing common words
show_wordcloud(only_partisan_text)


## Compute/Plot Descriptive Statistics
# Identify the unique words in tweets and the count of their occurances after removing the most common and uncommon words
most = pd.Series(' '.join(data.processed).split()).value_counts()
print("Total no.of unique words = ", len(most))
print("Top 10 words used in the tweets:")
most[:10]


# # Compute Sentiment Score for tweets
# Download vader lexicon from nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Compute the sentiment scores for the original tweets using vader sentiment analyzer
sentiment_scores = pd.DataFrame()
sid = SentimentIntensityAnalyzer()
for tweet in data['text']:
    sentiment_scores = sentiment_scores.append([[tweet]+list(sid.polarity_scores(tweet).values())], ignore_index=True)
sentiment_scores.columns = ['text', 'Negative', 'Neutral', 'Positive', 'Compound']
sentiment_scores = pd.merge(sentiment_scores, data[['text', 'audience', 'message', 'bias', 'bias:confidence']], on=['text'], how='inner')
sentiment_scores.to_csv('sentiment_scores.csv')
sentiment_scores.head()

# Plot the distribution of positive, negative, neutral and compound scores for neutral and partisan tweets
print("Plot showing the distribution of positive, negative, neutral and compound scores for neutral and partisan tweets ")
plt.figure(num = None, figsize = (20, 8), dpi = 80, facecolor = 'w', edgecolor = 'k')
k = 0
for i, label in enumerate(['neutral', 'partisan']):
    for j, sentiment in enumerate(['Negative', 'Neutral', 'Positive', 'Compound']):
        plt.subplot(2, 4, k+1)
        sentiment_scores[sentiment_scores.bias == label][sentiment].hist(bins=4)
        k += 1
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(label+'/'+sentiment)
    
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.5)
plt.show()
# The sentiment scores for partisan and neutral tweets are not clearly differentiable in this case

# Statistics of sentiment scores for neutral tweets
print("Statistics of sentiment scores for neutral tweets")
sentiment_scores[sentiment_scores.bias == 'neutral'][['Negative', 'Neutral', 'Positive', 'Compound']].describe()

# Statistics of sentiment scores for partisan tweets
print("Statistics of sentiment scores for partisan tweets")
sentiment_scores[sentiment_scores.bias == 'partisan'][['Negative', 'Neutral', 'Positive', 'Compound']].describe()

## Preprocess to Compute Language Features
# Feature Extraction from Text
# Import necessary packages to compute language features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set target variable to 0/1 based on the bias
data['target'] = data.bias.apply(lambda x: 1 if x == 'partisan' else 0)
# Split the tweet into list of words
texts = [text.split(' ') for text in data.processed.values]
    
# Compute TF-IDF features for the tweets 
# (With minimum document frequency of 5 and max. document frequency of 0.7)
# Keep the top 400 features
tfidfconverter = TfidfVectorizer(max_features=400, min_df=5, max_df=0.7)  
tfidf_data = tfidfconverter.fit_transform(data.processed.values).toarray()

# Compute the bigrams for the tweets
vectorizer = CountVectorizer(ngram_range=(2,2))
vectorizer.fit(data.processed) # build ngram dictionary
bigram = vectorizer.transform(data.processed).toarray() # get ngram
#print('vectorizer.vocabulary_: {0}'.format(vectorizer.vocabulary_))

# Compute the trigrams for the tweets
three_vectorizer = CountVectorizer(ngram_range=(3,3))
three_vectorizer.fit(data.processed) # build ngram dictionary
trigram = three_vectorizer.transform(data.processed).toarray() # get ngram

print("TF-IDF Data Size:", tfidf_data.shape)
print("Bigrams as Features - Data Size:", bigram.shape)
print("Trigrams as Features - Data Size:", trigram.shape)

# One Hot encoding of the categorical variables
# Import necessary packages for on-hot encoding for other categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Convert message type to one-hot encoding
label_encoder = LabelEncoder()
message_integer_encoded = label_encoder.fit_transform(data.message.values)
onehot_encoder = OneHotEncoder(sparse=False)
message_integer_encoded = message_integer_encoded.reshape(len(message_integer_encoded), 1)
message_onehot_encoded = onehot_encoder.fit_transform(message_integer_encoded)

# Add the one-hot encoded data to the data frame
for i, label in enumerate(label_encoder.classes_):
    data['message_'+label] = message_onehot_encoded[:,i]

# Convert audience type to one-hot encoding
label_encoder = LabelEncoder()
audience_integer_encoded = label_encoder.fit_transform(data.audience.values)
onehot_encoder = OneHotEncoder(sparse=False)
audience_integer_encoded = audience_integer_encoded.reshape(len(audience_integer_encoded), 1)
audience_onehot_encoded = onehot_encoder.fit_transform(audience_integer_encoded)

# Add the one-hot encoded data to the dataframe
for i, label in enumerate(label_encoder.classes_):
    data['audience_'+label] = message_onehot_encoded[:,i]

# Combine all features together
# Combine the computed language features with the metadata
tfidf_features = np.hstack((tfidf_data, data[['Retweet', 'Mention']].values, message_onehot_encoded, audience_onehot_encoded))
bigram_features = np.hstack((bigram, data[['Retweet', 'Mention']].values, message_onehot_encoded, audience_onehot_encoded))
trigram_features = np.hstack((trigram, data[['Retweet', 'Mention']].values, message_onehot_encoded, audience_onehot_encoded))

print("TF-IDF Data Size:", tfidf_features.shape)
print("Bigrams as Features - Data Size:", bigram_features.shape)
print("Trigrams as Features - Data Size:", trigram_features.shape)

# TF-IDF Features split
# Import package to split data for training and testing
from sklearn.model_selection import train_test_split

# Identify the tweets with low confidence level for bias classification
confident_indices = data['bias:confidence'].apply(lambda x: True if x == 1 else False).values
non_confident_indices = data['bias:confidence'].apply(lambda x: False if x == 1 else True).values

# Use the low-confident tweets for testing and the divide the rest into training and testing with 3:1 ratio
tfidf_train = tfidf_features[confident_indices]
target_train = data.target.values[confident_indices]
tfidf_test = tfidf_features[non_confident_indices]
target_test = data.target.values[non_confident_indices]
tfidf_test.shape

tf_train, tf_test, ytf_train, ytf_test = train_test_split(tfidf_train, target_train, test_size=0.25, 
                                                          random_state = 10)

tf_test = np.append(tf_test,  tfidf_test, axis=0)
ytf_test = np.append(ytf_test, target_test, axis = 0)

tf_train.shape, ytf_train.shape, tf_test.shape, ytf_test.shape


# Bigram test - train split
# Split bigrams into training and testing in a similar way as above
bigram_train = bigram_features[confident_indices]
bigram_test = bigram_features[non_confident_indices]

bi_train, bi_test, ybi_train, ybi_test = train_test_split(bigram_train, target_train, test_size=0.25, 
                                                          random_state = 10)

bi_test = np.append(bi_test,  bigram_test, axis=0)
ybi_test = np.append(ybi_test, target_test, axis = 0)

bi_train.shape, ybi_train.shape, bi_test.shape, ybi_test.shape

# Trigram test - train split
# Split trigrams into training and testing in a similar way as above
trigram_train = trigram_features[confident_indices]
trigram_test = trigram_features[non_confident_indices]

tri_train, tri_test, ytri_train, ytri_test = train_test_split(trigram_train, target_train, 
                                                              test_size=0.25, 
                                                              random_state = 10)

tri_test = np.append(tri_test,  trigram_test, axis=0)
ytri_test = np.append(ytri_test, target_test, axis = 0)

tri_train.shape, ytri_train.shape, tri_test.shape, ytri_test.shape

# Split the data into train and test in the same was as above
test = data[data['bias:confidence'] < 1]
train = data[data['bias:confidence'] == 1]
#train['target'] = train.bias.apply(lambda x: 1 if x == 'partisan' else 0)
train_texts = [text.split(' ') for text in train.processed.values]
test_texts = [text.split(' ') for text in test.processed.values]
train_data = train[['processed', 'message_attack', 'message_constituency', 'message_information', 
                    'message_media', 'message_mobilization', 'message_other', 'message_personal', 
                    'message_policy', 'message_support', 'audience_national', 
                    'audience_constituency', 'Retweet', 'Mention']].values
test_data = test[['processed', 'message_attack', 'message_constituency', 'message_information', 
                  'message_media', 'message_mobilization', 'message_other', 'message_personal', 
                  'message_policy', 'message_support', 'audience_national', 'audience_constituency', 
                  'Retweet', 'Mention']].values

x_train, x_test, y_train, y_test = train_test_split(train_data, train.target.values, test_size=0.25,
                                                    random_state = 10)

x_test = np.append(x_test, test_data, axis=0) 
y_test = np.append(y_test, test.target.values)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Word2Vec
# Import necesasry packages to compute word2vec and doc2vec representations
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from gensim.models import doc2vec
from sklearn import utils
from gensim.models import Doc2Vec

# Build vocabulary from the training data
x_train_text = [x.split(' ') for x in x_train[:, 0]]
x_test_text = [x.split(' ') for x in x_test[:, 0]]
tweet_w2v = Word2Vec(size=400, min_count=10)
tweet_w2v.build_vocab(x_train_text)
tweet_w2v.train(x_train_text, total_examples=tweet_w2v.corpus_count, epochs=20)  

# Compute TF-IDF for the training text
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train_text)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

# Build vectors based on the vocab and the tf-idf of the words
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

# Word2Vec vectors of length 400 for train data
train_vecs_w2v = np.concatenate([buildWordVector(z, 400) for z in x_train_text])
train_vecs_w2v = scale(train_vecs_w2v)

# Word2Vec vectors of length 400 for test data
test_vecs_w2v = np.concatenate([buildWordVector(z, 400) for z in x_test_text])
test_vecs_w2v = scale(test_vecs_w2v)
print("Shape of Word2Vec vectors:")
train_vecs_w2v.shape, test_vecs_w2v.shape

# Append meta data to the word2vec vectors
train_vecs_w2v_meta = np.append(train_vecs_w2v, x_train[:,1:], axis=1)
test_vecs_w2v_meta = np.append(test_vecs_w2v, x_test[:,1:], axis=1)
print("Shape of Word2Vec vectors with metadata:")
train_vecs_w2v_meta.shape, test_vecs_w2v_meta.shape

# Doc2Vec
# Label the sentences for Doc2Vec implentation
def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the complaint narrative.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v, [label]))
    return labeled

# Generate vectors from the corpus
def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors

# Label the train and test sentences
X_train = label_sentences(x_train_text, 'TRAIN')
X_test = label_sentences(x_test_text, 'TEST')
all_data = X_train + X_test

# Train and Distributed bag of words model to generate Doc2Vec vectors
model_dbow = Doc2Vec(dm=0, vector_size=400, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)

# Build vocabulary using all data
model_dbow.build_vocab([x for x in all_data])
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in all_data]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    
# Generate Vectors for training and test data 
train_vectors_dbow = get_vectors(model_dbow, len(X_train), 400, 'TRAIN')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 400, 'TEST')
print("Shape of Doc2Vec vectors:")
train_vectors_dbow.shape, test_vectors_dbow.shape

# Append metadata to the doc2vec vectors
train_vectors_dbow_meta = np.append(train_vectors_dbow, x_train[:,1:], axis=1)
test_vectors_dbow_meta = np.append(test_vectors_dbow, x_test[:,1:], axis=1)
print("Shape of Doc2Vec vectors with metadata:")
train_vectors_dbow_meta.shape, test_vectors_dbow_meta.shape

# Combine all training and testing data for clustering
doc2vec_data = np.append(train_vectors_dbow, test_vectors_dbow, axis=0)
vecs_w2v = np.concatenate([buildWordVector(z, 400) for z in texts])
vecs_w2v = scale(vecs_w2v)

# Save the Data into files
import numpy as np

np.save('data/tf_train.npy', tf_train)
np.save('data/tf_test.npy', tf_test)
np.save('data/ytf_train.npy', ytf_train)
np.save('data/ytf_test.npy', ytf_test)

np.save('data/tri_train.npy', tri_train)
np.save('data/tri_test.npy', tri_test)
np.save('data/ytri_train.npy', ytri_train)
np.save('data/ytri_test.npy', ytri_test)

np.save('data/bi_train.npy', bi_train)
np.save('data/bi_test.npy', bi_test)
np.save('data/ybi_train.npy', ybi_train)
np.save('data/ybi_test.npy', ybi_test)

np.save('data/y_train.npy', ybi_train)
np.save('data/y_test.npy', ybi_test)

np.save('data/train_vectors_dbow_meta.npy', train_vectors_dbow_meta)
np.save('data/test_vectors_dbow_meta.npy', test_vectors_dbow_meta)
np.save('data/train_vectors_dbow.npy', train_vectors_dbow)
np.save('data/test_vectors_dbow.npy', test_vectors_dbow)

np.save('data/train_vecs_w2v_meta.npy', train_vecs_w2v_meta)
np.save('data/test_vecs_w2v_meta.npy', test_vecs_w2v_meta)
np.save('data/train_vecs_w2v.npy', train_vecs_w2v)
np.save('data/test_vecs_w2v.npy', test_vecs_w2v)


## Clustering
# Import necessary modules for clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import pca
import matplotlib.pyplot as plt
import matplotlib

# Clustering of Word2Vec data
# PCA to reduce data to two features for visualization of clusters
r = pca.PCA(n_components=2).fit_transform(vecs_w2v)

# Computer clusters based on original data using K-Means
cluster_w2v = KMeans(n_clusters=2, init='k-means++', max_iter=100, )
cluster_w2v.fit(vecs_w2v)
print(sum(cluster_w2v.labels_ == data.target)/len(data.target))

colors = ['green', 'blue']

# Plot the original and computed clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
ax1.scatter(r[:,0], r[:,1], c=cluster_w2v.labels_, cmap=matplotlib.colors.ListedColormap(colors))

ax2.scatter(r[:,0], r[:,1], c=data.target, cmap=matplotlib.colors.ListedColormap(colors))

# Clustering of Doc2Vec data
# Do PCA to reduce data to two features for visualization of clusters
r = pca.PCA(n_components=2).fit_transform(doc2vec_data)

# Computer clusters based on original data using K-Means
cluster_d2v = KMeans(n_clusters=2, init='k-means++', max_iter=100)
cluster_d2v.fit(doc2vec_data)
print(sum(cluster_d2v.labels_ == data.target)/len(data.target))

colors = ['green', 'blue']

# Plot the original and computed clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
ax1.scatter(r[:,0], r[:,1], c=cluster_d2v.labels_, cmap=matplotlib.colors.ListedColormap(colors))
ax2.scatter(r[:,0], r[:,1], c=data.target, cmap=matplotlib.colors.ListedColormap(colors))

# Clustering of TF-IDF data
# Do PCA to reduce data to two features for visualization of clusters
r = pca.PCA(n_components=2).fit_transform(tfidf_features[:,:-13])

# Computer clusters based on original data using K-Means
cluster_tf = KMeans(n_clusters=2, init='k-means++', max_iter=100)
cluster_tf.fit(tfidf_features[:,:-13])
print(sum(cluster_tf.labels_ == data.target)/len(data.target))

colors = ['green', 'blue']

# Plot the original and computed clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
ax1.scatter(r[:,0], r[:,1], c=cluster_tf.labels_, cmap=matplotlib.colors.ListedColormap(colors))
ax2.scatter(r[:,0], r[:,1], c=data.target, cmap=matplotlib.colors.ListedColormap(colors))

# Clustering of Bigrams data
# Do PCA to reduce data to two features for visualization of clusters
r = pca.PCA(n_components=2).fit_transform(bigram_features[:,:-13])

# Computer clusters based on original data using K-Means
cluster_bi = KMeans(n_clusters=2, init='k-means++', max_iter=100)
cluster_bi.fit(bigram_features[:,:-13])
print(sum(cluster_bi.labels_ == data.target)/len(data.target))

colors = ['green', 'blue']

# Plot the original and computed clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(r[:,0], r[:,1], c=cluster_bi.labels_, cmap=matplotlib.colors.ListedColormap(colors))
ax2.scatter(r[:,0], r[:,1], c=data.target, cmap=matplotlib.colors.ListedColormap(colors))

# Clustering of Trigrams data
# PCA to reduce data to two features for visualization of clusters
r = pca.PCA(n_components=2).fit_transform(trigram_features[:,:-13])

# Computer clusters based on original data using K-Means
cluster_tri = KMeans(n_clusters=2, init='k-means++', max_iter=100)
cluster_tri.fit(trigram_features[:,:-13])
print(sum(cluster_tri.labels_ == data.target)/len(data.target))

# Plot the original and computed clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(r[:,0], r[:,1], c=cluster_tri.labels_, cmap=matplotlib.colors.ListedColormap(colors))
ax2.scatter(r[:,0], r[:,1], c=data.target, cmap=matplotlib.colors.ListedColormap(colors))


## Classification
# Create a dataframe to store the results
Results = pd.DataFrame()

# Import necessary modules for classification
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

# Define function to train classifier and record the results
def trainModel(classifier, trainData, testData, trainTarget, testTarget, feature, metadata, Results):

    classifier.fit(trainData, trainTarget)
    pred = classifier.predict(testData)
    conf = confusion_matrix(testTarget, pred)
    tp = conf[0][0]
    fp = conf[1][0]
    tn = conf[1][1]
    fn = conf[0][1]
    precision = round(tp/(tp+fp), 2)
    recall = round(tp/(tp+fn), 2)
    acc = round(np.sum(pred == testTarget)*100/len(pred), 2)
    fscore = round(2*precision*recall/(precision+recall), 2)
    Results = Results.append([[str(type(classifier)).split('.')[-1][:-2], feature, metadata, acc, precision, recall, fscore, tp, fp, tn, fn]])
    
    return Results

# Define function to run train model on all the features(with/without metadata)
def runOnAllFeatures(Results, classifier):
    
    global tf_train, tf_test, ytf_train, ytf_Test, bi_train, bi_test, ybi_train, ybi_test
    global tri_train, tri_test, ytri_train, ytri_test, train_vecs_w2v_meta, test_vecs_w2v_meta
    global y_train, y_test, train_vecs_w2v, test_vecs_w2v, train_vectors_dbow, test_vectors_dbow
    global train_vectors_dbow_meta, test_vectors_dbow_meta
    
    Results = trainModel(classifier, tf_train, tf_test, ytf_train, ytf_test, 'TFIDF', 'Y', Results)
    Results = trainModel(classifier, tf_train[:,:-11], tf_test[:,:-11], ytf_train, ytf_test, 'TFIDF', 'R/M', Results)
    Results = trainModel(classifier, tf_train[:,:-11], tf_test[:,:-11], ytf_train, ytf_test, 'TFIDF', 'N', Results)
    Results = trainModel(classifier, bi_train, bi_test, ybi_train, ybi_test, 'Bigrams', 'Y', Results)
    Results = trainModel(classifier, bi_train[:,:-11], bi_test[:,:-11], ybi_train, ybi_test, 'Bigrams', 'R/M', Results)
    Results = trainModel(classifier, bi_train[:,:-11], bi_test[:,:-11], ybi_train, ybi_test, 'Bigrams', 'N', Results)
    Results = trainModel(classifier, tri_train, tri_test, ytri_train, ytri_test, 'Trigrams', 'Y', Results)
    Results = trainModel(classifier, tri_train[:,:-11], tri_test[:,:-11], ytri_train, ytri_test, 'Trigrams', 'R/M', Results)
    Results = trainModel(classifier, tri_train[:,:-11], tri_test[:,:-11], ytri_train, ytri_test, 'Trigrams', 'N', Results)
    Results = trainModel(classifier, train_vecs_w2v_meta, test_vecs_w2v_meta, y_train, y_test, 'Word2Vec', 'Y', Results)
    Results = trainModel(classifier, train_vecs_w2v_meta[:,:-11], test_vecs_w2v_meta[:,:-11], y_train, y_test, 'Word2Vec', 'R/M', Results)
    Results = trainModel(classifier, train_vecs_w2v[:,:-11], test_vecs_w2v[:,:-11], y_train, y_test, 'Word2Vec', 'N', Results)
    Results = trainModel(classifier, train_vectors_dbow_meta, test_vectors_dbow_meta, y_train, y_test, 'Doc2Vec', 'Y', Results)
    Results = trainModel(classifier, train_vectors_dbow_meta[:,:-11], test_vectors_dbow_meta[:,:-11], y_train, y_test, 'Doc2Vec', 'R/M', Results)
    Results = trainModel(classifier, train_vectors_dbow[:,:-11], test_vectors_dbow[:,:-11], y_train, y_test, 'Doc2Vec', 'N', Results)

    return Results

# Define function to train xgboost model
def xgbmodel(trainData, testData, trainTarget, testTarget, feature, metadata, Results):

    param = {'eta': 1, 'silent': 0, 'objective': 'binary:logistic', 'max_depth':5}
    param['nthread'] = 2
    param['eval_metric'] = 'auc'
    num_round = 50

    dtrain = xgb.DMatrix(trainData, label=trainTarget)
    dtest = xgb.DMatrix(testData, label= testTarget)
    
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(param, dtrain, num_round, evals = evallist, verbose_eval=False)
    pred = [0 if y < 0.5 else 1 for y in bst.predict(dtest)]

    conf = confusion_matrix(testTarget, pred)
    tp = conf[0][0]
    fp = conf[1][0]
    tn = conf[1][1]
    fn = conf[0][1]
    precision = round(tp/(tp+fp), 2)
    recall = round(tp/(tp+fn), 2)
    acc = round(np.sum(pred == ytf_test)*100/len(pred), 2)
    fscore = round(2*precision*recall/(precision+recall), 2)

    Results = Results.append([['XGBoost', feature, metadata, acc, precision, recall, fscore, tp, fp, tn, fn]])
    
    return Results 

# Run Logistic Regression Classifier
Results = runOnAllFeatures(Results, LogisticRegression(C = 0.6, fit_intercept=False))

# Run SVM Classifier
Results = runOnAllFeatures(Results, SVC(C=1000, kernel='sigmoid'))

# Run Naive Bayes Classifier
Results = runOnAllFeatures(Results, GaussianNB())

# Run XGBoost Classifier
Results = xgbmodel(tf_train, tf_test, ytf_train, ytf_test, 'TFIDF', 'Y', Results)
Results = xgbmodel(tf_train[:,:-11], tf_test[:,:-11], ytf_train, ytf_test, 'TFIDF', 'R/M', Results)
Results = xgbmodel(tf_train[:,:-13], tf_test[:,:-13], ytf_train, ytf_test, 'TFIDF', 'N', Results)
Results = xgbmodel(bi_train, bi_test, ybi_train, ybi_test, 'Bigrams', 'Y', Results)
Results = xgbmodel(bi_train[:,:-11], bi_test[:,:-11], ybi_train, ybi_test, 'Bigrams', 'R/M', Results)
Results = xgbmodel(bi_train[:,:-13], bi_test[:,:-13], ybi_train, ybi_test, 'Bigrams', 'N', Results)
Results = xgbmodel(tri_train, tri_test, ytri_train, ytri_test, 'Trigrams', 'Y', Results)
Results = xgbmodel(tri_train[:,:-11], tri_test[:,:-11], ytri_train, ytri_test, 'Trigrams', 'R/M', Results)
Results = xgbmodel(tri_train[:,:-13], tri_test[:,:-13], ytri_train, ytri_test, 'Trigrams', 'N', Results)
Results = xgbmodel(train_vecs_w2v_meta, test_vecs_w2v_meta, y_train, y_test, 'Word2Vec', 'Y', Results)
Results = xgbmodel(train_vecs_w2v_meta[:,:-11], test_vecs_w2v_meta[:,:-11], y_train, y_test, 'Word2Vec', 'R/M', Results)
Results = xgbmodel(train_vecs_w2v[:,:-11], test_vecs_w2v[:,:-11], y_train, y_test, 'Word2Vec', 'N', Results)
Results = xgbmodel(train_vectors_dbow_meta, test_vectors_dbow_meta, y_train, y_test, 'Doc2Vec', 'Y', Results)
Results = xgbmodel(train_vectors_dbow_meta[:,:-11], test_vectors_dbow_meta[:,:-11], y_train, y_test, 'Doc2Vec', 'R/M', Results)
Results = xgbmodel(train_vectors_dbow[:,:-11], test_vectors_dbow[:,:-11], y_train, y_test, 'Doc2Vec', 'N', Results)

# Run Gradient Boosting Classifier
classifier = GradientBoostingClassifier(learning_rate=0.01, n_estimators=400, subsample=0.3)
Results = runOnAllFeatures(Results, classifier)

Results.columns = ['Model', 'Features', 'MetaData', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'TP', 'FP', 'TN', 'FN']
Results.reset_index(drop=True)
Results.to_csv('Classification_results.csv')
