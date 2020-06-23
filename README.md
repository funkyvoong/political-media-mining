# SocialMediaMining
This project presents a supervised
machine learning approach to predict whether a tweet is
politically biased or neutral. The approach uses a labeled
data set available at Crowdflower, where each tweet is
tagged with a partisan/neutral label plus its message type
and audience. The approach considers a combination of
linguistic features including Term Frequency-Inverse Document Frequency (TF-IDF), bigrams, and trigrams along
with metadata features including mentions, retweets, and
URLs, as well as these additional labels of message type
and audience. It trains both simple and ensemble classifiers
and assesses their the performance using precision, recall
and F-1 score. The results demonstrate that the classifiers
can predict the polarity of a tweet highly accurately when
trained on a combination of TF-IDF and metadata features
alone, which can be extracted automatically from the
tweets, obviating the need for additional tagging.
