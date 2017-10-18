import requests
import spacy
import re
from stop_words import get_stop_words
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, BaseSentimentAnalyzer


class TextPreprocessor(object):

    def __init__(self, stop_words = 'en', tfidf = True):
        self.stop_words = get_stop_words(stop_words)
        self.vectorizer = None
        self.tfidf = tfidf

    def _gen_request(self, url):
        req = requests.get(str(url.strip()))
        if req.status_code != 200:
            print 'WARNING', req.status_code
            self.error_writer.writerow(('nyt', url))
        else:
            return req

    def _correct_sentences(self, text):
        add_spaces = re.sub(r'(?<=[.!?])(?=[^\s])', r' \n', text)
        return add_spaces

    def _tokenize_encode_ascii(self, article):
        encoded = [] #this tokenizes the data
        for word in article.split():
            encoded.append(word.encode('ascii', 'ignore').lower())
        return encoded

    def _remove_stop_words(self, tokens):
        stopped_tokens = [tok for tok in tokens if tok not in self.stop_words]
        return stopped_tokens

    def _vectorize(self, tokens):
        if self.tfidf:
            self.vectorizer = TfidfVectorizer()
            vectorized = self.vectorizer.fit_transform(tokens)
        else:
            self.vectorizer = CountVectorizer()
            vectorized = self.vectorizer.fit_transform(tokens)

        return vectorized

    def _return_top_words(self, model, feature_names, n_top_words = 50):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_top_n_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_dict[topic_idx] = topic_top_n_words

        return topic_dict

    # ----------- non-private methods below this line -----------

    def new_article(self, url):
        req = self._gen_request(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ''
        for p in paragraphs:
            article = article + p.get_text()

        clean = self._correct_sentences(article)
        return clean

    def lda_dim_reduction(self, article):
        encoded_tokens = self._tokenize_encode_ascii(article)
        stopped_tokens = self._remove_stop_words(encoded_tokens)
        vectorized = self._vectorize(stopped_tokens)
        lda = LatentDirichletAllocation(n_components = 3, learning_method = 'batch').fit(vectorized)
        feature_names = self.vectorizer.get_feature_names()
        topic_dict = self._return_top_words(lda, feature_names)

        return topic_dict


    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)


#THIS I THINK WILL BE COMPLETED MUCH LATER
    def db_pipeline(self, list_articles):
        for article in list_articles:
            cleaned = self._correct_sentences(article)

            #TODO: train NaiveBayesClassifier to predict on downloaded dataset
            #TODO: identify orthogonal words/n-grams for each noun/noun_phrase predict sentiment of those
            #TODO: predict sentiment of sentence/paragraph


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    article_text = preprocessor.new_article('https://www.washingtonpost.com/local/virginia-politics/reeks-of-subtle-racism-tensions-after-black-candidate-left-off-fliers-in-virginia/2017/10/18/de74c47a-b425-11e7-a908-a3470754bbb9_story.html?utm_term=.2e8be491c0a3')
    article_topics = preprocessor.lda_dim_reduction(article_text)
