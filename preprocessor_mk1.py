import requests
import re
import pymongo
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

class TextPreprocessor(object):

    def __init__(self, stop_words = 'en', tfidf = True):
        self.stop_words = get_stop_words(stop_words)
        self.tfidf = tfidf

    def _launch_mongo(self, db_name, coll_name, uri):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

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

    def _lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmed = []
        for word in tokens:
            lem_word = lemmatizer.lemmatize(word)
            lemmed.append(word)
        return lemmed

    def _vectorize(self, tokens):
        if self.tfidf:
            vectorizer = TfidfVectorizer()
            vectorized = self.vectorizer.fit_transform(tokens)
        else:
            vectorizer = CountVectorizer()
            vectorized = self.vectorizer.fit_transform(tokens)

        return vectorizer, vectorized

    # ----------- private methods above this line -----------

    def new_article(self, url):
        req = self._gen_request(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ''
        for p in paragraphs:
            article = article + p.get_text()

        clean = self._correct_sentences(article)
        return clean

    def generate_vectors(self, article, lemmatize = False):
        encoded_tokens = self._tokenize_encode_ascii(article)
        stopped_tokens = self._remove_stop_words(encoded_tokens)
        if lemmatize:
            lemmed_tokens = self._lemmatize(stopped_tokens)
            vectorizer, vectorized_tokens = self._vectorize(lemmed_tokens)
        else:
            vectorizer, vectorized_tokens = self._vectorize(stopped_tokens)

        return vectorizer, vectorized_tokens

#THIS I THINK WILL BE COMPLETED MUCH LATER NEEDS TO RUN ON ENTIRE CORPUS
    def db_pipeline(self, db_name, coll_name, uri = None):
        coll = self._launch_mongo(db_name, coll_name, uri)
        for article in coll.find_all('article'):
            cleaned = self._correct_sentences(article)
            vectorizer, vectorized_tokens = generate_vectors(cleaned)
            #TODO: Use this to clean and vectorized all the data definitely connect to mongo for this
            #NOTE: could this connect to mongo and then just pass back the vectorizer and vectorized tokens
            #      this could then be called douwn by mk1 sentiment analyzer and used for clustering


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    article_text = preprocessor.new_article('https://www.washingtonpost.com/local/virginia-politics/reeks-of-subtle-racism-tensions-after-black-candidate-left-off-fliers-in-virginia/2017/10/18/de74c47a-b425-11e7-a908-a3470754bbb9_story.html?utm_term=.2e8be491c0a3')
    vectorizer, vectorized_tokens = preprocessor.generate_vectors(article_text, lemmatize = True)
    nolemvectorizer, nolem_tokens = preprocessor.generate_vectors(article_text, lemmatize = False)
