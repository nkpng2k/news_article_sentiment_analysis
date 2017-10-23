import requests
import re
import pymongo
import string
import pickle
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

class TextPreprocessor(object):

    def __init__(self, stop_words = 'en', tfidf = True, lemmatize = False):
        self.stop_words = get_stop_words(stop_words)
        self.tfidf = tfidf
        self.lemmatize = lemmatize
        self.vectorizer = None

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

    def _encode_ascii(self, tokens):
        encoded = []
        for word in tokens:
            encoded.append(word.encode('ascii', 'ignore'))
        return encoded

    def _remove_stop_words(self, article):
        tokens = []
        for word in article.split():
            word = word.lower()
            word = word.strip(string.punctuation)
            tokens.append(word)
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
        vectorized = self.vectorizer.transform(tokens)

        return vectorized

    def _gen_corpus(self, docs_tokens):
        for doc in docs_tokens:
            yield doc

    def _tokenize(self, article):
        stopped_tokens = self._remove_stop_words(article)
        encoded_tokens = self._encode_ascii(stopped_tokens)
        no_just_punc_tokens = [tok for tok in encoded_tokens if tok not in string.punctuation]

        return no_just_punc_tokens


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

    def generate_vectors(self, article):
        cleaned_tokens = self._tokenize(article)
        if self.lemmatize:
            lemmed_tokens = self._lemmatize(cleaned_tokens)
            vectorized_tokens = self._vectorize(lemmed_tokens)
        else:
            vectorized_tokens = self._vectorize(cleaned_tokens)

        return self.vectorizer, vectorized_tokens

    def db_pipeline(self, db_name, coll_name, uri = None):
        coll = self._launch_mongo(db_name, coll_name, uri)
        all_docs = []
        error_counter = 0
        success = 0
        for doc in coll.find(snapshot = True).batch_size(25):
            try:
                cleaned = self._correct_sentences(doc['article'])
                cleaned_tokens = self._tokenize(cleaned)
                if self.lemmatize:
                    lemmed_tokens = self._lemmatize(cleaned_tokens)
                    all_docs.append(lemmed_tokens)
                else:
                    all_docs.append(cleaned_tokens)
                success += 1
                print 'Success # {}'.format(success)
            except TypeError:
                error_counter += 1
                print 'TypeError, Moving On. Error #{}'.format(error_counter)

        corpus = self._gen_corpus(all_docs)

        if self.tfidf:
            self.vectorizer = TfidfVectorizer(preprocessor = lambda x: x,
                                              tokenizer = lambda x: x, min_df = 0.1).fit(corpus)
        else:
            self.vectorizer = CountVectorizer(preprocessor = lambda x: x,
                                              tokenizer = lambda x: x, min_df = 0.1).fit(corpus)

        print "success TFIDF Vectorizer has been trained"



if __name__ == "__main__":
        db_name = 'test_articles'
        coll_name = 'article_text_data'
        uri = 'mongodb://root:9EThDhBJiBGP@localhost'
        processor_filepath = '/home/bitnami/processor.pkl'
        classifier_filepath = '/home/bitnami/naivebayesclassifier.pkl'
        prep = TextPreprocessor()
        prep.db_pipeline(db_name, coll_name, uri)
        with open(processor_filepath, 'wb') as f:
            pickle.dump(prep, f)
