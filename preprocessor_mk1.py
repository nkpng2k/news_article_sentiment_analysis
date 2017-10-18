import requests
import spacy
import re
from stop_words import get_stop_words
from collections import Counter
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, BaseSentimentAnalyzer


class TextPreprocessor(object):

    def __init__(self, stop_words = 'en'):
        self.stop_words = get_stop_words(stop_words)

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
            encoded.append(word.encode('ascii', 'ignore'))
        return encoded

    def _remove_stop_words(self, tokens):
        stopped_tokens = [tok for tok in tokens if tok not in self.stop_words]
        return stopped_tokens

    def _vectorize(self, tokens):
        pass

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

    def lda_dim_reduction(article):
        encoded_tokens = self._tokenize_encode_ascii(article)
        stopped_tokens = self._remove_stop_words(encoded_tokens)
        vectorized = self._vectorize(stopped_tokens)

        #TODO: use lda to isolate topics


#THIS I THINK WILL BE COMPLETED MUCH LATER
    def db_pipeline(self, list_articles):
        for article in list_articles:
            cleaned = self._correct_sentences(article)

            #TODO: train NaiveBayesClassifier to predict on downloaded dataset
            #TODO: identify orthogonal words/n-grams for each noun/noun_phrase predict sentiment of those
            #TODO: predict sentiment of sentence/paragraph


if __name__ == "__main__":
    req = requests.get('http://www.foxnews.com/politics/2017/10/18/los-angeles-seeks-to-ban-tiki-torches-pepper-spray-shields-at-protests.html')
    soup = BeautifulSoup(req.text, 'html.parser')
    paragraphs = soup.find_all('p')

    article = ''
    for p in paragraphs:
        article = article + p.get_text()
    article = re.sub(r'(?<=[.!?])(?=[^\s])', r' \n', article)
    article.split()
    encoded = article.encode('ascii', ' ').split()
    encoded = []
    for word in article.split():
        encoded.append(word.encode('ascii', 'ignore'))

    stop_words = get_stop_words('en')
    stopped_tokens = [tok for tok in encoded if tok not in stop_words]
    stopped_tokens


    blob = TextBlob(article, analyzer = NaiveBayesAnalyzer())

    for sentence in blob.sentences:
        print sentence, sentence.sentiment

    test = TextBlob('stealing', analyzer = NaiveBayesAnalyzer())
    test.sentiment

    nlp = spacy.load('en')
    doc = nlp(article)
    list(doc.sents)
    list(doc.noun_chunks)

    for word in doc:
        for character in word.string:
            print word.string
        # print word.text, word.orth_, word.lemma_, word.tag, word.tag_, word.pos, word.pos_


    def pos_words(sentence, token, ptag):
        sentences = [sent for sent in sentence.sents if token in sent.string]
        pwrds = []
        for sent in sentences:
            for word in sent:
                for character in word.string:
                       pwrds.extend([child.string.strip() for child in word.children
                                                          if child.pos_ == ptag] )
        return Counter(pwrds).most_common(10)

    pos_words(doc, 'weapons', 'VERB')
