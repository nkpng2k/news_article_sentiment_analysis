import requests
import spacy
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, BaseSentimentAnalyzer

"""
Preprocessor steps:
1. Separate Data Into Paragraphs or sentences?
2. For each sentence identify 1,2,or 3 word n-grams as nouns
3. Find sentiment within each sentence associate to the word
"""

req = requests.get('https://www.theguardian.com/us-news/2017/oct/17/senators-reach-bipartisan-deal-to-salvage-obamacare-subsidies-trump-eliminated')
soup = BeautifulSoup(req.text, 'html.parser')
paragraphs = soup.find_all('p')

article = ''
for p in paragraphs:
    article = article + p.get_text()
article = re.sub(r'(?<=[.!?])(?=[^\s])', r' \n', article)

blob = TextBlob(article, analyzer = NaiveBayesAnalyzer())

for sentence in blob.sentences:
    print sentence, sentence.sentiment

test = TextBlob('taxing', analyzer = NaiveBayesAnalyzer())
test.sentiment

nlp = spacy.load('en')
doc = nlp(article)
list(doc.sents)
list(doc.noun_chunks)
for chunks in doc.noun_chunks:
    print chunks.text, str(list(chunks.children))
    # print word.text, word.orth_, word.lemma_, word.tag, word.tag_, word.pos, word.pos_


class TextPreprocessor(object):

    def __init__(self):
        pass

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

    def new_article(self, url):
        req = self._gen_request(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ''
        for p in paragraphs:
            article = article + p.get_text()

        clean = self._correct_sentences(article)
        return clean

    def pipeline(self, list_articles):
        for article in list_articles:
            cleaned = self._correct_sentences(article)

            #TODO: train NaiveBayesClassifier to predict on downloaded dataset
            #TODO: identify orthogonal words/n-grams for each noun/noun_phrase predict sentiment of those
            #TODO: predict sentiment of sentence/paragraph
