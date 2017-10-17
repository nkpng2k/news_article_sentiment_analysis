import requests
import spacy
import re
from bs4 import BeautifulSoup
from textblob import TextBlob

"""
Preprocessor steps:
1. Separate Data Into Paragraphs or sentences?
2. For each sentence identify 1,2,or 3 word n-grams as nouns
3. Find sentiment within each sentence associate to the word
"""

req = requests.get('http://www.foxnews.com/politics/2017/10/17/trump-doubles-down-on-slain-soldier-comments-obama-didnt-call-john-kelly-when-son-died.html')
soup = BeautifulSoup(req.text, 'html.parser')
paragraphs = soup.find_all('p')

article = ''
for p in paragraphs:
    article = article + p.get_text()
article = re.sub(r'(?<=[.!?])(?=[^\s])', r' ', article)

blob = TextBlob(article)

for sentence in blob.sentences:
    print sentence.noun_phrases

blob.sentiment.polarity



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
        add_spaces = re.sub(r'(?<=[.!?])(?=[^\s])', r' ', text)
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
