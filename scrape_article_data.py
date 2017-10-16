import requests
import csv
from bs4 import BeautifulSoup


class ArticleData(object):

    def __init__(self, article_filepath, error_filepath, urls_filepath):
        self.article_file = open(article_filepath, 'a')
        self.error_file = open(error_filepath, 'a')
        self.article_writer = csv.writer(self.article_file)
        self.error_writer = csv.writer(self.error_file)
        self.urls_df = pd.read_csv(urls_filepath, header = None)

    def _gen_request(self, url):
        req = request.get(url)
        if req.status_code != 200:
            print 'WARNING', req.status_code
            self.error_writer.write(('nyt', url))
        else:
            return req

    def nyt_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            headline = soup.find('h1', class_='headline').text
            author = soup.find('span', class_= 'byline-author')['data-byline-name']
            date = soup.find('time', class_= 'dateline')['datetime']
            paragraphs = soup.find_all('p', class_='story-body-text story-content')
            article = ''

            for p in paragraphs:
                article = article + p.get_text()

            self.article_writer.write((headline, author, date, article))
        except:
            print "ERROR MOVING ON"
            self.error_writer.write(('nyt', url))

    def guardian_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            headline = soup.find('h1', class_='content__headline').text
            author = soup.find('a', rel = 'author').text
            date = soup.find('time', class_ = 'content__dateline-wpd js-wpd')['datetime']
            article = soup.find('div', class_='content__article-body from-content-api js-article__body').text

            self.article_writer.write(headline, author, date, article)
        except:
            print "ERROR MOVING ON"
            self.error_writer.write(('guardian',url))

    def cnn_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')

            headline = soup.find('h1', class_ = 'pg-headline').text
            author = soup.find('p', class_ = 'metadata__byline').find('a').text
            article = soup.find('div', class_ = 'pg-rail-tall__body').text
            date = soup.find('p', class_ = 'update-time').text

            self.article_writer.write((headline, author, date, article))
        except:
            print "ERROR MOVING ON"
            self.error_writer.write(('cnn', url))

    def wap_articles(self, url):
        req = self._gen_request(url)
        try:
            headline = soup.find('h1', itemprop = 'headline').text
            author = soup.find('span', class_ = 'pb-byline').text
            date = soup.find('span', class_ = 'pb-timestamp')['content']
            article = soup.find('article', class_ = 'paywall').text

            self.article_writer.write((headline, author, date, article))
        except:
            print "ERROR MOVING ON"
            self.error_writer.write(('wash_post', url))

    def fox_articles(self, url):
        req = self._gen_request(url)
        try:
            headline = soup.find('h1', class_ = 'headline head1').text
            author = soup.find('div', class_ = 'author-byline').text
            date = soup.find('div', class_ = 'article-date').find('time')['data-time-published']
            article = soup.find('div', class_ = 'article-body').text

            self.article_writer.write((headline, author, date, article))
        except:
            print "ERROR MOVING ON"
            self.error_writer.write(('fox', url))

    def scraper(urls):
        for i, row in self.urls_df.iterrows():
            if row[0]







if __name__ = "__main__":
    url = 'http://www.foxnews.com/world/2017/10/16/north-korea-suspected-source-mad-dog-trump-leaflets.html'
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
