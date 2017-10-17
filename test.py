import requests
import csv
from bs4 import BeautifulSoup
import pymongo
import time
import pandas as pd
import numpy as np


class ArticleData(object):

    def __init__(self, article_filepath, error_filepath, urls_filepath, db_name, coll_name, uri = None):
        self.article_file = open(article_filepath, 'a')
        self.error_file = open(error_filepath, 'a')
        self.article_writer = csv.writer(self.article_file)
        self.error_writer = csv.writer(self.error_file)
        self.urls_df = pd.read_csv(urls_filepath, header = None).sample(frac = 1)
        self.db_name = db_name
        self.coll_name = coll_name
        self.coll = self._launch_mongo(uri)
        self.coll.create_index('url', unique = True)


    def _launch_mongo(self, uri):
        mc = pymongo.MongoClient(uri)
        db = mc[self.db_name]
        coll = db[self.coll_name]
        return coll


    def _gen_request(self, url):
        req = requests.get(str(url.strip()))
        if req.status_code != 200:
            print 'WARNING', req.status_code
            self.error_writer.writerow(('nyt', url))
        else:
            return req

    def _check_duplicate(self, url):
        count = self.coll.find({'url':url}).count()
        if count > 0:
            return False
        else:
            return True


    def nyt_articles(self, url):
        req = self._gen_request(url)

        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find('h1', class_='headline').text
        paragraphs = soup.find_all('p', class_='story-body-text story-content')
        article = ''

        for p in paragraphs:
            article = article + p.get_text()
        print headline, article

        print "success"
        return headline, article

    def guardian_articles(self, url):
        req = self._gen_request(url)

        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find('h1', class_='content__headline').text
        paragraphs = soup.find('article', id='article').find_all('p')
        article = ''

        for p in paragraphs:
            article = article + p.get_text()
        print headline, article

        print "success"
        return headline, article


    def cnn_articles(self, url):
        req = self._gen_request(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find('h1', class_ = 'pg-headline').text
        article = soup.find('div', class_ = 'pg-rail-tall__body').text
        print headline, article

        print "success"
        return headline, article


    def wap_articles(self, url):
        req = self._gen_request(url)

        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find('h1', itemprop = 'headline').text
        article = soup.find('div', class_ = 'article-body').text
        print headline, article

        print "success"
        return headline, article


    def fox_articles(self, url):
        req = self._gen_request(url)

        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find('h1', class_ = 'headline head1').text
        article = soup.find('div', class_ = 'article-body').text
        print headline, article

        print "success"
        return headline, article


    def scraper(self):
        for i, row in self.urls_df.iterrows():
            if self._check_duplicate(row[2]):
                print "sending scrape request"
                print row[1], row[2]
                if row[1] == 'nyt':
                    headline, article = self.nyt_articles(row[2])
                    self.coll.insert_one({'site':row[1], 'headline': headline, 'article': article, 'url':row[2]})
                elif row[1] == 'guardian':
                    headline, article = self.guardian_articles(row[2])
                    self.coll.insert_one({'site':row[1], 'headline': headline, 'article': article, 'url':row[2]})
                elif row[1] == 'wash_post':
                    headline, article = self.wap_articles(row[2])
                    self.coll.insert_one({'site':row[1], 'headline': headline, 'article': article, 'url':row[2]})
                elif row[1] == 'cnn':
                    headline, article = self.cnn_articles(row[2])
                    self.coll.insert_one({'site':row[1], 'headline': headline, 'article': article, 'url':row[2]})
                elif row[1] == 'fox':
                    headline, article = self.fox_articles(row[2])
                    self.coll.insert_one({'site':row[1], 'headline': headline, 'article': article, 'url':row[2]})
                else:
                    print 'NOT A REAL ARTICLE'
            else:
                print "duplicate url"

            time.sleep(5 + np.random.rand())


if __name__ == "__main__":
    scrape_articles = ArticleData('article_data.csv', 'errors.csv', 'cleaned_articles_urls.csv', 'news_articles', 'article_text_data', uri = None)
    scrape_articles.scraper()
