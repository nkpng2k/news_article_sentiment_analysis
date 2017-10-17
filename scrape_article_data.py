import requests
import csv
from bs4 import BeautifulSoup
import pymongo
import time
import pandas as pd
import numpy as np


class ArticleData(object):

    def __init__(self, error_filepath, urls_filepath, db_name, coll_name, uri = None):
        self.error_file = open(error_filepath, 'a')
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
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            paragraphs = soup.find_all('p', class_='story-body-text story-content')
            article = ''

            for p in paragraphs:
                article = article + p.get_text()

            print "success"
            return article
        except:
            print "ERROR MOVING ON"
            self.error_writer.writerow(('nyt', url))

    def guardian_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            paragraphs = soup.find('div', class_="content__article-body from-content-api js-article__body").find_all('p')
            article = ''

            for p in paragraphs:
                article = article + p.get_text()

            print "success"
            return article
        except:
            print "ERROR MOVING ON"
            self.error_writer.writerow(('guardian',url))

    def cnn_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            paragraphs = soup.find_all('p', class_ = 'zn-body__paragraph speakable')
            article = ''

            for p in paragraphs:
                article = article + p.get_text()

            print "success"
            return article
        except:
            print "ERROR MOVING ON"
            self.error_writer.writerow(('cnn', url))

    def wap_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            article = soup.find('div', class_ = 'article-body').text

            print "success"
            return article
        except:
            print "ERROR MOVING ON"
            self.error_writer.writerow(('wash_post', url))

    def fox_articles(self, url):
        req = self._gen_request(url)
        try:
            soup = BeautifulSoup(req.text, 'html.parser')
            paragraphs = soup.find('div', class_ = 'article-body').find_all('p')
            article = ''

            for p in paragraphs:
                article = article + p.get_text()

            print "success"
            return article
        except:
            print "ERROR MOVING ON"
            self.error_writer.writerow(('fox', url))

    def scraper(self):
        for i, row in self.urls_df.iterrows():
            if self._check_duplicate(row[2]):
                print "sending scrape request"
                print row[1], row[2]
                try:
                    if row[1] == 'nyt':
                        article = self.nyt_articles(row[2])
                        self.coll.insert_one({'site':row[1], 'article': article, 'url':row[2]})
                        print 'in mongodb'
                    elif row[1] == 'guardian':
                        article = self.guardian_articles(row[2])
                        self.coll.insert_one({'site':row[1], 'article': article, 'url':row[2]})
                        print 'in mongodb'
                    elif row[1] == 'wash_post':
                        article = self.wap_articles(row[2])
                        self.coll.insert_one({'site':row[1], 'article': article, 'url':row[2]})
                        print 'in mongodb'
                    elif row[1] == 'cnn':
                        article = self.cnn_articles(row[2])
                        self.coll.insert_one({'site':row[1], 'article': article, 'url':row[2]})
                        print 'in mongodb'
                    elif row[1] == 'fox':
                        article = self.fox_articles(row[2])
                        self.coll.insert_one({'site':row[1], 'article': article, 'url':row[2]})
                        print 'in mongodb'
                    else:
                        print 'NOT A REAL ARTICLE'
                except:
                    print "error error error"
                    self.error_writer.writerow((row[1], row[2]))
            else:
                print "duplicate url"

            time.sleep(2 + np.random.rand())


if __name__ == "__main__":
    scrape_articles = ArticleData('errors.csv', 'cleaned_articles_urls.csv', 'news_articles', 'article_text_data', uri = 'mongodb://root:TWV7Y1t7hS7P@localhost')
    scrape_articles.scraper()
