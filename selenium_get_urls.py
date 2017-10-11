"""
This will be the webscraper loaded into an AWS EC2 instance that will get news articles from
News websites.

Specifically: NYT, CNN, FOX, WSJ, The Guardian, and Washington Post
"""

import pymongo
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

class SeleniumUrls(object):

    def __init__(self, db_name, collection_name, uri = None):
        self.db_name = db_name
        self.collection_name = collection_name
        self.coll = _launch_mongo(uri)

    def _launch_mongo(self, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc.get_database(self.db_name)
        coll = mc.get_database(self.collection_name)

        return coll

    def get_urls_page_number(self, url_base, num_pages , increments = None):
        """
        Launches Mongo instance and stores urls in collection within database
        Inputs: url_base: base url --> format, "www.-----/page={}".format(page_number)
                num_pages: number of pages wanted to scrape , currently about 10 per page so need ~1000 pages
                increments: some websites have both page number and increment if None provided, ignore
        Outputs: None
        """
        driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
        for i in xrange(1, num_wanted+1):
            page_number = i
            increment_ten = i*10
            article_urls = set()
            driver.get(url_base)
            urls = driver.find_elements_by_class_name('story-link') #TODO: change this to actual class name
            for url in urls:
                article_urls.add(url.get_attribute('href'))

            #TODO: add to collection in mongo instance


    def get_fox_urls(url):
        soup = make_soup(url)


    def get_nyt_urls(url):
        soup = make_soup(url)

    def get_wash_post_urls(url):
        soup = make_soup(url)

    def get_cnn_urls(url):
        soup = make_soup(url)

    def get_wsj_urls(url):
        soup = make_soup(url)


if __name__ == '__main__':
    page_number = 1
    increments_twenty = 0
    increment_ten = 0
    nyt = "https://query.nytimes.com/search/sitesearch/?action=click&contentCollection&region=TopBar&WT.nav=searchWidget&module=SearchSubmit&pgtype=Homepage#/politics/since1851/document_type%3A%22article%22/{}/allauthors/newest/".format(page_number)
    guardian = "https://www.theguardian.com/us-news/us-politics?page={}".format(page_number)
    wash_post = "https://www.washingtonpost.com/newssearch/?query=politics&sort=Date&datefilter=All%20Since%202005&contenttype=Article&spellcheck&startat={}#top".format(increments_twenty)
    wsj = "https://www.wsj.com/search/term.html?KEYWORDS=politics&min-date=2013/10/09&max-date=2017/10/09&page={}&isAdvanced=true&daysback=4y&andor=AND&sort=date-desc&source=wsjarticle".format(page_number)
    cnn = "http://www.cnn.com/search/?q=politics&size=10&page={}&type=article&from={}".format(page_number, increment_ten)
    fox = "http://www.foxnews.com/search-results/search?q=politics&ss=fn&sort=latest&start={}".format(increment_ten)


    req = requests.get(cnn)
    soup = BeautifulSoup(req.text, 'html.parser')
    soup.find_all('h3')









"""
bottom of page
"""
