"""
This will be the webscraper loaded into an AWS EC2 instance that will get news articles from
News websites.

Specifically: NYT, CNN, FOX, WSJ, The Guardian, and Washington Post
"""

import pymongo
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

class SeleniumUrls(object):

    def __init__(self, db_name, collection_name, site_name, uri = None):
        self.db_name = db_name
        self.collection_name = collection_name
        self.coll = self._launch_mongo(uri)
        self.site_name = site_name

    def _launch_mongo(self, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc.get_database(self.db_name)
        coll = db.get_collection(self.collection_name)

        return coll

    def _check_health(self, driver, urls, site_url, class_name, tag):
        if len(urls) < 1:
            print 'reloading driver'
            driver.quit()
            driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
            driver.get(site_url)
            time.sleep(10)
            urls = driver.find_elements_by_class_name(class_name)[0].find_elements_by_tag_name(tag)
        return urls, driver

    def _retrieve_urls(self, urls, art_id):
        article_urls = []
        print len(urls)
        for url in urls:
            candidate_url = str(url.get_attribute('href'))
            if art_id:
                if art_id in candidate_url.split('/'):
                    article_urls.append(candidate_url)
            else:
                article_urls.append(candidate_url)

        return article_urls


    def get_urls_page_number(self, url_base, num_pages, class_name, tag , date_ranges = None, art_id = None):
        """
        Launches Mongo instance and stores urls in collection within database
        Inputs: url_base: base url --> format, "www.-----/page={}".format(page_number)
                num_pages: number of pages wanted to scrape , currently about 10 per page so need ~1000 pages
        Outputs: None
        """
        driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
        for i in xrange(1, num_pages+1):
            page_number = i
            increment_ten = i*10
            increment_twenty = i*20
            site_url = url_base.format(pg_num = page_number,inc_ten = increment_ten,inc_twenty = increment_twenty)
            driver.get(site_url)
            print "loaded page {}, waiting 10 seconds".format(i)
            time.sleep(10)
            urls = driver.find_elements_by_class_name(class_name)[0].find_elements_by_tag_name(tag)
            urls, driver = self._check_health(driver, urls, site_url, class_name, tag)
            article_urls = self._retrieve_urls(urls, art_id)

            self.coll.find_one_and_update({'site':self.site_name}, { '$addToSet':{'urls':{ '$each' : article_urls}}}, upsert = True)

            print "page {} done".format(i)
        driver.quit()

if __name__ == '__main__':
    page_number = 1
    increments_twenty = 0
    increment_ten = 0
    nyt = "https://query.nytimes.com/search/sitesearch/?action=click&contentCollection&region=TopBar&WT.nav=searchWidget&module=SearchSubmit&pgtype=Homepage#/politics/from{from_date}to{to_date}/document_type%3A%22article%22/{pg_num}/allauthors/newest/"
    guardian = "https://www.theguardian.com/us-news/us-politics?page={pg_num}"
    wash_post = "https://www.washingtonpost.com/newssearch/?query=politics&sort=Date&datefilter=All%20Since%202005&contenttype=Article&spellcheck&startat={inc_twenty}#top"
    wsj = "https://www.wsj.com/search/term.html?KEYWORDS=politics&min-date=2013/10/09&max-date=2017/10/09&page={pg_num}&isAdvanced=true&daysback=4y&andor=AND&sort=date-desc&source=wsjarticle"
    cnn = "http://www.cnn.com/search/?q=politics&size=10&page={pg_num}&type=article&from={inc_ten}"
    fox = "http://www.foxnews.com/search-results/search?q=politics&ss=fn&sort=latest&start={inc_ten}"


    # #NYT --> element = searchResults, tag = a
    # nyt_selenium = SeleniumUrls(db_name = 'news_articles', collection_name = 'urls', site_name = 'nyt')
    # nyt_selenium.get_urls_page_number(nyt, 1500, 'searchResults', 'a')

    #WSJ --> element = search-results-sector, tag = a, art_id = 'articles'
    wsj_selenium = SeleniumUrls(db_name = 'news_articles', collection_name = 'urls', site_name = 'wsj')
    wsj_selenium.get_urls_page_number(wsj, 750, 'search-results-sector', 'a', art_id = 'articles')

    #guardian --> element = l-side-margins, tag = a, art_id = www.theguardian.com
    guardian_selenium = SeleniumUrls(db_name = 'news_articles', collection_name = 'urls', site_name = 'guardian')
    guardian_selenium.get_urls_page_number(guardian, 1000, 'l-side-margins', 'a', art_id = 'www.theguardian.com')

    #washington post --> element = 'pb-results-container', tag = a , art_id = www.washingtonpost.com, INCREMENTS!
    wash_post_selenium = SeleniumUrls(db_name = 'news_articles', collection_name = 'urls', site_name = 'wash_post')
    wash_post_selenium.get_urls_page_number(wash_post, 500, 'pb-results-container', 'a', art_id = 'www.washingtonpost.com')


    # driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
    # driver.get("https://www.wsj.com/search/term.html?KEYWORDS=politics&min-date=2013/10/09&max-date=2017/10/09&page=1&isAdvanced=true&daysback=4y&andor=AND&sort=date-desc&source=wsjarticle")
    # urls = driver.find_elements_by_class_name('search-results-sector')[0].find_elements_by_tag_name('a')
    # article_urls = []
    # for url in urls:
    #     candidate_url = str(url.get_attribute('href'))
    #     if 'articles' in candidate_url.split('/'):
    #         article_urls.append(candidate_url)
    #
    #
    # article_urls


"""
bottom of page
"""
