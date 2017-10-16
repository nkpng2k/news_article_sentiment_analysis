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
import csv

class SeleniumUrls(object):

    def __init__(self, filepath, csv_name, site_name, uri = None):
        self.site_name = site_name
        self.driver = None
        self.url_base = None
        self.filepath = filepath
        self.file = open(filepath + csv_name +'.csv', 'a')
        self.writer = csv.writer(self.file)

    def _check_health(self, urls, site_url, class_name, tag):
        if len(urls) < 1:
            print 'reloading driver'
            self.driver.quit()
            self.driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
            self.driver.get(site_url)
            time.sleep(10)
            urls = self.driver.find_elements_by_class_name(class_name)[0].find_elements_by_tag_name(tag)
        return urls

    def _retrieve_urls(self, urls, art_id):
        article_urls = set()
        print len(urls)
        for url in urls:
            candidate_url = str(url.get_attribute('href'))
            if art_id:
                if art_id in candidate_url.split('/'):
                    article_urls.add((self.site_name, candidate_url))
            else:
                article_urls.add((self.site_name, candidate_url))

        return article_urls

    def _scrape(self, pg_num, class_name, tag, art_id = None, from_date = None, to_date = None):
        page_number = pg_num
        increment_ten = pg_num*10
        increment_twenty = pg_num*20
        site_url = self.url_base.format(pg_num = page_number,inc_ten = increment_ten,inc_twenty = increment_twenty, from_date = from_date, to_date = to_date)
        self.driver.get(site_url)
        print "loaded page {}, waiting 10 seconds".format(page_number)
        time.sleep(10)
        urls = self.driver.find_elements_by_class_name(class_name)[0].find_elements_by_tag_name(tag)
        urls = self._check_health(urls, site_url, class_name, tag)
        article_urls = self._retrieve_urls(urls, art_id)

        self.writer.writerows(article_urls)

        print "page {} done".format(page_number)

    def get_urls_page_number(self, url_base, num_pages, class_name, tag , start_page = 1, date_ranges = None, art_id = None):
        """
        Launches Mongo instance and stores urls in collection within database
        Inputs: url_base: base url --> format, "www.-----/page={}".format(page_number)
                num_pages: number of pages wanted to scrape , currently about 10 per page so need ~1000 pages
        Outputs: None
        """
        self.url_base = url_base
        self.driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
        if date_ranges:
            for dates in date_ranges:
                from_date, to_date = dates
                for i in xrange(start_page, num_pages+1):
                    self._scrape(i, class_name, tag, art_id, from_date, to_date)
        else:
            for i in xrange(start_page, num_pages+1):
                self._scrape(i, class_name, tag, art_id)
        self.driver.quit()

if __name__ == '__main__':
    nyt = "https://query.nytimes.com/search/sitesearch/?action=click&contentCollection&region=TopBar&WT.nav=searchWidget&module=SearchSubmit&pgtype=Homepage#/politics/from{from_date}to{to_date}/document_type%3A%22article%22/{pg_num}/allauthors/newest/"
    guardian = "https://www.theguardian.com/us-news/us-politics?page={pg_num}"
    wash_post = "https://www.washingtonpost.com/newssearch/?query=politics&sort=Date&datefilter=All%20Since%202005&contenttype=Article&spellcheck&startat={inc_twenty}#top"
    wsj = "https://www.wsj.com/search/term.html?KEYWORDS=politics&min-date=2013/10/09&max-date=2017/10/09&page={pg_num}&isAdvanced=true&daysback=4y&andor=AND&sort=date-desc&source=wsjarticle"
    cnn = "http://www.cnn.com/search/?q=politics&size=10&page={pg_num}&type=article&from={inc_ten}"
    fox = "http://www.foxnews.com/search-results/search?q=politics&ss=fn&sort=latest&start={inc_ten}"
    filepath = '/Users/npng/galvanize/dsi/news_article_sentiment_analysis/'
    #NYT --> element = searchResults, tag = a
    # d_ranges = [('20171001','20171031'), ('20170901','20170930'), ('20170801', '20170830'),\
    #             ('20170701', '20170730'), ('20170601', '20170630'), ('20170501', '20170530'), ('20170401', '20170430'),\
    #             ('20170301', '20170430'), ('20170301', '20170330'), ('20170201', '20170228'), ('20170101', '20170130')]
    # nyt_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'nyt')
    # nyt_selenium.get_urls_page_number(nyt, 100, 'searchResults', 'a', date_ranges = d_ranges)

    # #WSJ --> element = search-results-sector, tag = a, art_id = 'articles'
    # wsj_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'wsj')
    # wsj_selenium.get_urls_page_number(wsj, 750, 'search-results-sector', 'a', start_page = 435, art_id = 'articles')

    # #guardian --> element = l-side-margins, tag = a, art_id = www.theguardian.com
    # guardian_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'guardian')
    # guardian_selenium.get_urls_page_number(guardian, 1000, 'l-side-margins', 'a', start_page = 684, art_id = 'www.theguardian.com')
    #
    # #washington post --> element = 'pb-results-container', tag = a , art_id = www.washingtonpost.com, INCREMENTS!
    # wash_post_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'wash_post')
    # wash_post_selenium.get_urls_page_number(wash_post, 500, 'pb-results-container', 'a', art_id = 'www.washingtonpost.com')
    #
    # #cnn --> element = cnn-search__results, tag = a, art_id = www.cnn.com
    # cnn_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'cnn')
    # cnn_selenium.get_urls_page_number(cnn, 1000, 'cnn-search__results', 'a', art_id = 'www.cnn.com')

    #fox --> element = ng-scope , tag = a, art_id = www.foxnews.com
    fox_selenium = SeleniumUrls(filepath = filepath, csv_name = 'news_articles', site_name = 'fox')
    fox_selenium.get_urls_page_number(fox, 1000, 'ng-scope', 'a',start_page = 949, art_id = 'www.foxnews.com')


"""
bottom of page
"""
