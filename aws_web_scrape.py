"""
This will be the webscraper loaded into an AWS EC2 instance that will get news articles from
News websites.

Specifically: NYT, CNN, FOX, WSJ, The Guardian, and Washington Post
"""

import requests
from bs4 import BeautifulSoup

#functions needed for all websites
def make_soup(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    return soup

def get_urls(url):
    pass
#website distinct functions
def get_fox_urls(url):
    soup = make_soup(url)


def get_nyt_urls(url):
    soup = make_soup(url)

def get_wash_post_urls(url):
    soup = make_soup(url)

def get_guardian_urls(url):
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



    req = requests.get('http://www.foxnews.com/search-results/search?q=politics&ss=fn&sort=latest&start=10')
    soup = BeautifulSoup(req.text, 'html.parser')
    soup.find_all('#search-container > div:nth-child(4)')
    #search-container > div:nth-child(4)









"""
bottom of page
"""
