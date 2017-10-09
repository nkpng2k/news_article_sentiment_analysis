import pymongo
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from collections import defaultdict
import time
import pymongo

# def retrieve_key(file_path, api):
#     with open(file_path) as f:
#         for line in f:
#             if line.startswith(api):
#                 api_key = line.split(':')[1].strip()
#         return api_key
#
#
# def single_query(link, payload):
#     response = requests.get(link, params=payload)
#     if response.status_code != 200:
#         print 'WARNING', response.status_code
#     else:
#         return response.json()


def run_selenium(base_url, topics):
    driver = webdriver.Chrome('/Users/npng/.ssh/chromedriver')
    url_dict = defaultdict(list)
    for topic in topics:
        driver.get(base_url + topic)
        button = True
        driver.find_element_by_css_selector('button.button.load-more-button').click()
        for i in xrange(10):
            print 'scrolling once'
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
        urls = driver.find_elements_by_class_name('story-link')
        for url in urls:
            url_dict[topic].append(str(url.get_attribute('href')))
    driver.quit()
    return url_dict


def one_request(url, topic):
    article_dict = defaultdict(list)
    req = requests.get(url)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        soup = BeautifulSoup(req.text, 'html.parser')
        headline = soup.find_all('h1', class_='headline')[0].contents[0]
        author = soup.find_all('span', class_= 'byline-author')[0]['data-byline-name']
        date = soup.find_all('time', class_= 'dateline')[0]['datetime']
        paragraphs = soup.find_all('p', class_='story-body-text story-content')
        article = ''
        for p in paragraphs:
            article = article + p.get_text()

        for item in [author, date, article, topic]:
            article_dict[headline].append(item)

    return article_dict


def scrape(url_dictionary, coll):
    topics = url_dictionary.keys()
    for top in topics:
        print 'Starting --> Topic: {}'.format(top)
        url_list = url_dictionary[top]
        for url in url_list:
            print 'Starting --> Topic: {} --- URL: {}'.format(top, url)
            art = one_request(url, top)
            coll.insert_one(art)
            print 'Moving To Next Article in 30 seconds'
            time.sleep(30)
        print 'COMPLETE ALL ARTICLES'
    print 'COMPLETE ALL TOPICS'

def launch_mongo(database, collection, uri = None):
    mc = pymongo.MongoClient(uri)
    db = mc[database]
    coll = db[collection]

    return coll


if __name__ == '__main__':
    # path = '/Users/npng/.ssh/api_keys.txt'
    # api_wanted = "nyt"
    # api_key = retrieve_key(file_path = path, api = api_wanted)
    # link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    # payload = {'api-key': api_key}
    # html_str = single_query(link, payload)

    mongo_coll = launch_mongo('news_articles', 'nyt_articles')
    topics = ['politics', 'business', 'world', 'us', 'science', 'health']
    base_url = 'https://www.nytimes.com/section/'
    url_dict = run_selenium(base_url, topics)
    scrape(url_dict, mongo_coll)




    # req = requests.get('https://www.nytimes.com/section/politics?action=click&pgtype=Homepage&region=TopBar&module=HPMiniNav&contentCollection=Politics&WT.nav=page')
    # soup = BeautifulSoup(req.text, 'html.parser')
    #
    # filter_soup = soup.find_all('a', class_='story-link')
    # for row in filter_soup:
    #     print row['href']
