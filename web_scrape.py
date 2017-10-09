import pymongo
import requests
from bs4 import BeautifulSoup

def retrieve_key(file_path, api):
    with open(file_path) as f:
        for line in f:
            if line.startswith(api):
                api_key = line.split(':')[1].strip()
        return api_key


def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        return response.json()


def mult_queries_by_topic(topic_url)

if __name__ == '__main__':
    path = '/Users/npng/.ssh/api_keys.txt'
    api_wanted = "nyt"
    api_key = retrieve_key(file_path = path, api = api_wanted)
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {'api-key': api_key}
    html_str = single_query(link, payload)


    req = requests.get('https://www.nytimes.com/section/politics')
    soup = BeautifulSoup(req.text, 'html.parser')

    filter_soup = soup.find_all('a', class_='story-link')

    for row in filter_soup:
        print row['href']
