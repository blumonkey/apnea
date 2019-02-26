from __future__ import division
import hashlib
import json
from time import sleep
import numpy as np

import requests
import urllib
from bs4 import BeautifulSoup
from pycorenlp import StanfordCoreNLP
import unidecode
from nltk.corpus import wordnet as wn

# TODO: Fix this to get an offline version of Mercury/Readability


def clean_html(html):

    if html is None or html == '':
        return ''

    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text(' ', strip=True)

    # Converts unicode characters to the nearest ascii
    # TODO: Make this work with unicode itself.

    nearest_ascii = unidecode.unidecode(text)
    nearest_ascii = ' '.join(nearest_ascii.split())
    return nearest_ascii


def clean_url(url):
    BASE_URL = "https://mercury.postlight.com/parser"
    API_KEY = 's7QjSWSzwS8YuQWRBWycklzyV2F0Q7oP4LoMarLV'

    target_url = BASE_URL + '?' + urllib.urlencode({'url': url})

    # Handling the unstable API.
    timeout = 1
    MAX_TRIES = 5
    tries = 0

    content = ''
    title = ''

    while tries < MAX_TRIES:
        sleep(timeout)
        try:
            response = requests.get(target_url, headers={'x-api-key': API_KEY})
            json = response.json()
            title = json['title']
            content = json['content']
            if len(content) > 40:
                break
            else:
                tries += 1
                timeout *= 2
        except KeyError:
            tries += 1
            timeout *= 2
        except:
            tries += 1
            timeout *= 2

    if tries == 5:
        print 'MAX_TRIES exhausted!'

    clean_text = clean_html(content)
    # Matches all periods that are not followed by a space.
    # sentences_separated = re.sub(r'\.(?! )', '. ', clean_text)

    return {'text': clean_text, 'title': title, 'content': content}


def check_SNLP_server():
    StanfordCoreNLP('http://localhost:9000').annotate('Hello World!', properties={
        'annotators': 'tokenize',
        'outputFormat': 'json'
    })


def is_disambiguation_page(categories):
    for category in categories:
        if category['title'] == u'Category:All disambiguation pages':
            return True
    return False


def get_wiki_text(term):
    sleep(3)
    headers = {'User-Agent': 'SOCA/v0.1'}
    response = requests.get('https://en.wikipedia.org/w/api.php',
                            params={
                                'action': 'query',
                                'format': 'json',
                                'titles': term,
                                'prop': 'extracts|categories',
                                'explaintext': True,
                                'redirects': True
                            }, headers=headers).json()

    try:
        page_id = next(iter(response['query']['pages']))
        page = response['query']['pages'][page_id]
        text = page['extract']
        categories = page['categories']
    except KeyError:
        # TODO: Use the code from Fan to get the entity page and try again.
        return ''

    # TODO: Use the categories to add relevant terms
    if is_disambiguation_page(categories):
        return ''

    # Replace newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', '')

    # Replace special characters with space
    to_remove = '!"#$%&\'()\[\]*+,-./\\:;=<>'
    table = {ord(x): ord(' ') for x in to_remove}
    text = text.translate(table)

    # Encode to ascii, ignoring non-ascii chars
    # text = text.encode('ascii', 'ignore')
    text = unidecode.unidecode(text)
    # text = text.lower()
    return text


def compute_md5(obj):
    # From Stack overflow, https://stackoverflow.com/questions/40762886/
    return hashlib.md5(json.dumps(obj, sort_keys=True)).hexdigest()


def normalize(vec):
    total = np.sum(vec.values())
    for key in vec:
        vec[key] = vec[key] / total
    return vec


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''
