from APNEA import APNEA
import json
import csv
from sentiment_analyzers import DocumentTokenizer

with open('t_advertisers.json', 'r') as ad_fp:
    ads = json.load(ad_fp)

with open('blacklist.json', 'r') as bl_fp:
    blacklist = json.load(bl_fp)

SORT_ORDER = {
    "amazon": 1,
    "facebook": 2,
    "uk tourism": 3,
    "apple": 4,
    "nokia": 5,
    "adobe": 6,
    "comcast": 7,
    "flipkart": 8,
    "hp": 9,
    "netflix": 10,
    "at&t": 11,
    "atlassian": 12,
    "cisco": 13,
    "microsoft": 14,
    "samsung": 15,
    "glassdoor": 16,
    "yelp": 17,
    "paypal": 18,
    "ibm": 19,
    "virgin mobile": 20,
    "t-mobile": 21,
    "skype": 22,
    "lexmark": 23,
    "best buy": 24,
    "costco": 25,
    "home depot": 26,
    "sears": 27,
    "old navy": 28,
    "target": 29,
    "applebees": 30,
    "dominos": 31,
    "mcdonalds": 32,
    "ups": 33,
    "fedex": 34,
    "usps": 35,
    "dhl": 36,
    "france tourism": 37,
    "usa tourism": 38,
    "spain tourism": 39,
    "china tourism": 40,
    "italy tourism": 41,
    "tripadvisor": 42,
    "holiday inn": 43,
    "expedia": 44,
    "makemytrip": 45,
    "turkish airlines": 46,
    "emirates": 47,
    "lufthansa": 48,
    "malaysia airlines": 49,
    "united airlines": 50,
    "shell": 51,
    "china national petroleum corporation": 52,
    "exxon mobil": 53,
    "royal dutch shell": 54,
    "bp": 55,
    "chevron corporation": 56,
    "bmw": 57,
    "honda": 58,
    "fiat": 59,
    "ford": 60,
    "toyota": 61,
    "kia": 62,
    "volkswagen group": 63,
    "a-lawyers": 64,
    "hilton": 65,
    "marriott": 66,
    "avis": 67,
    "budget": 68
}


def process_url(url, apnea_obj):
    """
    Process and Print output for a single URL, for evaluation.
    :param url: URL to process
    :param apnea_obj: APNEA Object
    :return: None.
    """
    print url
    ranks = apnea_obj.page_ad_matching(url)

    enum_ranks = []
    for idx, elem in enumerate(ranks):
        enum_ranks.append((idx, elem[0]))

    enum_ranks.sort(key=lambda x: SORT_ORDER[x[1]])

    for elem in enum_ranks:
        print elem[0], '\t',
    print '\n',


def process_urls():
    """
    Process a CSV file for evaluation.
    :return: None
    """

    with open('valid.csv', 'r') as fp:
        csvreader = csv.reader(fp)
        csvreader.next()
        for row in csvreader:
            bushi_td = APNEA(ads, blacklist, params={'expand': True,
                                                         'use_sentiment': True,
                                                         'sent_analyzer': APNEA.SA_OPINION_MINER,
                                                         'scorer': APNEA.SC_SENT_TD,
                                                         'use_vector': True,
                                                         'neg_sent': True,
                                                         'targeted_sent': True,
                                                         'analysis_level': DocumentTokenizer.SENTENCE,
                                                         'n_gram': 3,
                                                         'blacklist': True,
                                                         'reduction_factor': 2.0})
            url = row[1]
            try:
                process_url(url, bushi_td)

            except Exception as e:
                print e


if __name__ == '__main__':
    process_urls()
