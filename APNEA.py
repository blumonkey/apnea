from __future__ import division
import nltk

from utils import clean_url, check_SNLP_server, get_wiki_text, compute_md5
from sentiment_analyzers import LexiconBased, DocumentTokenizer, SentimentAnalyzerSVM, StanfordSentimentAnalyzer
import logging
import numpy as np
from math import log
from collections import Counter
from nltk.tag import pos_tag
import json
import pickle
import re


class APNEA:
    # Metadata filenames
    META_FILE = 'APNEA.db'
    MD5_FILE = 'APNEA.db.md5'

    # Sentiment Analyzer Models.
    SA_SVM = 'svm'
    SA_REDDIT_NEWS = 'reddit_news'
    SA_OPINION_MINER = 'opinion_miner'
    SA_SNLP = 'snlp'
    SA_SENTIWORDNET = 'sentiwordnet'
    SA_BL_POS = 'baseline_pos'

    # Scoring Functions
    SC_RATIO_LOG = 'RATIO_LOG'
    SC_SUB = 'SUB'
    SC_RATIO = 'RATIO_TB'

    SC_SENT_MAX = 'SENT_MAX'
    SC_SENT_TD = 'SENT_TD'
    SC_SENT_TM = 'SENT_TM'

    def __init__(self, ads, blacklist=None, params=None):
        """
        Initialzation Point for the APNEA system.
        :param ads: Input Advertisements, as a Python Dictionary, formatted as in t_advertisers.json
        :param blacklist: Optional Input Blacklist, formatted as in blacklist.json
        :param params: APNEA system parameters.
        """
        # Error handling
        if type(ads) is not dict:
            raise TypeError('ads: Expected dict, got ' + type(ads))

        if params and type(params) is not dict:
            raise TypeError('params: Expected dict, got ' + type(params))

        if blacklist is None:
            blacklist = {}

        # Check if the Stanford CoreNLP Server is Running.
        check_SNLP_server()
        self.logger = logging.getLogger('APNEA')

        # Default Parameters
        self.params = {
            'epsilon': 0.01,
            'expand': True,
            'sent_analyzer': APNEA.SA_OPINION_MINER,
            'analysis_level': DocumentTokenizer.SENTENCE,
            'scorer': APNEA.SC_SENT_TD,
            'use_sentiment': True,
            'use_vector': True,
            'neg_sent': True,
            'targeted_sent': False,
            'blacklist': False,
            'reduction_factor': 2.0
        }

        if params:
            for key in params.iterkeys():
                self.params[key] = params[key]

        # Curated Stopwords from the nltk.corpus module
        self.stopwords = {u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
                          u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',
                          u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs',
                          u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those',
                          u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
                          u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if',
                          u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about',
                          u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to',
                          u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again',
                          u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
                          u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'only',
                          u'own', u'same', u'so', u'than', u'too', u'very', u'can', u'will', u'just', u'should',
                          u'now'}

        # Data structures to find triggered advertisements.
        self.keyword_to_company = {}
        self.blacklist_to_company = self.invert_dictionary(blacklist)
        self.all_companies = {}
        self.all_companies_scalar = {}

        # Data structures to find relevance between ad and page
        self.ads = ads
        self.ad_keywords = {}

        # Dict to check if company cares about the sentiment of the bidding words.
        self.ad_negatives = {}

        # Initialization of the url-independent objects
        if self.params['sent_analyzer'] == APNEA.SA_SVM:
            self.analyzer_obj = SentimentAnalyzerSVM()
        elif self.params['sent_analyzer'] == APNEA.SA_REDDIT_NEWS:
            self.analyzer_obj = LexiconBased('resources/news.pkl')
        elif self.params['sent_analyzer'] == APNEA.SA_SNLP:
            self.analyzer_obj = StanfordSentimentAnalyzer()
        elif self.params['sent_analyzer'] == APNEA.SA_SENTIWORDNET:
            self.analyzer_obj = LexiconBased('resources/sentiwordnet.pkl')
        elif self.params['sent_analyzer'] == APNEA.SA_BL_POS:
            self.analyzer_obj = LexiconBased(None)
        else:
            self.analyzer_obj = LexiconBased()

        self.tokenizer = DocumentTokenizer(level=self.params['analysis_level'], n_gram=self.params['n_gram'])

        # Checking and loading metadata.
        try:
            md5 = open(self.MD5_FILE, 'r')
            md5 = md5.read()
            if not self.compare_md5(md5):
                raise AssertionError("MD5s don't match")
            else:
                self.load_metadata_from_file()
        except (IOError, AssertionError):
            self.pre_process_ads()
            self.save_metadata()

        # Loading pre-fetched articles.
        with open('articles.db', 'rb') as fp:
            self.articles = pickle.load(fp)

    @staticmethod
    def invert_dictionary(dict_):
        """
        Function to invert a mapping
        :param dict_: Input Mapping as a Python Dictionary
        :return: Inverted Python dictionary.
        """
        inverted_dict = {}
        for key, values in dict_.iteritems():
            for value in values:
                if value in inverted_dict:
                    inverted_dict[value].add(key)
                else:
                    inverted_dict[value] = {key}
        return inverted_dict

    def compare_md5(self, md5):
        """
        Compares the MD5 of existing file against computed MD5
        :param md5: Existing MD5
        :return: Boolean
        """
        # Hash both the ads data and the parameter `expand`
        hash_data = [self.params['expand'], self.ads, self.params['reduction_factor']]
        ad_md5 = compute_md5(hash_data)
        return ad_md5 == md5

    def load_metadata_from_file(self):
        """
        Loads ad-metadata from file
        :return: None
        """
        with open(self.META_FILE, 'r') as meta_file:
            self.keyword_to_company = pickle.load(meta_file)
            self.all_companies = pickle.load(meta_file)
            self.ad_keywords = pickle.load(meta_file)
            self.ad_negatives = pickle.load(meta_file)

        self.all_companies_scalar = self.all_companies.copy()
        for c in self.all_companies_scalar:
            self.all_companies_scalar[c] = 0

    def save_metadata(self):
        """
        Saves ad-metadata to file
        :return: None
        """
        # Save the data structures
        with open(self.META_FILE, 'w') as meta_file:
            pickle.dump(self.keyword_to_company, meta_file)
            pickle.dump(self.all_companies, meta_file)
            pickle.dump(self.ad_keywords, meta_file)
            pickle.dump(self.ad_negatives, meta_file)

        # Update the MD5 hash.
        hash_data = [self.params['expand'], self.ads]
        new_md5 = compute_md5(hash_data)

        with open(self.MD5_FILE, 'w') as md5_file:
            md5_file.write(new_md5)

        self.all_companies_scalar = self.all_companies.copy()
        for c in self.all_companies_scalar:
            self.all_companies_scalar[c] = 0

    def remove_stop_words(self, tokens):
        """
        Simple pre-processor for advertisements.
        :param tokens: Input tokens
        :return: Pre-processed tokens
        """
        # Filter tokens of length 1
        tokens = filter(lambda t: len(t) > 1, tokens)

        # Filter tokens that are digits or stopwords
        tokens = filter(lambda t: (not t.isdigit()) and (t.lower() not in self.stopwords), tokens)

        return tokens

    def expand_term(self, term):
        """
        Wiki Term expansion
        :param term: Term/Phrase to expand
        :return: Set of expanded terms.
        """
        wiki_terms = set()
        text = get_wiki_text(term)

        tokens = nltk.word_tokenize(text)
        tokens = self.remove_stop_words(tokens)

        # Consider only the top 15 terms as candidates.
        counts = Counter(tokens)
        top_15 = counts.most_common(15)
        count = 0

        for word, freq in top_15:
            tag_list = pos_tag([word])
            word, tag = tag_list[0]

            # Consider only Nouns
            if tag.startswith('N'):
                word = self.analyzer_obj.lemmatize(str(word))
                wiki_terms.add(word)
                count += 1
                if count == 5:
                    # Max of 5 terms to be included.
                    break

        return wiki_terms

    @staticmethod
    def is_phrase(phrases):
        # Determines if list of tokens is a phrase.
        return len(phrases) > 1

    def add_keyword(self, key_phrase, key_neg_sent, key_weight, company):
        """
        Adds keyword to keyword_to_company, ad_keywords, and ad_negatives
        :param key_weight: Weight associated with this keyword.
        :param key_phrase: keyword to add to database
        :param key_neg_sent: if neg_sent applies to this keyword
        :param company: company bidding on key
        :return: None
        """
        key_phrase = self.analyzer_obj.lemmatize(str(key_phrase))
        key_phrase = key_phrase.lower()

        keywords = key_phrase.split()
        if APNEA.is_phrase(keywords):
            keywords = self.remove_stop_words(keywords)

        for keyword in keywords:
            # Update keyword_to_company
            try:
                self.keyword_to_company[keyword].add(company)
            except KeyError:
                self.keyword_to_company[keyword] = {company}

            # Update ad_keywords (ad_vectors)
            if company in self.ad_keywords:
                APNEA.add_to_dict(self.ad_keywords[company], keyword, key_weight)
            else:
                self.ad_keywords[company] = {}
                APNEA.add_to_dict(self.ad_keywords[company], keyword, key_weight)

            # Update ad_negatives
            if company in self.ad_negatives:
                if keyword not in self.ad_negatives[company]:
                    self.ad_negatives[company][keyword] = key_neg_sent
            else:
                self.ad_negatives[company] = {}
                self.ad_negatives[company][keyword] = key_neg_sent

    def pre_process_ads(self):
        """
        Function to Pre-process ads, and term expand if necessary.
        :return: None
        """

        ads = self.ads
        expand_terms = self.params['expand']

        companies = ads.keys()

        for company in companies:
            keywords = ads[company]

            self.all_companies[company.lower()] = np.array([0, 0])

            # Assumption: Company is careful about sentiment directly affecting its brand.
            self.add_keyword(company, False, 1.0, company.lower())

            keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

            self.logger.debug(keywords)
            for key in keywords:
                key_phrase = key[0]
                key_neg_sent = key[1]
                key_weight = key[2]

                self.add_keyword(key_phrase, key_neg_sent, key_weight, company.lower())

                if expand_terms:
                    expanded_terms = self.expand_term(key_phrase)

                    # Expanded terms have less weight.
                    for term in expanded_terms:
                        self.add_keyword(term, key_neg_sent,
                                         key_weight/self.params['reduction_factor'], company.lower())

    def score_sentiment(self, sentiment, method):
        """
        Sentiment Scorers
        :param sentiment: Input Sentiment Vector
        :param method: Function to use
        :return: SentimentScore, float.
        """

        self.logger.debug(sentiment)
        x = sentiment[0]    # Pos Score
        y = sentiment[1]    # Neg Score
        if method == APNEA.SC_RATIO_LOG:
            # Ratio - Logs
            if y == 0:
                v = max(x, self.params['epsilon'])
            else:
                v = x/y * (log(abs(x - y) + 1)) * np.sign(x-y)
        elif method == APNEA.SC_RATIO:
            # Ratio
            # Ratio of pos:neg
            if y == 0:
                v = x
            else:
                v = x/y
        elif method == APNEA.SC_SENT_MAX:
            v = np.sign(x - y)*max(x, y)
        elif method == APNEA.SC_SUB:
            v = x - y
        elif method == APNEA.SC_SENT_TM:
            v = np.sign(x - y)*(1 + abs(max(x, y)))
        else:
            # Default to SENT_TD.
            v = np.sign(x - y)*(1 + abs(x - y))
        return v

    @staticmethod
    def add_to_dict(dict_, key, value):
        """
        Adds key, value to dictionary, creates if key doesn't exist.
        :param dict_: Dictionary
        :param key: Key
        :param value: Value to insert
        :return: None
        """
        try:
            dict_[key] = dict_[key] + value
        except KeyError:
            dict_[key] = value

    @staticmethod
    def cosine(ad_vec, doc_vec):
        """
        Function to compute cosine similarity between 2 vectors, formatted as dictionaries.
        :param ad_vec: Ad Vector
        :param doc_vec: Doc Vector
        :return: Cosine Similarity Score, float.
        """
        num = 0
        den_ad = 0
        den_doc = 0

        for k in ad_vec:
            if k in doc_vec:
                num += ad_vec[k] * doc_vec[k]
            den_ad += ad_vec[k] ** 2

        for k in doc_vec:
            den_doc += doc_vec[k] ** 2
        return num / (np.sqrt(den_ad) * np.sqrt(den_doc))

    def get_target_company(self, title):
        """
        Extracts target companies from title
        :param title: Input title
        :return: Target Companies, set.
        """
        candidates = set()
        stripper = re.compile('([^&\-a-zA-Z0-9 ])')
        title = stripper.sub(r'', title).lower().split()

        for company in self.all_companies.keys():
            if company in title:
                candidates.add(company)

        return candidates

    def page_ad_matching(self, url):
        """
        Function that matches URL to existing advertisers by relevance scores.
        :param url: Input URL
        :return: Advertisers, sorted by relevance scores.
        """
        self.logger.debug('Keyword_2_company: ' + str(self.keyword_to_company))
        self.logger.debug('ADVECS: ' + str(self.ad_keywords))

        if url in self.articles:
            cleaned = self.articles[url]
        else:
            cleaned = clean_url(url)

        text = cleaned['title'] + '. ' + cleaned['text']

        # if self.params['use_sentiment']:
        #     candidates = self.all_companies.copy()
        # else:
        #     candidates = self.all_companies_scalar.copy()

        # triggers = {}

        doc_vec = {}
        bid_companies = set()
        blacklisted_companies = set()

        scores = self.all_companies_scalar.copy()

        # Fetching target companies.
        targets = self.get_target_company(cleaned['title'])
        self.logger.debug('-----------' + str(targets))
        self.logger.debug('-----------' + cleaned['title'])

        # Reverting back to regular analysis if no targets are found.
        if len(targets) == 0:
            self.params['targeted_sent'] = False

        first_chunk = True
        sentiment = np.array([0, 0])

        for chunk in self.tokenizer.get_chunk(text):
            # Sentiment Extraction from Chunk
            if self.params['use_sentiment']:
                sentiment, tokens = self.analyzer_obj.get_sentiment_from_text(chunk)
                if first_chunk:
                    # Fixed attention weight for the first chunk.
                    sentiment = 2*sentiment
                self.logger.debug('Sentiment extracted: ' + str(sentiment))
                self.logger.debug(chunk)
            else:
                tokens = self.analyzer_obj.preprocess(chunk, join_negatives=False)

            for idx, token in enumerate(tokens):
                # Bidding Companies
                try:
                    companies = self.keyword_to_company[token]
                    self.logger.debug('Found trigger: ' + token)
                    if self.params['use_sentiment']:
                        if not np.any(sentiment):
                            # Neutral Sentiment
                            sentiment += np.array([0.5, 0])
                        APNEA.add_to_dict(doc_vec, token, sentiment)
                    else:
                        APNEA.add_to_dict(doc_vec, token, 1)

                    bid_companies |= companies
                except KeyError:
                    pass

                try:
                    # Blacklist companies.
                    companies = self.blacklist_to_company[token]
                    blacklisted_companies |= companies
                except KeyError:
                    try:
                        # Handling 2 worded blacklist phrases.
                        if idx + 1 < len(tokens):
                            companies = self.blacklist_to_company[token + ' ' + tokens[idx+1]]
                            blacklisted_companies |= companies
                    except KeyError:
                        pass

            if first_chunk:
                first_chunk = False

        # Calculating the Relevance Scores, based on the system configuration.
        for c in bid_companies:
            ad_vec = self.ad_keywords[c]
            doc_vec_copy = doc_vec.copy()
            if self.params['use_sentiment']:
                for k in doc_vec_copy:
                    if self.params['targeted_sent']:
                        if c in targets:
                            if self.params['neg_sent'] and k in ad_vec and self.ad_negatives[c][k]:
                                # Targeted but Sentiment Insensitive.
                                doc_vec_copy[k] = abs(self.score_sentiment(doc_vec_copy[k],
                                                                           method=self.params['scorer']))
                            else:
                                # Targeted and Sentiment Sensitve
                                doc_vec_copy[k] = self.score_sentiment(doc_vec_copy[k], method=self.params['scorer'])
                        else:
                            # Non-targets are taken as absolute values
                            doc_vec_copy[k] = abs(self.score_sentiment(doc_vec_copy[k], method=self.params['scorer']))
                    else:
                        if self.params['neg_sent'] and k in ad_vec and self.ad_negatives[c][k]:
                            # Sentiment Insensitive
                            doc_vec_copy[k] = abs(self.score_sentiment(doc_vec_copy[k], method=self.params['scorer']))
                        else:
                            # Sentiment Sensitve
                            doc_vec_copy[k] = self.score_sentiment(doc_vec_copy[k], method=self.params['scorer'])

            self.logger.debug('Company========>: ' + str(c))
            self.logger.debug('ad_vec: ' + str(ad_vec))
            self.logger.debug('doc_vec: ' + str(doc_vec))
            self.logger.debug('doc_vec_copy: ' + str(doc_vec_copy))

            scores[c] = self.cosine(ad_vec, doc_vec_copy)

        # Handling blacklist companies.
        if self.params['blacklist']:
            for blacklisted_company in blacklisted_companies:
                scores[blacklisted_company] = -(abs(scores[blacklisted_company]))

        self.logger.debug('scores: ' + str(scores))

        # Scoring and Sorting.
        scored = sorted(scores.iteritems(),
                        key=lambda x: x[1],
                        reverse=True)
        return scored


def trim_top5(companies):
    return str(companies[:5]) + ' ... ' + str(companies[-5:])


def main():
    url = 'https://www.cbsnews.com/news/obesity-rates-now-top-35-percent-in-7-states/'

    with open('t_advertisers.json', 'r') as ad_file:
        ads = json.load(ad_file)

    with open('blacklist.json', 'r') as bl_fp:
        blacklist = json.load(bl_fp)

    log_format = "%(asctime)-15s:%(levelname)s:%(name)s::%(message)s"
    logging.basicConfig(format=log_format, level=logging.CRITICAL)

    logging.debug(ads)

    bushi_td = APNEA(ads, blacklist, params={'expand': True,
                                                'use_sentiment': True,
                                                'sent_analyzer': APNEA.SA_OPINION_MINER,
                                                'scorer': APNEA.SC_SENT_TD,
                                                'use_vector': True,
                                                'neg_sent': True,
                                                'targeted_sent': False,
                                                'analysis_level': DocumentTokenizer.SENTENCE,
                                                'n_gram': 3,
                                                'blacklist': True,
                                                'reduction_factor': 2.0})

    print url
    print trim_top5(bushi_td.page_ad_matching(url))


if __name__ == '__main__':
    main()
