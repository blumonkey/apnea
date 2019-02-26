from abc import ABCMeta, abstractmethod
import pickle
import nltk
import numpy as np
import re
import logging

import unidecode
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP
from utils import get_wordnet_pos
from nltk.wsd import lesk


class SentimentAnalyzer:
    """Base class for all sentiment analyzer algorithms, SentimentAnalyzer"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_sentiment_from_text(self, text):
        pass

    @abstractmethod
    def get_analyzer_type(self):
        pass

    def lemmatize(self, text):
        pass


class StanfordSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        self.snlp = StanfordCoreNLP('http://localhost:9000')

    def lemmatize(self, text):
        lemmatized = self.snlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json'
        })

        sentence = lemmatized['sentences'][0]
        tokens = [x['lemma'] for x in sentence['tokens']]
        return ' '.join(tokens)

    def get_analyzer_type(self):
        return 'snlp'

    def get_sentiment_from_text(self, text):
        nearest_ascii = unidecode.unidecode(text)
        nearest_ascii = ' '.join(nearest_ascii.split())
        resp = self.snlp.annotate(nearest_ascii, properties={
                    'timeout': '50000',
                    'annotators': 'tokenize, ssplit, pos, lemma, sentiment',
                    'outputFormat': 'json'
             })

        sentiment = np.zeros(2)
        tokens = []
        # logging.debug(resp)
        for sentence in resp['sentences']:
            i = int(sentence['sentimentValue'])
            sentiment += np.array([max(i - 2, 0), max(2 - i, 0)])
            tokens += [x['lemma'] for x in sentence['tokens']]

        return sentiment, tokens


class SentimentIdentifierSVM:
    def __init__(self, path_to_model='resources/svm_si_bal.pkl', path_to_w2v='resources/glove.pkl'):
        self.stop_edit = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
                          u'yours',
                          u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
                          u'herself',
                          u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
                          u'which',
                          u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
                          u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did',
                          u'doing',
                          u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while',
                          u'of',
                          u'at', u'by', u'for', u'with', u'about', u'between', u'into', u'through', u'during',
                          u'before',
                          u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off',
                          u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when',
                          u'where',
                          u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some',
                          u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u'can', u'will', u'just',
                          u'should', u'now']
        with open(path_to_model, 'rb') as fp:
            self.clf = pickle.load(fp)

        with open(path_to_w2v, 'rb') as fp:
            self.word2vec = pickle.load(fp)

        self.snlp = StanfordCoreNLP('http://localhost:9000')

    def lemmatize(self, text):
        lemmatized = self.snlp.annotate(str(text), properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json'
        })

        sentence = lemmatized['sentences'][0]
        tokens = [x['lemma'] for x in sentence['tokens']]
        return ' '.join(tokens)

    def preprocessor(self, text):
        """
        Function that preprocesses the input string, based on some heuristics.
        """
        regex = re.compile('\.{2,}')
        regex2 = re.compile(r'(\w+)\.(\w+)')

        text = text.encode('ascii', 'ignore')

        sentences = sent_tokenize(text)
        total = []
        for s in sentences:
            txt = regex2.sub(r'\1. \2', s)  # Add space after fullstops.
            tokens = word_tokenize(txt)
            tokens = [s.lower() for s in tokens]  # Convert to lowercase
            tokens = [regex.sub('', x) for x in tokens]  # Remove elipses
            tokens = [x for x in tokens if x not in '.,;!?']  # Remove punctuations
            tokens = [x for x in tokens if len(x) != 0]  # Remove empty tokens
            tokens = [x if x != "n't" else "not" for x in tokens]  # Replace n't, 's and 'd with not, is and would
            tokens = [x if x != "'s" else "is" for x in tokens]
            tokens = [x if x != "'d" else "would" for x in tokens]
            tokens = map(self.lemmatize, tokens)  # Lemmatize the words
            tokens = [x for x in tokens if x not in self.stop_edit]  # Remove stop-words
            total = total + tokens
            total.append("<TERM>")  # Terminate sentence
        return total

    def list2vec(self, tokens):
        out = np.zeros(50)
        for t in tokens:
            try:
                out += self.word2vec[t]
            except KeyError:
                pass
        return out

    def preprocessor_w2v(self, inp):
        tokens = self.preprocessor(inp)
        result = self.list2vec(tokens)

        return result, tokens

    def get_objectivity_from_text(self, text):
        text_w2v, tokens = self.preprocessor_w2v(text)
        prediction = self.clf.predict(text_w2v.reshape(1, -1))[0]
        # print prediction
        prediction = int(prediction)

        result = np.zeros(2)
        result[1 - prediction] = 1
        return result

    @staticmethod
    def get_classifier_type():
        return 'svm_si'


class SentimentAnalyzerSVM(SentimentAnalyzer):
    def __init__(self, path_to_model='resources/svm.pkl', path_to_w2v='resources/glove.pkl'):
        self.stop_edit = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
                          u'yours',
                          u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
                          u'herself',
                          u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
                          u'which',
                          u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
                          u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did',
                          u'doing',
                          u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while',
                          u'of',
                          u'at', u'by', u'for', u'with', u'about', u'between', u'into', u'through', u'during',
                          u'before',
                          u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off',
                          u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when',
                          u'where',
                          u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some',
                          u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u'can', u'will', u'just',
                          u'should', u'now']
        with open(path_to_model, 'rb') as fp:
            self.clf = pickle.load(fp)

        with open(path_to_w2v, 'rb') as fp:
            self.glove = pickle.load(fp)

        self.snlp = StanfordCoreNLP('http://localhost:9000')

    def lemmatize(self, text):
        lemmatized = self.snlp.annotate(str(text), properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json'
        })

        sentence = lemmatized['sentences'][0]
        tokens = [x['lemma'] for x in sentence['tokens']]
        return ' '.join(tokens)

    def preprocessor(self, text):
        """
        Function that preprocesses the input string, based on some heuristics.
        """
        regex = re.compile('\.{2,}')
        regex2 = re.compile(r'(\w+)\.(\w+)')

        text = text.encode('ascii', 'ignore')

        sentences = sent_tokenize(text)
        total = []
        for s in sentences:
            txt = regex2.sub(r'\1. \2', s)  # Add space after fullstops.
            tokens = word_tokenize(txt)
            tokens = [s.lower() for s in tokens]  # Convert to lowercase
            # tokens = [unicode(x, errors='ignore') for x in tokens]  # Convert to unicode
            tokens = [regex.sub('', x) for x in tokens]  # Remove elipses
            tokens = [x for x in tokens if x not in '.,;!?']  # Remove punctuations
            tokens = [x for x in tokens if len(x) != 0]  # Remove empty tokens
            tokens = [x if x != "n't" else "not" for x in tokens]  # Replace n't, 's and 'd with not, is and would
            tokens = [x if x != "'s" else "is" for x in tokens]
            tokens = [x if x != "'d" else "would" for x in tokens]
            tokens = map(self.lemmatize, tokens)  # Lemmatize the words
            tokens = [x for x in tokens if x not in self.stop_edit]  # Remove stop-words
            total = total + tokens
            total.append("<TERM>")  # Terminate sentence
        return total

    def list2vec(self, tokens):
        out = np.zeros(50)
        for t in tokens:
            try:
                out += self.glove[t]
            except KeyError:
                pass
        return out

    def preprocessor_w2v(self, inp):
        tokens = self.preprocessor(inp)
        result = self.list2vec(tokens)

        return result, tokens

    def get_sentiment_from_text(self, text):
        text_w2v, tokens = self.preprocessor_w2v(text)
        prediction = self.clf.predict(text_w2v.reshape(1, -1))
        result = np.zeros(2)
        result[1 - prediction] = 1
        return result, tokens

    def get_analyzer_type(self):
        return 'svm'


class LexiconBased(SentimentAnalyzer):
    """Using the OpinionMiner from Bing Liu to extract Sentiment"""

    def __init__(self, path_to_lexicon='resources/opinion_miner.pkl'):
        if path_to_lexicon is not None:
            with open(path_to_lexicon, 'rb') as f:
                self.lexicon = pickle.load(f)
            if 'news' in path_to_lexicon:
                self.a_type = 'reddit_news'
            elif 'miner' in path_to_lexicon:
                self.a_type = 'opinion_miner'
            else:
                self.a_type = 'sentiwordnet'
        else:
            self.a_type = 'baseline_pos'

        self.stop_words = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
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
                           u'now']

        self.negatives = ['never', 'not', 'nothing', 'no', 'neither', 'nor']
        self.snlp = StanfordCoreNLP('http://localhost:9000')
        self.logger = logging.getLogger('LexiconBased')

    def get_analyzer_type(self):
        return self.a_type

    def get_lexicon_shape(self):
        return self.lexicon.values()[0].shape

    def lemmatize(self, text):
        lemmatized = self.snlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json'
        })

        sentence = lemmatized['sentences'][0]
        tokens = [x['lemma'] for x in sentence['tokens']]
        return ' '.join(tokens)

    def preprocess(self, text, join_negatives=True):
        ellipses = re.compile('\.{2,}')
        periods = re.compile(r'(\w+)\.(\w+)')

        # Add space after fullstops.
        text = periods.sub(r'\1. \2', text)
        text = ellipses.sub('.', text)

        text = text.encode('ascii', 'ignore')

        lemmatized = self.snlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit, pos, lemma',
            'outputFormat': 'json'
        })

        sentences = lemmatized['sentences']

        result = []
        for sentence in sentences:
            tokens = [x['lemma'] for x in sentence['tokens']]
            # Convert to lowercase
            tokens = [s.lower() for s in tokens]
            # Remove punctuations
            tokens = [x for x in tokens if x not in '.,;!?']
            # Remove empty tokens
            tokens = [x for x in tokens if len(x) != 0]
            # Replace n't, 's and 'd with not, is and would
            tokens = [x if x != "n't" else "not" for x in tokens]
            tokens = [x if x != "'s" else "is" for x in tokens]
            tokens = [x if x != "'d" else "would" for x in tokens]
            # Remove stop-words
            tokens = [x for x in tokens if x not in self.stop_words]

            if not join_negatives:
                return tokens

            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token in self.negatives:
                    if i < len(tokens) - 1:
                        result.append(token + '_' + tokens[i + 1])
                        i += 1
                    else:
                        logging.debug('Negative at end of text' + str(tokens))
                        result.append(token)
                else:
                    result.append(token)
                i += 1

        logging.debug(result)
        return result

    def get_sentiment_from_lexicon(self, token):
        try:
            return self.lexicon[token]
        except KeyError as e1:
            if '_' in token:
                parts = token.split('_')
                try:
                    return 1-self.lexicon[parts[1]]
                except KeyError:
                    # Returning a negative for the modifier.
                    return np.array([0., 1.])
            else:
                raise e1

    def get_sentiment_linear_product(self, tokens):
        # Specifically for FanEtAl
        total_sentiment = 1
        for token in tokens:
            try:
                current = -1 if self.get_sentiment_from_lexicon(token)[1] > 0.5 else 1
                logging.getLogger().debug('CURRENT: ' + token + '::' + str(current))
                total_sentiment *= current
            except KeyError:
                pass
        return total_sentiment

    def get_sentiment_from_text_traditional(self, text, pre_process=True):
        """
        Function that calculates the sentiment score as per the opinion miner module,from text.
        """
        if pre_process:
            tokens = self.preprocess(text)
        else:
            tokens = nltk.word_tokenize(text)
        total_sentiment = np.zeros(self.get_lexicon_shape())
        split_tokens = []

        for token in tokens:
            try:
                total_sentiment += self.get_sentiment_from_lexicon(token)
                logging.getLogger().debug('CURRENT: ' + token + '::' + str(self.get_sentiment_from_lexicon(token)))
            except KeyError:
                pass

            if '_' in token:
                split_tokens += token.split('_')
            else:
                split_tokens.append(token)

        return total_sentiment, split_tokens

    def get_sentiment_from_text_sentiwordnet(self, text):
        tokens = nltk.word_tokenize(text)
        tokens_tagged = nltk.pos_tag(tokens)
        total_sentiment = np.zeros(self.get_lexicon_shape())

        for tagged_pair in tokens_tagged:
            word, pos = tagged_pair[0], get_wordnet_pos(tagged_pair[1])
            if pos != '':
                synset = lesk(tokens, word, pos)
                if synset is not None:
                    try:
                        total_sentiment += self.get_sentiment_from_lexicon(synset.name())
                    except KeyError:
                        pass

        self.logger.debug('Text: ' + text)
        tokens = self.preprocess(text, join_negatives=False)
        self.logger.debug('Tokens: ' + str(tokens))

        return total_sentiment[:2], tokens

    def get_sentiment_from_text(self, text):
        if self.a_type == 'sentiwordnet':
            return self.get_sentiment_from_text_sentiwordnet(text)
        elif self.a_type == 'baseline_pos':
            return np.array([1., 0]), self.preprocess(text, join_negatives=False)
        else:
            return self.get_sentiment_from_text_traditional(text)


class DocumentTokenizer:
    """docstring for DocumentTokenizer"""
    SENTENCE = 'SENTENCE'
    SENT_NGRAM = 'SENT_NGRAM'
    DOCUMENT = 'DOCUMENT'

    def __init__(self, level, n_gram=3):
        self.level = level
        self.n_gram = n_gram

    def get_chunk(self, text):
        if self.level == DocumentTokenizer.DOCUMENT:
            for document in [text]:
                yield document
        if self.level == DocumentTokenizer.SENTENCE:
            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                yield sentence
        elif self.level == DocumentTokenizer.SENT_NGRAM:
            sentences = nltk.sent_tokenize(text)
            n_grams = nltk.ngrams(sentences, self.n_gram)
            for n_gram in n_grams:
                text_ = ' '.join(n_gram)
                yield text_

    def set_level(self, level, n_gram=3):
        self.level = level
        self.n_gram = n_gram
