# -*- coding: utf-8 -*-
#Authors: vlad@fiscalnote
from operator import itemgetter
from math import log, sqrt
import os, sys
import logging
import json
import math
import re
import string
import numpy as np

logger = logging.getLogger('module')

class TextPreprocessor(object):
    """
    Preprocess text document for downstream processing.
    Parameters
    ----------
    stop: boolean, optional
        remove stop words
    stem: boolean, optional
        apply snowball stemmer
    punctuation: boolean, optional
        split and keep punctuation
    token_pattern: string, optional
        how to split tokens
    state: string, optional
        adds state specific stop words
    encoding: string, optional
        how to decode a byte string
    custom_stop_list: list, optional
        list of additional stop words
    min_length: int, optional
        minimum character length of tokens to keep
    lower: boolean, optional
        lowercase entire string
    brackets: boolean, optional
        remove entirety of text within brackets [] and parenthesis ()
    """

    def apply_pattern_split(self, doc):
            return self._token_pattern.findall(doc)

    def apply_split(self, doc):
            return doc.split()

    def _isnotin(self, x):
            return x not in self.not_in and len(x) > self.min_length

    def _islength(self, x):
            return len(x) > self.min_length

    def _load_patterns(self):

        self._num_pattern = re.compile(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
        self._dollar_pattern = re.compile(r'\$\d+(?:[\,\d]+)(?:\.\d+)?(?:[eE][+-]?\d+)?')

        #match dates in numeric form (03/30/2015) or free form (March 30, 2015), accounting for legitimate date logic (number of days in month)
        self._date_pattern = re.compile(r'(?:(September|April|June|November|Sept.|Nov.) +(0?[1-9]|[12]\d|30), *((?:19|20)\d\d))|(?:(January|March|May|July|August|October|December|Jan.|Aug.|Oct.|Dec.) +(0?[1-9]|[12]\d|3[01]), *((?:19|20)\d\d))|(?:(February|Feb.) +(?:(?:(0?[1-9]|1\d|2[0-8]), *((?:19|20)\d\d))|(?:(29), *((?:(?:19|20)(?:04|08|12|16|20|24|28|32|36|40|44|48|52|56|60|64|68|72|76|80|84|88|92|96))|2000))))', re.I)
        self._date_pattern2 = re.compile(r'(\d+)[/.-](\d+)[/.-](\d+)')

        #self._punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
        #string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

        self._punc_pattern = re.compile(r'([%s])' % re.escape(string.punctuation))

        #restrict to punctuation that's going to be removed even when keeping some punctuation to everything that's not sentence/clause breaking
        self._limited_punc_pattern = re.compile(r'([%s])' % re.escape('"#$%&\'()*+/<=>@[\]^`{|}~'), re.UNICODE)

        #going to take remaining punctuation and split off except hyphens
        permitted_hyphens = ''.join([u'_', u'-', u'\u2012', u'\u2011', u'\u2012', u'\u2013', u'\u2014'])
        hyphen_mapping = dict.fromkeys(map(ord, permitted_hyphens))

        punc_hyphen_remove = string.punctuation.translate(hyphen_mapping)

        self._split_limited_punc_pattern = re.compile(r'([%s])' % re.escape(punc_hyphen_remove), re.UNICODE)


        self._section_pattern = re.compile(r"(?u)\b([ivx]+)\b", re.I)

        self._alpha_pattern = re.compile(r'[a-zA-Z]+', re.UNICODE)

        #punctuation patterns
        self._slash_pattern = re.compile(r'[\\/]', re.UNICODE)
        self._hyphen_pattern = re.compile(r'[-_]', re.UNICODE)
        self._breaking_hyphen_pattern = re.compile(r'([a-zA-Z])-\s+([a-zA-Z])', re.UNICODE)

        self._period_pattern = re.compile(r'[\.]', re.UNICODE)
        self._bracket_pattern = re.compile(r'\[[\w\.,\- ]+\]', re.UNICODE)
        self._paren_pattern = re.compile(r'\([\w\.,\- ]+\)', re.UNICODE)

        #javascript junk starting with $(
        self._func1 = re.compile(r'\$\(\S+')
        #http addresses
        self._func2 = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


        self._non_breaking_prefixes= [" A"," B"," C"," D"," E"," F"," G"," H"," I"," J"," K"," L"," M"," N"," O"," P"," Q"," R"," S"," T"," U"," V"," W"," X"," Y"," Z","Adj","Adm","Adv","Asst","Bart","Bldg","Brig","Bros","Capt","Cmdr","Col","Comdr","Con","Corp","Cpl","DR","Dr","Drs","Ens","Gen","Gov","Hon","Hr","Hosp","Insp","Lt","MM","MR","MRS","MS","Maj","Messrs","Mlle","Mme","Mr","Mrs","Ms","Msgr","Op","Ord","Pfc","Ph","Prof","Pvt","Rep","Reps","Res","Rev","Rt","Sen","Sens","Sfc","Sgt","Sr","St","Supt","Surg","vs","i.e" ]
        self._non_breaking_prefixes_pattern = re.compile(r'(%s) \.' % '|'.join(self._non_breaking_prefixes), re.UNICODE)

        #set splitting pattern
        if self._token_pattern:
            self._token_pattern = re.compile(self._token_pattern)
            self.tok = self.apply_pattern_split #lambda doc: self._token_pattern.findall(doc)
        else:

            self.tok = self.apply_split #lambda doc: doc.split()

    def __init__(self,
                stop=True,
                stem=True,
                punctuation=False,
                token_pattern=r"(?u)\b\w\w+\b",
                state=None,
                encoding='utf-8',
                decode_error='ignore',
                custom_stop_list=None,
                min_length=2,
                lower=True,
                brackets=True):

        self._eng_stem = None #Stemmer.Stemmer('en')
        #apply stemming
        self.stem = stem
        self._token_pattern = token_pattern
        #filter stop words
        self.stop = stop
        self.custom_stop_list = custom_stop_list
        self._initialize_stop(state=state)

        self.min_length = min_length
        self.punctuation = punctuation
        self.lower = lower
        self. brackets = brackets
        #grab the regex patterns used for subbing
        self._load_patterns()

        #set string encoding
        self.encoding = encoding
        self.decode_error = decode_error

    def _initialize_stop(self, state=None):
        '''
        generic stop list + legislative stop list + state specific logic (incomplete)
        '''

        self.not_in=[]
        self._stop_words = ["a","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","all","allow","allows","almost","alone","along","already","also","although",
        "always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","are","around","as","aside","ask","asking","associated",
        "at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both",
        "brief","but","by","c","came","can","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains",
        "corresponding","could","course","currently","d","definitely","described","despite","did","different","do","does","doing","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough",
        "entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former",
        "formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","happens","hardly","has","have","having","he","hello","help",
        "hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate",
        "indicated","indicates","inner","insofar","instead","into","inward","is","it","its","itself","j","just","k","keep","keeps","kept","know","knows","known","l","last","lately","later","latter","latterly","least","less","lest","let",
        "like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd",
        "near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh",
        "ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please",
        "plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying",
        "says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","since","six","so","some","somebody","somehow",
        "someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","take","taken","tell","tends","th","than","thank","thanks","thanx",
        "that","thats","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","theres","thereupon","thereon", "thereof", "therein","these","they","think","third","this","thorough",
        "thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely",
        "until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","way","we","well","went","were","what","whatever","when","whence","whenever",
        "where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","whose","why","will","willing","with","within","without","wonder","would","would","x","y",
        "yes","yet","you","your","yours","yourself","yourselves","z","zero"]
        self._bill_stop_words = ["bill","hb","version","rsa","paragraph","amend", "amends","deleted","insert","inserted","delete", "reference", "referred", "committee", "session", "autho",
        "person", "date", "time","chapter", "make", "section", "sections", "subparagraph", "subsections", "subsection", "title", "article","articles","verdate","jkt","frm","fmt","sfmt","po","sbpn" ,"italics","underscored", "gentleman","gentlewoman"]

        self.not_in+=(self._stop_words)
        self.not_in+=(self._bill_stop_words)

        #add custom stop logic id any
        if self.custom_stop_list:
            self.not_in += (self.custom_stop_list)

        #any state specific trash removal
        if state == 'al':
            self.not_in += ['pfd', 'rfd', 'lcg', 'llr', 'lbh', 'sjr','lr', 'alabama', 'arizona' ]
        elif state == 'ar':
            self.not_in += ['arkansa', 'endoftext', 'allsponsor', 'billend', 'documenttitl', 'startbilltext', 'scheduleend', 'schedulestart', 'subtitl', 'leb']
        elif state == 'ak':
            self.not_in += ['hba', 'hjra', 'hjr', 'hra', 'underlin','pfd', 'hcr', 'hcra', 'hbb', 'cshcr', 'cshb', 'alaska']
        elif state == 'az':
            self.not_in += ['arizona', 'sectiona', 'tpt']
        elif state == 'fl':
            self.not_in += ['florida','code','stricken','delet','underlin','word']
        elif state == 'us':
            self.not_in += ['prefix','txt', 'vol', 'stuburl', 'dropdownnavig', 'loginemailaddress', 'searchwithinhint', 'cleartimeout', 'sql_nclob', 'amdt', 'pdf']

        self.not_in.append('|||')

        self.isn=self._isnotin #lambda x: x not in not_in and len(x) > 2

    def pretokenizer(self, tokens):
        '''
        removing newlines, tabs, and extra spaces.
        '''
        tokens = re.sub(' +',' ', re.sub('\t',' ', re.sub('\n',' ', tokens)))
        return tokens

    def tokenize(self, tokens):
        '''
        primary method for processing string
        '''
        
        #get rid of hanging whitespaces
        tokens = self.pretokenizer(tokens)

        #replace nums, punctuation
        tokens = self.replace(tokens, None)

        #break into list of tokens
        tokens = self.tok(tokens)

        #remove stopwords and less than 3 characters
        if self.stop:
            tokens = self.stopper(tokens)
        #else:
        #    tokens = filter(self._islength, tokens)
        #stem using snowball

        return tokens

    def replace(self, tokens, num_token='_NNVV_', money_token='_MMVV_', date_token='_DDVV_'):

        def _get_numeric_type(tokens):
            #get dollar amounts
            tokens = self._dollar_pattern.sub(money_token,tokens) if money_token else self._dollar_pattern.sub(' ',tokens)

            #get dates
            tokens = self._date_pattern.sub(date_token,tokens) if date_token else self._date_pattern.sub(' ',tokens)
            tokens = self._date_pattern2.sub(date_token,tokens) if date_token else self._date_pattern2.sub(' ',tokens)

            #finally, remove all remaining numbers, replace with <NUM> token or nothing
            #careful: when it hits bill numbers, like HB2002 the 2002 gets replaced with nothing and then hb gets wiped out
            tokens = self._num_pattern.sub(num_token,tokens) if num_token else self._num_pattern.sub(' ',tokens)


            return tokens

        #TODO: guess at taking out html/xml markup, other nasties
        tokens = self._func1.sub(' ', tokens)
        tokens = self._func2.sub(' ', tokens)

        if self.brackets:
            #delete anything within brackets (indicates strikethrough or other annotations)
            tokens = self._bracket_pattern.sub(' ', tokens)
            tokens = self._paren_pattern.sub(' ', tokens)

        #special processing for different numeric types
        tokens = _get_numeric_type(tokens)

        #handle hyphenated words
        #TODO: check dictionary before concat to make sure we're getting a real word
        tokens = self._breaking_hyphen_pattern.sub('\g<1>\g<2>', tokens)

        if self.punctuation:
            #remove certain punctuation anyway :P
            tokens = self._limited_punc_pattern.sub('', tokens)
            #keep and split remaining punctuation into separate token
            tokens = self._split_limited_punc_pattern.sub(' \g<1> ', tokens)
            #recombine all nonbreaking prefixes into single token with previous word
            tokens = self._non_breaking_prefixes_pattern.sub('\g<1>.', tokens)
        else:
            #remove punctuation
            tokens = self._punc_pattern.sub(' ', tokens)

        #take out section heading stuff ([ivx]) (need to figure out if it needs casing or not)
        tokens = self._section_pattern.sub(' ', tokens)

        #lowercase
        if self.lower:
            tokens = tokens.lower()

        return tokens

    def stopper(self, tokens):
        return filter(self.isn, tokens)

