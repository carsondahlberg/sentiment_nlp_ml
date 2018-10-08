#!/usr/bin/python
# -*- coding: utf-8 -*-

# ##############################################################

## helpers with NLP preprocessing and feature generation
## Carson Dahlberg, 2017/2018
## Python 3

## Many of these are inspired by the NLTK online book, and the 
## workshops I've attended by Andreas Mueller, PhD, Data Scientist,
## Scikit-Learn core developer and his recent book "Introduction
## to Machine Learning with Python", and of course stackoverflow!

# ##############################################################

## TODO list
# 1. NLTK cannot detect sentence boundaries under this circumstance
#    edge case: I liked the book he sent.The book was good.  
#    possible solution: https://github.com/lukeorland/splitta

# 2. split document into multiple sentences and do NER & POS


import numpy as np
import pandas as pd
import itertools
from time import time

import re, string, unicodedata

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# colormap for heatmap
col_map = ListedColormap(['#0000aa', '#ff2020'])

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# for cleaning html and getting metadata if provided, like tags
from bs4 import BeautifulSoup

# for NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet # for synonyms from WordNet
from nltk.stem import PorterStemmer # word stemming english words
from nltk.stem import SnowballStemmer # stemming non-English words
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer # another lemmatizer

# additional smaller but useful libraries 
# contractions is capable of resolving contractions (and slang)
# could use to convert slang or create additional text
import contractions # expanding contractions here https://pypi.org/project/contractions/
print("contractions version: {}".format('contractions 0.0.16'))

# inflect.py - Correctly generate plurals, singular nouns, ordinals, indefinite articles; convert numbers to words.
# https://pypi.org/project/inflect/ 
# In generating these inflections, inflect.py follows the Oxford English Dictionary and 
# the guidelines in Fowler’s Modern English Usage, preferring the former where the two disagree.
# The module is built around standard British spelling, but is designed to cope with common 
# American variants as well. Slang, jargon, and other English dialects are not explicitly catered for.
import inflect # generating plurals, singular nouns, ordinals, indefinite articles, and converting numbers to words
print("inflect version: {}".format('inflect 0.3.1'))

# ##############################################################

## Noise Removal Helper Functions

# ##############################################################

# file headers & footers; HTML, XML, markup and metadata;
# extracting valuable data from formats like JSON

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# ##############################################################

## Normalize text

# ##############################################################

def replace_contractions(text, slang=False):
    """replace contractions in string of text
    you're -> you are
    i'm    -> I am
    # uses \b boundaries for "unsafe"
    ima    -> I am going to
    yall  -> you all
    gotta  -> got to
    
    """
    return contractions.fix(text, slang=slang)


def remove_non_ascii(words):
    """remove non-ASCII chars from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """convert all chars to lowercase from tokenized word list"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

 
def remove_punctuation(words):
    """remove puncts from tokenized word list"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

 
def replace_numbers(words):
    """replace all integers with their textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

 
def remove_stopwords(words):
    """remove stop words from tokenized word list"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

 
def normalize_test(words):
    '''check the reduction of vocabulary after each step'''
    print('word count...')
    count_start = len(words)
    print('... before normalization {}'.format(len(words)))
    words = remove_non_ascii(words)
    print('... after removing non_ascii {}'.format(len(words)))
    words = to_lowercase(words)
    words = remove_punctuation(words)
    print('... after removing punctuation {}'.format(len(words)))
    words = replace_numbers(words)
    print('... after replacing numbers {}'.format(len(words)))
    words = remove_stopwords(words)
    count_end = len(words)
    print('... after removing stop words {}'.format(len(words)))
    print('word count reduced from {} to {}'.format(count_start, count_end))
    return words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


# ##############################################################

## check token/word frequencies

# ##############################################################

def token_freqs(tokens):
    # let's count token frequencies and look at the ranks
    token_freq = nltk.FreqDist(tokens)
    #for key, val in freq.items():
    #    print(str(key) + ":" + str(val))
    print('-'*55)
    print('TOKENS: 10 most frequent:')
    print(pd.Series(dict(token_freq), name='freq_count').sort_values(ascending=False)[:10])
    print('-'*55)
    print('TOKENS: 10 least frequent:')
    print(pd.Series(dict(token_freq), name='freq_count').sort_values(ascending=False)[-10:])
    #freq.plot(20, cumulative=False)

# ##############################################################

## calling the stemming and lemming functions
#  how to look inside of wordnet lemmatizer https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
# ##############################################################

def stem_words(words):
    """stem words in tokenized word list"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

 
def lemmatize_verbs(words):
    """lemmatize verbs in tokenized word list"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

# stems, lemmas = stem_and_lemmatize(words)


# ##############################################################

## fine-grained selection of words

# ##############################################################

def get_brigrams(text):
    """#get_brigrams(["try","this",'fashizzle'])
    [('try', 'this'), ('this', 'fashizzle')]
    """
    from nltk.util import bigrams
    return list(bigrams(text))

def vocab_size(sentence):
    return len(set(w.lower() for w in sentence))

def char_size(sentence):
    return sum(len(word) for word in sentence)

def lexical_diversity(text):
    return len(text) / len(set(text))

def word_to_char_frac(text):
    return char_size(text) / len(text)

def unusual_words(text):
    '''find unusual and mispelled words using 
    Word Corpus from /user/share/dict/words Unix file use by spellcheckers
    #unusual_words(nltk.corpus.nps_chat.words())
    #unusual_words(["try","this",'fashizzle']) > ['fashizzle']
    '''
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

def frac_non_stop_words(text):
    '''find frac stop words using nltk english stopwords
    frac_stop_words(["try","this",'fashizzle']) > 0.6666
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    content = [word for word in text if word.lower() not in stopwords]
    return len(content) / len(text)

def chat_word_fraction(text):
    '''test_non_vowels = ["grrrr","1234","maybe","butter"] > 0.5
    '''
    chat_words = set(word for word in nltk.corpus.nps_chat.words())
    chat_content = [word for word in text if word.lower() in chat_words]
    return len(chat_content) / len(text)

def non_vowel_tokens(text):
    """emojis grrr cyb3r zzzz
    test_non_vowels = ["grrrr","1234","maybe","butter"]
    non_vowel_tokens(test_non_vowels)
    > ['grrrr', '1234']
    """
    return [word for word in text if re.findall('[^aeiouAEIOU]+$', word)]


#test_tokened_sent = ["I", "bless", "the", "rains", "down", "in", "Africa"]
#print('vocab_size is', vocab_size(test_tokened_sent))
#print('lexical_diversity is', lexical_diversity(test_tokened_sent))
#print('word_to_char_frac is', word_to_char_frac(test_tokened_sent))

def engineered_sentence_features(sentence):
    """ sentence: [w1, w2, ...]
        will use like features(untag(tagged), index) to append"""
    return {
        'is_capitalized': sentence[0].istitle(), # cased chars/titlecased
        'all_caps_sent': len([word for word in sentence if word.isupper()]) == len(sentence),
        'all_caps_word_cnt': len([word for word in sentence if word.isupper()]),
        'is_all_lower': len([word for word in sentence if word.islower()]) == len(sentence),
        'num_chars': char_size(sentence), #sum(len(word) for word in sentence),
        'word_count': len(sentence),
        #'num_sents': len(gutenberg.sents(fileid)),
        'vocab_size': vocab_size(sentence),
        'diversity_score': lexical_diversity(sentence),
        'word_to_char_frac': word_to_char_frac(sentence),
        'frac_unusual': len(unusual_words(sentence)) / len(sentence),
        'frac_non_stop_words': frac_non_stop_words(sentence)
    }


# ##############################################################

## data viz

# ##############################################################

# eval metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def get_metrics(y_test, y_predicted):
    '''metrics to help with evaluating strengths/weaknesses of models'''
    
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, 
                                y_predicted, 
                                pos_label=None,
                                average='weighted')   
    
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, 
                          y_predicted, 
                          pos_label=None,
                          average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, 
                  y_predicted, 
                  pos_label=None, 
                  average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    
    # return metrics
    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    '''gain insight from inspect types of mistakes by models
       guide decision process for model improvement/utilization
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)


def visualize_coefficients(coefficients, feature_names, n_top_features=25):
    # get coefficients with large absolute values
    """coefficients = grid.best_estimator_.named_steps["model_name"].coef_
    """
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = [col_map(1) if c < 0 else col_map(0) for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features),
               feature_names[interesting_coefficients], rotation=60, ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=None, vmax=None)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
    import matplotlib.patches as mpatches
    from sklearn.metrics import f1_score, accuracy_score
    
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (22,14))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[int(j//3), j%3].set_xticks([0.45, 1.45, 2.45])
                ax[int(j//3), j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[int(j//3), j%3].set_xlabel("Training Set Size")
                ax[int(j//3), j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0.0, ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()