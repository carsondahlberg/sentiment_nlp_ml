# Machine Learning Nanodegree Capstone Project
-----

This is the README.md file for Carson Dahlberg's Capstone Project investigating machine learning/nlp solutions to sentiment analysis.

## Directory Structure
```
capstone_project
│
├── README.md                            <- top-level README for reviewers using this project.
│
├── data
│   ├── combined_raw_data             <- original data combined CSV
│   ├── engineered_sentence_features  <- The final, canonical data sets for modeling CSV
│   └── sentiment labelled sentences  <- The original, immutable data dump DIRECTORY - TXT
│
├── images                            <- figures generated/used in JPG format
│
├── models                            <- Trained/serialized models PKL format, model predictions
│
├── proposal.pdf                      <- completed pre-requisite Capstone Proposal document
├── project_report.pdf                <- Definition, Analysis, Methodology, Results, Conclusion PDF
├── predict_review_sentiment          <- Main Jupyter Notebook models and analysis in Python3 
├── predict_review_sentiment-custom_features <- Jupyter Notebook exploring hand-crafter feature engineering Python2
├── character_ngrams.ipynb            <- Jupyter Notebook exploring character-level n-grams in Python3
│
└── helpers                           <- helper functions for nlp tasks and data viz
```

## Installation

### Requirements
* Python 2 is used in one notebook
* Python version: 3.5.5 | packaged by conda-forge
* NumPy version: 1.13.1
* pandas version: 0.20.3
* seaborn version: 0.8.1
* SciPy version: 0.19.1
* scikit-learn version: 0.19.0
* IPython version: 6.1.0
* matplotlib version: 2.2.2
* contractions version: 0.0.16
* inflect version: 0.3.1
* beautifulsoup4 version: 4.6.0
* nltk version: 3.2.5
* gensim version: 3.4.0
* __IMPORTANT:__ Due to size limitations, the word2vec model *__`GoogleNews-vectors-negative300.bin.gz`__*, used with the *__`Gensim`__* library, will need to be downloaded and installed into the directory *__`./data`__*. You can download Google’s pre-trained model here: *__`https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit`__* I have not included code to import the model, as I have made assumptions that A) you likely already possess the model, or B) you would prefer not to curl some resource from a site before vetting said site


## Data

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

The sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants.
* __Score__ is either 1 (for positive) or 0 (for negative)
* __Sources__ websites/fields: imdb.com, amazon.com, yelp.com
* Format: sentence \t score \n
* For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.
  
For the full datasets look:
* __imdb:__ Maas et. al., 2011 'Learning word vectors for sentiment analysis'
* __amazon:__ McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating * dimensions with review text'
* __yelp:__ Yelp dataset challenge http://www.yelp.com/dataset_challenge