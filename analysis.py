import pandas as pd
import numpy as np
import nltk
import re
import string
import spacy
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import os


# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns

# Read the csv file
df = pd.read_csv('cedele.csv')

nlp=spacy.load("en_core_web_sm")
nlp.pipe_names

