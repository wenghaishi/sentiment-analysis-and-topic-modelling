from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
result = model(tokens)
result.logits
int(torch.argmax(result.logits))+1

df = pd.read_csv('cedele.csv')

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

sentiment_score(df['combined'].iloc[1])
df['sentiment'] = df['combined'].apply(lambda x: sentiment_score(x[:512]))

df.to_csv('cedele_senti.csv', index = False, encoding='utf-8')