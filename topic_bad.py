
if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import requests
    from bs4 import BeautifulSoup
    import re
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    df = pd.read_csv('cedeleBad.csv')

    df.shape

    # Remove punctuation
    df['combined_processed'] = \
    df['combined'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    df['combined_processed'] = \
    df['combined_processed'].map(lambda x: x.lower())

    # Print out the first rows of papers
    # print(df['combined_processed'].head())

    # Import the wordcloud library
    from wordcloud import WordCloud

    # Join the different processed titles together.
    long_string = ','.join(list(df['combined_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
    data = df.combined_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    #print(data_words[:1][0][:30])

    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    #print(corpus[:1][0][:30])

    from pprint import pprint

    # number of topics
    num_topics = 2

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
