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

    df = pd.read_csv('cedele_good.csv')

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

    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

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
    num_topics = 5

    # Build LDA model

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]


    # Visualize the topics
    
    '''LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    LDAvis_prepared'''
    