import streamlit as st
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, CategoriesOptions, EmotionOptions, KeywordsOptions, EntitiesOptions

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

# Define hyperparameters
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_size = 35000

data = pd.read_csv('training.csv')

sentences = data['text']
labels = data['label']

# Split the data into training and testing sets
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

model2 = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile the model2
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Load the weights
model2.load_weights('sentiment_model_weights.h5')

def ask(sentance):
    new_sentance = sentance

    new_sentance = tokenizer.texts_to_sequences([new_sentance])
    new_sentance = pad_sequences(new_sentance, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return model2.predict(new_sentance)
print("model loaded")

authenticator = IAMAuthenticator(st.secrets["API"])
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(st.secrets["URL"])


def getEmotions(data):
    emotions = []
    emotions_dict = {}
    emotions_scores = []
    response_emo = natural_language_understanding.analyze(
                    text = data,
                    features = Features( emotion=EmotionOptions())
                    ).get_result() 

    for i in list(response_emo['emotion']['document']['emotion'].keys()):
        emotions.append([i, str(response_emo['emotion']['document']['emotion'][i])])
        emotions_scores.append(str(response_emo['emotion']['document']['emotion'][i]))

    for i in list(response_emo['emotion']['document']['emotion'].keys()):
        emotions_dict[i] = str(response_emo['emotion']['document']['emotion'][i])   

    emotions_scores.sort()

    for e in emotions:
        if emotions_scores[-1] == e[1]:
            max_emo = e[0]    

    return emotions, max_emo, emotions_dict    

def getKeywords(data):
    Keywords = []
    Sentiments = []
    score = 0
    response_keyw = natural_language_understanding.analyze(
                    text = data,
                    features=Features(keywords=KeywordsOptions(sentiment=True, limit=10))).get_result()

    i = 0
    while i < len(response_keyw['keywords']):
        Keywords.append(response_keyw['keywords'][i]['text'])
        Sentiments.append(response_keyw['keywords'][i]['sentiment']['label'])
        score += response_keyw['keywords'][i]['sentiment']['score']
        i+=1
        
    return Keywords, Sentiments, score/(len(Keywords))


st.header("S.W.A.G Tone Analyser")

data = st.text_area('Analyse Your Text')

if st.button('Analyse'):
    emotionsList, max_emotion, emo_dict = getEmotions(data)
    print(data)
    st.success('Your text is analysed!')
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    for i in range(len(emotionsList)):
        a = emotionsList[i][0]
        emotionsList[i][0] = emotionsList[i][1]
        emotionsList[i][1] = a        
    
    emotionsList.sort(reverse=True)

    for i in emotionsList:
        col1.write(i[1])
        col2.write(i[0])

    keywords, sentiments, score = getKeywords(data)
    for i in keywords:
        col3.write(i)
    for j in sentiments:    
        col4.write(j)
    col5.write('Sentiment Score')
    col5.write(score)


    if score < -0.7:
        if max_emotion == 'joy' or emo_dict['joy'] > '0.55':
            col6.write('NOT HATEFUL')    
        elif max_emotion == 'anger' and emo_dict['fear'] > '0.05' and emo_dict['disgust'] > '0.05':
            col6.write('HATEFUL & Inflammatory')
        elif max_emotion == 'sadness' and emo_dict['anger'] > '0.1':
            col6.write('HATEFUL')
        elif max_emotion == 'disgust':
            if emo_dict['disgust'] > '0.2' and emo_dict['anger'] > '0.05':
                col6.write('HATEFUL')
            elif emo_dict['anger'] > '0.1':
                col6.write('HATEFUL') 
    elif emo_dict['disgust'] > '0.17' and emo_dict['anger'] > '0.05':
        col6.write('HATEFUL')
    elif max_emotion == 'sadness' and emo_dict['anger'] > '0.1':
            col6.write('HATEFUL')
    else:
        col6.write('NOT HATEFUL')

    val = ask(data)
    if val > 0.6:
        col7.write("Positive Review 😊"+ str(val))
    elif val < 0.4:
        col7.write("Negative Review ☹️" + str(val))
    else:
        col7.write("Neutral 🙄" + str(val))
